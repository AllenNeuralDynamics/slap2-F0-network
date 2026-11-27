"""Network baselines for GIAnT's registered 80 Hz movies (the localization input).

``slap2-f0 predict-registered MOCO_DIR --ckpt PATH --out DIR`` reads each
registered movie, ``*_REGISTERED_DOWNSAMPLED-80Hz.tif`` or ``.h5`` (channels
interleaved by frame), and its ``*_ALIGNMENTDATA.h5`` in GIAnT's
motion-correction folder, prepares the trial
the way SiLo's ``loadAndProcessTrialAsync`` and ``localizeSources_vIM`` do
(motion-discarded frames from recNegErr, pixels imaged in at least 1 -
nanThresh of the frames), runs the network over it (``predict.predict_trial``)
and writes DIR/<stem>_BASELINE-80Hz.h5:

    baseline   (frame, A, B) float16, counts; MATLAB's h5read gives B x A x
               frame, the orientation of SiLo's movie. NaN outside the valid
               pixels.
    sd         the same for the network's noise SD (with --sd)
    attrs      checkpoint, step, gain, scale, n_discarded, seconds, ...

The network sees the trial in the orientation of the training exports
(MATLAB's rows and columns), so the movie's pages are transposed: TIFF pages,
and the ``/data`` pages of an HDF5 movie (MATLAB ``[rows cols pages]``), read
as (MATLAB col, row). GIAnT-Python writes HDF5 movies by default, MATLAB
TIFF. Without GIAnT's
activity, the gain (variance/mean in counts) comes from the movie alone
(``predict.movie_gain``). The movie, mean image and variance factor are as in
the exporter (``code/matlab/exportF0TrainingData.m``) except that the exporter
also discards frames where most of the 200 Hz samples were discarded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from .predict import load_checkpoint, movie_gain, predict_trial, prepare, tile_frames, valid_index


def _medfilt1_zeropad(x: np.ndarray, n: int) -> np.ndarray:
    """MATLAB medfilt1(x, n) (default zero padding): for even n, the median of
    x(k - n/2 : k + n/2 - 1)."""
    lo = n // 2 if n % 2 == 0 else (n - 1) // 2
    hi = n - 1 - lo
    xp = np.concatenate([np.zeros(lo), x, np.zeros(hi)])
    w = np.lib.stride_tricks.sliding_window_view(xp, n)
    return np.median(w, axis=1)


def _dilate(b: np.ndarray, w: int) -> np.ndarray:
    h = w // 2
    p = np.concatenate([np.zeros(h, bool), b, np.zeros(h, bool)])
    return np.lib.stride_tricks.sliding_window_view(p, w).any(1)


def _erode(b: np.ndarray, w: int) -> np.ndarray:
    h = w // 2
    p = np.concatenate([np.ones(h, bool), b, np.ones(h, bool)])
    return np.lib.stride_tricks.sliding_window_view(p, w).all(1)


def _close(b: np.ndarray, w: int) -> np.ndarray:
    """MATLAB imclose(b, ones(w, 1)): zero padding, dilation, erosion, crop."""
    h = w // 2
    p = np.concatenate([np.zeros(h, bool), b, np.zeros(h, bool)])
    return _erode(_dilate(p, w), w)[h:-h] if h else b.copy()


def discard_frames(rec_neg_err: np.ndarray, align_hz: float = 80.0, motion_thresh: float = 2.0,
                   discard_initial_s: float = 0.0) -> np.ndarray:
    """SiLo's motion-discarded frames (loadAndProcessTrialAsync)."""
    x = np.asarray(rec_neg_err, np.float64).ravel()
    tmp = x - _medfilt1_zeropad(x, round(4 * align_hz))
    tmp = -tmp / min(-0.005, float(np.percentile(tmp, 5, method="hazen")))  # MATLAB prctile
    w = 2 * int(np.ceil(0.025 * align_hz)) + 1
    d = _dilate(tmp > motion_thresh, w) & (tmp > motion_thresh / 2)
    d = _close(d, w)
    d[: int(np.ceil(discard_initial_s * align_hz))] = True
    return d


class ArrayTrial:
    """A trial held in memory with the attributes ``predict_trial`` reads
    (as ``io.TrialFile``): movie (frame, row, col) in counts, NaN where not imaged."""

    def __init__(self, movie: np.ndarray, valid: np.ndarray, discard: np.ndarray, varfac: np.ndarray):
        self.n_frames, self.n_rows, self.n_cols = movie.shape
        self.channel = 0
        self._f = {"movie": movie[None]}
        self.valid, self.discard = valid, discard
        ok = ~discard
        with np.errstate(invalid="ignore", divide="ignore"):
            self.mean_image = np.where(valid, np.nanmean(movie[ok], 0), np.nan).astype(np.float32)
        good = valid & np.isfinite(self.mean_image)
        self.scale = float(np.nanmedian(self.mean_image[good])) if good.any() else 1.0
        vf = np.nan_to_num(varfac.astype(np.float32))
        okv = valid & (vf > 0)
        self.varfac_ref = float(np.median(vf[okv])) if okv.any() else 1.0
        self.varfac = vf / np.float32(self.varfac_ref)


def movie_stats(movie: np.ndarray, valid: np.ndarray, discard: np.ndarray, block: int = 32) -> dict:
    """Per-pixel mean and variance over the kept frames (the network's variance
    map input) and the movie-only gain."""
    T, R, C = movie.shape
    mean = np.full((R, C), np.nan, np.float32)
    var = np.full((R, C), np.nan, np.float32)
    keep = np.nonzero(~discard)[0]
    for r0 in range(0, R, block):
        m = movie[keep, r0:r0 + block].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean[r0:r0 + block] = np.nanmean(m, 0)
            var[r0:r0 + block] = np.nanvar(m, 0)
    mean[~valid] = np.nan
    var[~valid] = np.nan
    st = {"mean_movie": mean, "var_movie": var, "gain": 1.0}
    st["gain"] = movie_gain(st, valid)
    return st


def _read_channel(path: Path, n_ch: int, channel: int) -> np.ndarray:
    """One channel's pages of a registered TIFF or HDF5 movie, as stored:
    (frame, MATLAB col, MATLAB row)."""
    if path.suffix == ".h5":
        with h5py.File(path, "r") as f:
            data = f[next(iter(f.keys()))]  # /data: pages x MATLAB col x row
            n = data.shape[0] // n_ch
            return data[channel:n * n_ch:n_ch].astype(np.float32, copy=False)
    import tifffile

    with tifffile.TiffFile(path) as t:
        n = len(t.pages) // n_ch
        return t.asarray(key=range(channel, n * n_ch, n_ch)).astype(np.float32, copy=False)


def registered_movies(moco: Path) -> list[Path]:
    """The registered movies of a motion-correction folder, one per trial:
    ``.tif`` or ``.h5``, the newer file where a trial has both."""
    by_stem: dict[str, Path] = {}
    for path in sorted(moco.glob("*_REGISTERED_DOWNSAMPLED-80Hz.tif")) + sorted(
            moco.glob("*_REGISTERED_DOWNSAMPLED-80Hz.h5")):
        stem = path.name[: path.name.index("_REGISTERED")]
        old = by_stem.get(stem)
        if old is None or path.stat().st_mtime > old.stat().st_mtime:
            by_stem[stem] = path
    return [by_stem[k] for k in sorted(by_stem)]


def load_registered(movie_path: Path, adata: Path, channel: int = 0, nan_thresh: float = 0.33,
                    motion_thresh: float = 2.0, discard_initial_s: float = 0.0):
    """The activity channel of a registered movie (TIFF or HDF5) in training
    orientation (frame, MATLAB row, MATLAB col), with SiLo's discards and
    valid pixels."""
    with h5py.File(adata, "r") as f:
        n_ch = int(np.asarray(f["numChannels"]).ravel()[0])
        align_hz = float(np.asarray(f["alignHz"]).ravel()[0])
        rne = np.asarray(f["recNegErr"]).ravel()
        vf = np.nanmean(f["slap2/varFacDS"][:], axis=0) if "slap2/varFacDS" in f else None
    movie = _read_channel(Path(movie_path), n_ch, channel)
    n = movie.shape[0]
    movie = np.ascontiguousarray(movie.transpose(0, 2, 1))  # pages are (MATLAB col, row)
    discard = discard_frames(rne[:n], align_hz, motion_thresh, discard_initial_s)
    movie[discard] = np.nan
    valid = np.isnan(movie).mean(0) < nan_thresh
    movie[:, ~valid] = np.nan
    varfac = vf.T if vf is not None else np.ones(valid.shape, np.float32)
    return movie, valid, discard, varfac, align_hz


def _peak_rss_gb() -> float | None:
    """Peak resident memory of this process so far (Linux), or None."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _report(path: str, entry: dict) -> None:
    """Append one JSON line to the --report file."""
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def run(args: argparse.Namespace) -> None:
    moco, out = Path(args.moco_dir), Path(args.out)
    movies = registered_movies(moco)
    if args.files:
        keys = [k.strip() for k in args.files.split(",") if k.strip()]
        movies = [m for m in movies if any(k in m.name for k in keys)]
    if not movies:
        raise SystemExit(f"no *_REGISTERED_DOWNSAMPLED-80Hz.tif or .h5 in {moco}")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, wcfg, ck = load_checkpoint(args.ckpt, device)
    amp_dtype = prepare(model, device, args.precision)
    print(f"{args.ckpt} (step {ck.get('step')}, box radius {model.cfg.box_radius}, varfac {model.cfg.varfac_norm}) "
          f"on {device}, {next(model.parameters()).dtype}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    for movie_path in movies:
        stem = movie_path.name[: movie_path.name.index("_REGISTERED")]
        dest = out / f"{stem}_BASELINE-80Hz.h5"
        if dest.exists() and not args.overwrite:
            print(f"{dest.name} exists, skipped", flush=True)
            _report(args.report, {"stem": stem, "movie": movie_path.name, "skipped": True})
            continue
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        movie, valid, discard, varfac, hz = load_registered(
            movie_path, moco / f"{stem}_ALIGNMENTDATA.h5", args.channel - 1, args.nan_thresh, args.motion_thresh,
            args.discard_initial_s)
        t1 = time.time()
        tr = ArrayTrial(movie, valid, discard, varfac)
        st = movie_stats(movie, valid, discard)
        t2 = time.time()
        rows, cols = valid_index(tr)
        # the tile size predict_trial would choose, made explicit for the report
        tile = args.out_frames if args.out_frames > 0 else tile_frames(
            model, wcfg, movie.shape[0], device, amp_dtype, wcfg.out_px)
        mu, var = predict_trial(model, wcfg, tr, st, device, amp_dtype, tile, None, 1, rows, cols)
        t3 = time.time()
        T, R, C = movie.shape
        del movie
        full = np.full((T, R, C), np.nan, np.float16)
        full[:, rows, cols] = mu.astype(np.float16)
        tmp = dest.with_suffix(".h5.part")
        with h5py.File(tmp, "w") as f:
            kw = dict(compression="gzip", compression_opts=1, shuffle=True, chunks=(min(256, T), min(64, C), min(64, R)))
            f.create_dataset("baseline", data=full.transpose(0, 2, 1), **kw)
            if args.sd:
                full[:, rows, cols] = np.sqrt(np.maximum(var, 0)).astype(np.float16)
                f.create_dataset("sd", data=full.transpose(0, 2, 1), **kw)
            f.attrs.update({"checkpoint": str(args.ckpt), "step": int(ck.get("step") or 0),
                            "box_radius": int(model.cfg.box_radius), "gain": float(st["gain"]), "scale": tr.scale,
                            "align_hz": hz, "n_frames": T, "n_valid": int(valid.sum()), "n_discarded": int(discard.sum()),
                            "source": movie_path.name, "precision": str(next(model.parameters()).dtype),
                            "seconds_load": t1 - t0, "seconds_stats": t2 - t1, "seconds_network": t3 - t2})
        tmp.replace(dest)
        t4 = time.time()
        print(f"{stem}: {T} frames, {R}x{C}, {int(valid.sum())} valid px, {int(discard.sum())} discarded; "
              f"load {t1 - t0:.0f} s, stats {t2 - t1:.0f} s, network {t3 - t2:.0f} s, write {t4 - t3:.0f} s",
              flush=True)
        _report(args.report, {
            "stem": stem, "movie": movie_path.name, "baseline": dest.name, "frames": T, "rows": R, "cols": C,
            "valid_px": int(valid.sum()), "discarded": int(discard.sum()), "out_frames": int(tile),
            "load_s": t1 - t0, "stats_s": t2 - t1, "network_s": t3 - t2, "write_s": t4 - t3,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
            "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else None,
            "peak_rss_gb_so_far": _peak_rss_gb(), "device": str(device)})
    print(f"done in {time.time() - t_all:.0f} s", flush=True)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("moco_dir", help="GIAnT motion_correction folder")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--files", default="", help="comma-separated substrings of the movie names to include")
    p.add_argument("--channel", type=int, default=1, help="activity channel, 1-based (SiLo's activityChannel)")
    p.add_argument("--nan_thresh", type=float, default=0.33)
    p.add_argument("--motion_thresh", type=float, default=2.0)
    p.add_argument("--discard_initial_s", type=float, default=0.0)
    p.add_argument("--out_frames", type=int, default=0)
    p.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16", "amp", "fp32"])
    p.add_argument("--sd", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--report", default="", help="append one JSON line per movie to this file: times per step "
                                              "(load, stats, network, write), tile size, peak GPU and host memory")
