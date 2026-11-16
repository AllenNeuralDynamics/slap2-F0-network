"""Full-trial baselines from saved checkpoints, for viewing and comparison.

``slap2-f0 predict ROOT --ckpt name=path [name=path ...] --out DIR`` runs each
checkpoint over whole trials and writes one HDF5 file per trial,
DIR/<session>/<path>/<trial>.h5 (session as in training: the trial's label,
else its grandparent folder), which ``slap2-f0 view`` displays.

Tiling: each output tile is ``out_px`` x ``out_px`` pixels (by default the
checkpoint's training size) and ``out_frames`` frames, and its input window
adds the model's reach on every side, read from the trial with NaN past its
edges, exactly as in training. Tiles start at multiples of ``out_px`` (keep
it even: the half-resolution paths pair rows and columns from the window's
corner), so a slab model's output does not depend on the tiling. A box
model's pooled features with half-widths beyond its reach (32 and the whole
window by default) see only the part inside the tile's window, as in
training; with the training tile size every pixel sees the same window
geometry as in training, and seams can show where those features change.
The variance-factor input is relative to its median over the window, with
unimaged pixels counted as 0 (``make_inputs``), which also depends on the
window; the training tile size keeps that as in training too.

SiLo's baseline (``silo``: an 8-frame moving mean, then a 321-frame moving
median, not blind) is added unless ``--no_silo``.

Output file (arrays over the trial's valid pixels, N of them, in the order of
``rows``/``cols``):

    rows, cols                 (N,) int32   pixel positions in the trial grid
    mask/valid, mask/labeled, mask/footprint_support   (row, col) uint8
    mean_image                 (row, col) float32, counts
    frame/discard              (frame,) uint8
    movie, activity            (frame, N) float16, counts (movie NaN on
                               discarded frames); omitted with --no_movie
    baseline/<name>            (frame, N) float16, counts
    sd/<name>                  (frame, N) float16, noise SD in counts (--sd)

Root attributes: ``slap2_f0_predictions`` (format version), session, path,
trial, source (file relative to ROOT), n_frames, n_rows, n_cols, scale, gain,
frame_rate, n_sources and ``models`` (names in the order given). Each
baseline dataset carries the checkpoint path, its training step and its kind
(``network`` or ``silo``). Several files for the same trial (say, a later
run with one more checkpoint and ``--no_movie``) are merged by the viewer.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import WindowConfig
from .io import TrialFile, list_trials
from .model import ModelConfig, SlabBlindSpotNet, make_inputs
from .stats import load_stats
from .train import silo_baseline

FORMAT_VERSION = 1
FRAME_RATE = 80.0  # exported trials are on an 80 Hz grid (see io.py)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[SlabBlindSpotNet, WindowConfig, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = SlabBlindSpotNet(ModelConfig(**ck["config"])).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, WindowConfig(**ck["window"]), ck


def valid_index(tr: TrialFile) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(tr.valid)
    return rows.astype(np.int32), cols.astype(np.int32)


def read_valid(tr: TrialFile, rows: np.ndarray, cols: np.ndarray, block: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Movie and activity at the valid pixels, (frame, N) float32 in counts;
    the movie is NaN on discarded frames. Read in row blocks, so a large
    trial never needs a full (frame, row, col) array."""
    T = tr.n_frames
    movie = np.empty((T, len(rows)), np.float32)
    act = np.empty((T, len(rows)), np.float32)
    for r0 in range(0, tr.n_rows, block):
        r1 = min(r0 + block, tr.n_rows)
        k = np.nonzero((rows >= r0) & (rows < r1))[0]
        if not len(k):
            continue
        m = tr._f["movie"][tr.channel, :, r0:r1, :]
        a = tr._f["activity"][tr.channel, :, r0:r1, :]
        movie[:, k] = m[:, rows[k] - r0, cols[k]]
        act[:, k] = a[:, rows[k] - r0, cols[k]]
    movie[tr.discard] = np.nan
    return movie, act


@torch.no_grad()
def predict_trial(model: SlabBlindSpotNet, wcfg: WindowConfig, tr: TrialFile, st: dict, device: torch.device,
                  amp_dtype=None, out_frames: int = 512, out_px: int | None = None, batch: int = 4,
                  rows: np.ndarray | None = None, cols: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Baseline and noise variance of the whole trial at its valid pixels:
    (frame, N) float32 in counts and counts^2."""
    if rows is None or cols is None:
        rows, cols = valid_index(tr)
    T, R, C = tr.n_frames, tr.n_rows, tr.n_cols
    tR, sR = wcfg.t_reach, wcfg.s_reach
    op = out_px or wcfg.out_px
    nt, ns = out_frames + 2 * tR, op + 2 * sR
    index = np.full((R, C), -1, np.int64)
    index[rows, cols] = np.arange(len(rows))
    mu_out = np.full((T, len(rows)), np.nan, np.float32)
    var_out = np.full((T, len(rows)), np.nan, np.float32)
    region = (slice(tR, tR + out_frames), slice(sR, sR + op), slice(sR, sR + op))
    s = np.float32(tr.scale)
    # whole-trial maps, padded so every tile's window is a plain slice; as in TrialFile.read
    pr0, pr1, pc0, pc1 = sR, sR + op, sR, sR + op
    mean_p = np.full((R + pr0 + pr1, C + pc0 + pc1), np.nan, np.float32)
    mean_p[pr0:pr0 + R, pc0:pc0 + C] = np.where(tr.valid, tr.mean_image, np.nan)
    vf_p = np.zeros_like(mean_p)
    vf_p[pr0:pr0 + R, pc0:pc0 + C] = np.nan_to_num(tr.varfac)
    var_p = np.full_like(mean_p, np.nan)
    var_p[pr0:pr0 + R, pc0:pc0 + C] = st["var_movie"]
    var_p = np.where(np.isfinite(mean_p), var_p, np.nan).astype(np.float32)
    pt0, pt1 = tR, tR + out_frames
    for r0 in range(0, R, op):  # one band of rows at a time: each compressed chunk is read about twice
        cs = [c0 for c0 in range(0, C, op) if (index[r0:r0 + op, c0:c0 + op] >= 0).any()]
        if not cs:
            continue
        band = np.full((T + pt0 + pt1, ns, C + pc0 + pc1), np.nan, np.float32)
        rs, re = max(r0 - sR, 0), min(r0 - sR + ns, R)
        m = tr._f["movie"][tr.channel, :, rs:re, :]
        m[:, ~tr.valid[rs:re]] = np.nan
        m[tr.discard] = np.nan
        band[pt0:pt0 + T, rs - (r0 - sR): re - (r0 - sR), pc0:pc0 + C] = m
        del m
        tiles = [(t0, c0) for c0 in cs for t0 in range(0, T, out_frames)]
        def infer(chunk):
            """mu and var of a list of tiles; halves the list when the GPU runs out of memory."""
            # padded coordinates: frame t0 - tR -> t0, column c0 - sR -> c0, row r0 - sR -> r0
            sp = lambda a, c0: a[r0:r0 + ns, c0:c0 + ns]  # noqa: E731
            try:
                movie = np.stack([band[t0:t0 + nt, :, c0:c0 + ns] for t0, c0 in chunk])
                scale = torch.full((len(chunk),), float(s), device=device)
                gain = torch.full((len(chunk),), float(st["gain"]), device=device)
                x = make_inputs(torch.from_numpy(movie).to(device),
                                torch.from_numpy(np.stack([sp(mean_p, c0) for _, c0 in chunk])).to(device),
                                torch.from_numpy(np.stack([sp(vf_p, c0) for _, c0 in chunk])).to(device), scale,
                                torch.from_numpy(np.stack([sp(var_p, c0) for _, c0 in chunk])).to(device), gain)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    mu, var = model(x, scale, gain, region)
                return (mu.float() * s).cpu().numpy(), (var.float() * s * s).cpu().numpy()
            except torch.OutOfMemoryError:
                if len(chunk) == 1:
                    raise SystemExit("out of GPU memory with one tile per pass; try a smaller --out_frames")
                x = mu = var = None
                torch.cuda.empty_cache()
                h = len(chunk) // 2
                (m1, v1), (m2, v2) = infer(chunk[:h]), infer(chunk[h:])
                return np.concatenate([m1, m2]), np.concatenate([v1, v2])

        for i in range(0, len(tiles), batch):
            chunk = tiles[i:i + batch]
            mu, var = infer(chunk)
            for k, (t0, c0) in enumerate(chunk):
                t1 = min(t0 + out_frames, T)
                idx = index[r0:r0 + op, c0:c0 + op]
                ok = idx >= 0
                mu_out[t0:t1, idx[ok]] = mu[k, :t1 - t0, :idx.shape[0], :idx.shape[1]][:, ok]
                var_out[t0:t1, idx[ok]] = var[k, :t1 - t0, :idx.shape[0], :idx.shape[1]][:, ok]
    return mu_out, var_out


@torch.no_grad()
def silo_trial(movie: np.ndarray, device: torch.device, chunk: int = 256) -> np.ndarray:
    """SiLo's baseline of (frame, N) movie values, in the movie's units."""
    T, N = movie.shape
    out = np.empty((T, N), np.float32)
    one = torch.ones(1, device=device)
    for k0 in range(0, N, chunk):
        m = torch.from_numpy(movie[:, k0:k0 + chunk]).to(device)[None, :, :, None]
        b = silo_baseline(m, one, (slice(0, T), slice(0, m.shape[2]), slice(0, 1)))
        out[:, k0:k0 + chunk] = b[0, :, :, 0].cpu().numpy()
    return out


def _f16(a: np.ndarray) -> np.ndarray:
    return np.clip(a, -65000, 65000).astype(np.float16)


def _write(f: h5py.File, name: str, a: np.ndarray, **attrs) -> None:
    if name in f:
        del f[name]
    N = a.shape[1]
    d = f.create_dataset(name, data=_f16(a), chunks=(min(64, a.shape[0]), max(1, min(N, 4096))),
                         compression="gzip", compression_opts=1, shuffle=True)
    for k, v in attrs.items():
        d.attrs[k] = v


def run(args: argparse.Namespace) -> None:
    root, out = Path(args.root), Path(args.out)
    files = list_trials(root)
    if args.files:
        keys = [k.strip().replace("\\", "/") for k in args.files.split(",") if k.strip()]
        files = [f for f in files if any(k in f.relative_to(root).as_posix() for k in keys)]
    if not files:
        raise SystemExit(f"no trial files under {root}" + (f" matching {args.files}" if args.files else ""))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = None
    if device.type == "cuda" and not args.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stats = load_stats(files, args.stats_cache or (out / "stats"), workers=args.workers)
    print(f"stats for {len(files)} trials in {time.time() - t0:.0f} s; device {device} (amp {amp_dtype})", flush=True)
    specs = [(s.split("=", 1) if "=" in s else (Path(s).stem, s)) for s in (args.ckpt or [])]
    models = []
    for name, path in specs:
        model, wcfg, ck = load_checkpoint(path, device)
        models.append((name, path, model, wcfg, ck))
        print(f"{name}: {path} (step {ck.get('step')}, box radius {model.cfg.box_radius})", flush=True)
    for fn in files:
        rel = fn.relative_to(root)
        with TrialFile(fn) as tr:
            session = str(tr.attrs.get("label", fn.parent.parent.name))
        dest = out / session / fn.parent.name / fn.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with TrialFile(fn) as tr, h5py.File(dest, "a") as f:
            st = stats[str(fn)]
            rows, cols = valid_index(tr)
            f.attrs.update({"slap2_f0_predictions": FORMAT_VERSION, "session": session,
                            "path": fn.parent.name, "trial": fn.stem,
                            "source": rel.as_posix(), "n_frames": tr.n_frames, "n_rows": tr.n_rows,
                            "n_cols": tr.n_cols, "scale": tr.scale, "gain": float(st["gain"]),
                            "frame_rate": FRAME_RATE, "n_sources": int(tr.attrs.get("nSources", 0) or 0)})
            for name, a in (("rows", rows), ("cols", cols), ("mask/valid", tr.valid), ("mask/labeled", tr.labeled),
                            ("mask/footprint_support", tr.support), ("mean_image", tr.mean_image),
                            ("frame/discard", tr.discard)):
                if name in f:
                    del f[name]
                f.create_dataset(name, data=a.astype(np.uint8) if a.dtype == bool else a)
            movie = None
            if not args.no_movie or not args.no_silo:
                t1 = time.time()
                movie, act = read_valid(tr, rows, cols)
                if not args.no_movie:
                    _write(f, "movie", movie)
                    _write(f, "activity", act)
                print(f"{session}/{fn.parent.name}/{fn.stem}: {tr.n_frames} frames, {len(rows)} valid px read in {time.time() - t1:.0f} s", flush=True)
            names = list(f.attrs.get("models", []))
            for name, path, model, wcfg, ck in models:
                t1 = time.time()
                mu, var = predict_trial(model, wcfg, tr, st, device, amp_dtype, args.out_frames, args.out_px,
                                        args.batch, rows, cols)
                _write(f, f"baseline/{name}", mu, checkpoint=str(path), step=int(ck.get("step") or 0), kind="network",
                       box_radius=int(model.cfg.box_radius))
                if args.sd:
                    _write(f, f"sd/{name}", np.sqrt(np.maximum(var, 0)))
                names = [n for n in names if n != name] + [name]
                print(f"  {name}: {time.time() - t1:.0f} s", flush=True)
            if not args.no_silo:
                t1 = time.time()
                _write(f, "baseline/silo", silo_trial(movie, device), kind="silo",
                       checkpoint="8-frame moving mean, then 321-frame moving median")
                names = [n for n in names if n != "silo"] + ["silo"]
                print(f"  silo: {time.time() - t1:.0f} s", flush=True)
            f.attrs["models"] = np.array(names, dtype=h5py.string_dtype())
    print(f"done in {time.time() - t0:.0f} s; view with: slap2-f0 view {out}", flush=True)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("root", help="folder with exported trial files (any depth)")
    p.add_argument("--ckpt", nargs="*", default=[], help="name=path of each checkpoint, best first")
    p.add_argument("--out", required=True, help="output folder (one file per trial, same relative paths)")
    p.add_argument("--files", default="", help="comma-separated parts of trial paths to keep "
                                               "(e.g. 780125_2025-03-14_09-30-06/Path1/trial_002); default all")
    p.add_argument("--out_frames", type=int, default=512, help="output frames per tile")
    p.add_argument("--out_px", type=int, default=0, help="output pixels per tile side; 0 = the checkpoint's training size")
    p.add_argument("--batch", type=int, default=1, help="tiles per forward pass (halved when the GPU runs out of memory)")
    p.add_argument("--no_silo", action="store_true", help="skip SiLo's baseline")
    p.add_argument("--no_movie", action="store_true", help="skip the movie and activity (when adding checkpoints)")
    p.add_argument("--sd", action="store_true", help="also save each network's noise SD")
    p.add_argument("--stats_cache", default="", help="trial stats cache folder (default <out>/stats)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--no_amp", action="store_true")
