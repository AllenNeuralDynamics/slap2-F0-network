"""Full-trial baselines from saved checkpoints, for viewing and comparison.

``slap2-f0 predict ROOT --ckpt name=path [name=path ...] --out DIR`` runs each
checkpoint (or ``name=avg:SPEC;SPEC;...``, a weighted average of checkpoints
made in memory, as in ``evaluate``) over whole trials and writes one HDF5 file per trial,
DIR/<session>/<path>/<trial>.h5 (session as in training: the trial's label,
else its grandparent folder), which ``slap2-f0 view`` displays.

Tiling: each output tile is ``out_px`` x ``out_px`` pixels (by default the
checkpoint's training size) and ``out_frames`` frames (by default the whole
trial, or as many frames as fit in GPU memory: the temporal reach on both
sides then costs least), and its input window adds the model's reach on every
side, read from the trial with NaN past its edges, exactly as in training.
Tiles are placed to cover the valid pixels with few tiles (``tile_origins``),
at even rows and columns (the half-resolution paths pair rows and columns
from the window's corner), so a slab model's output does not depend on the
tiling. A box
model's pooled features with half-widths beyond its reach (32 and the whole
window by default) see only the part inside the tile's window, as in
training; with the training tile size every pixel sees the same window
geometry as in training, and seams can show where those features change.
Checkpoints up to v6 take the variance-factor input relative to its median
over the window, with unimaged pixels counted as 0 (``ModelConfig.varfac_norm``
"window"), which also depends on the window; the training tile size keeps that
as in training too. Later checkpoints use the trial's median.

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
from .model import SlabBlindSpotNet, config_from_dict, make_inputs
from .stats import load_stats
from .train import averaged_checkpoint, silo_baseline

FORMAT_VERSION = 1
FRAME_RATE = 80.0  # exported trials are on an 80 Hz grid (see io.py)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[SlabBlindSpotNet, WindowConfig, dict]:
    """A saved checkpoint, or ``avg:SPEC;SPEC;...``: a weighted average of
    checkpoints made in memory (``train.averaged_checkpoint``, as in
    ``evaluate``), e.g. a WiSE-FT blend of a fine-tuning run's kept steps."""
    if str(path).startswith("avg:"):
        ck = averaged_checkpoint(str(path)[4:])
    else:
        ck = torch.load(path, map_location=device, weights_only=False)
    model = SlabBlindSpotNet(config_from_dict(ck["config"])).to(device).eval()
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


def tile_origins(want: np.ndarray, op: int) -> list[tuple[int, list[int]]]:
    """Tiles of op x op pixels covering the pixels in `want` (row, col): bands
    of rows, each starting at the first row not yet covered, and in each band
    tiles starting at the first column not yet covered (the fewest tiles for
    the band). Origins are even, as the half-resolution paths need (they pair
    rows and columns from the window's corner). Returns [(r0, [c0, ...])]."""
    rows = np.nonzero(want.any(1))[0]
    out, r = [], 0
    for rr in rows:
        if rr < r:
            continue
        r0 = int(rr - rr % 2)
        r = r0 + op
        cs, c = [], 0
        for cc in np.nonzero(want[r0:r].any(0))[0]:
            if cc >= c:
                c0 = int(cc - cc % 2)
                cs.append(c0)
                c = c0 + op
        out.append((r0, cs))
    return out


def _is_oom(e: BaseException) -> bool:
    # the CUDA driver can raise a plain RuntimeError
    return isinstance(e, torch.OutOfMemoryError) or (isinstance(e, RuntimeError) and "out of memory" in str(e))


@torch.no_grad()
def tile_frames(model: SlabBlindSpotNet, wcfg: WindowConfig, T: int, device: torch.device, amp_dtype=None,
                out_px: int | None = None, fraction: float = 0.85, margin: float = 1.1) -> int:
    """Output frames per tile for ``predict_trial``: the whole trial if one
    tile of it fits in `fraction` of the free GPU memory, else the trial split
    into equal tiles that do. Peak memory is measured on tiles of 64 and 320
    output frames and extrapolated linearly with a `margin` on the slope (the
    peak grows faster for long tiles, when the layers that run on the output
    frames only dominate). Also caps this process at 95% of the GPU's memory,
    so that running out raises the allocator's error, which predict_trial
    handles by splitting the tile, rather than failing in the driver (after
    which the allocator can be left broken). 512 on the CPU."""
    if device.type != "cuda":
        return min(T, 512)
    torch.cuda.set_per_process_memory_fraction(0.95)  # the current device
    tR, sR = wcfg.t_reach, wcfg.s_reach
    op = out_px or wcfg.out_px
    ns = op + 2 * sR
    sizes = (64, 320)
    peaks = []
    for nf in sizes:
        x = torch.rand(1, model.cfg.n_inputs, nf + 2 * tR, ns, ns, device=device) + 0.5
        x[:, 1] = 1
        one = torch.ones(1, device=device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            model(x, one, 10 * one, (slice(tR, tR + nf), slice(sR, sR + op), slice(sR, sR + op)))
        peaks.append(torch.cuda.max_memory_allocated() - base)
        del x
    slope = margin * max((peaks[1] - peaks[0]) / (sizes[1] - sizes[0]), 1.0)
    fixed = max(peaks[1] - slope * (sizes[1] + 2 * tR), 0.0)
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info()[0]  # the current device (torch 2.4 wants an index, not "cuda")
    most = int((fraction * free - fixed) / slope) - 2 * tR
    if most >= T:
        return T
    if most < 32:
        raise SystemExit(f"a tile does not fit in GPU memory ({free / 2**30:.1f} GB free)")
    n = -(-T // most)
    return -(-T // n)


@torch.no_grad()
def predict_trial(model: SlabBlindSpotNet, wcfg: WindowConfig, tr: TrialFile, st: dict, device: torch.device,
                  amp_dtype=None, out_frames: int = 0, out_px: int | None = None, batch: int = 1,
                  rows: np.ndarray | None = None, cols: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Baseline and noise variance of the whole trial at its valid pixels:
    (frame, N) float32 in counts and counts^2. out_frames 0 = as long as fits
    (``tile_frames``); a tile that runs out of GPU memory is split in two in
    time (a batch of tiles is first split into single tiles)."""
    if rows is None or cols is None:
        rows, cols = valid_index(tr)
    T, R, C = tr.n_frames, tr.n_rows, tr.n_cols
    tR, sR = wcfg.t_reach, wcfg.s_reach
    op = out_px or wcfg.out_px
    if out_frames <= 0:
        out_frames = tile_frames(model, wcfg, T, device, amp_dtype, op)
    ns = op + 2 * sR
    index = np.full((R, C), -1, np.int64)
    index[rows, cols] = np.arange(len(rows))
    mu_out = np.full((T, len(rows)), np.nan, np.float32)
    var_out = np.full((T, len(rows)), np.nan, np.float32)
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
    for r0, cs in tile_origins(index >= 0, op):  # one band of rows at a time: each compressed chunk is read about twice
        band = np.full((T + pt0 + pt1, ns, C + pc0 + pc1), np.nan, np.float32)
        rs, re = max(r0 - sR, 0), min(r0 - sR + ns, R)
        m = tr._f["movie"][tr.channel, :, rs:re, :]
        m[:, ~tr.valid[rs:re]] = np.nan
        m[tr.discard] = np.nan
        band[pt0:pt0 + T, rs - (r0 - sR): re - (r0 - sR), pc0:pc0 + C] = m
        del m
        sp = lambda a, c0: a[r0:r0 + ns, c0:c0 + ns]  # noqa: E731

        def infer(chunk):
            """mu and var of a list of tiles (t0, frames, c0) of equal length, each
            (tiles, frames, out_px, out_px); on GPU out-of-memory, smaller pieces."""
            # padded coordinates: frame t0 - tR -> t0, column c0 - sR -> c0, row r0 - sR -> r0
            nf = chunk[0][1]
            try:
                movie = np.stack([band[t0:t0 + nf + 2 * tR, :, c0:c0 + ns] for t0, _, c0 in chunk])
                scale = torch.full((len(chunk),), float(s), device=device)
                gain = torch.full((len(chunk),), float(st["gain"]), device=device)
                x = make_inputs(torch.from_numpy(movie).to(device),
                                torch.from_numpy(np.stack([sp(mean_p, c0) for *_, c0 in chunk])).to(device),
                                torch.from_numpy(np.stack([sp(vf_p, c0) for *_, c0 in chunk])).to(device), scale,
                                torch.from_numpy(np.stack([sp(var_p, c0) for *_, c0 in chunk])).to(device), gain,
                                varfac_norm=model.cfg.varfac_norm)
                del movie
                region = (slice(tR, tR + nf), slice(sR, sR + op), slice(sR, sR + op))
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    mu, var = model(x, scale, gain, region)
                return (mu.float() * s).cpu().numpy(), (var.float() * s * s).cpu().numpy()
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if not _is_oom(e):
                    raise
            # retry outside the except block: the exception's traceback holds the
            # failed pass's tensors, so memory is only freed once it is gone
            x = mu = var = None
            torch.cuda.empty_cache()
            if len(chunk) > 1:
                h = len(chunk) // 2
                (m1, v1), (m2, v2) = infer(chunk[:h]), infer(chunk[h:])
                return np.concatenate([m1, m2]), np.concatenate([v1, v2])
            if nf < 64:
                raise SystemExit("out of GPU memory with a 32-frame tile")
            (t0, _, c0), h = chunk[0], nf // 2
            (m1, v1), (m2, v2) = infer([(t0, h, c0)]), infer([(t0 + h, nf - h, c0)])
            return np.concatenate([m1, m2], 1), np.concatenate([v1, v2], 1)

        tiles = [(t0, out_frames, c0) for c0 in cs for t0 in range(0, T, out_frames)]
        for i in range(0, len(tiles), batch):
            chunk = tiles[i:i + batch]
            mu, var = infer(chunk)
            for k, (t0, nf, c0) in enumerate(chunk):
                t1 = min(t0 + nf, T)
                idx = index[r0:r0 + op, c0:c0 + op]
                ok = idx >= 0
                mu_out[t0:t1, idx[ok]] = mu[k, :t1 - t0, :idx.shape[0], :idx.shape[1]][:, ok]
                var_out[t0:t1, idx[ok]] = var[k, :t1 - t0, :idx.shape[0], :idx.shape[1]][:, ok]
    return mu_out, var_out


def movie_gain(st: dict, valid: np.ndarray) -> float:
    """The trial's variance/mean ratio from the movie alone: the median over
    the dimmer half of the valid pixels. The stats' gain is measured on movie -
    activity at pixels without fitted sources, which exist only after GIAnT's
    extraction; on the five sessions' viewer trials this is within 3% of it
    (events inflate the variance mostly on brighter pixels)."""
    m, v = st["mean_movie"], st["var_movie"]
    ok = valid & np.isfinite(m) & np.isfinite(v) & (m > 0)
    if not ok.any():
        return float(st["gain"])
    m, v = m[ok], v[ok]
    dim = m <= np.median(m)
    return float(np.median(v[dim] / m[dim]))


def prepare(model: SlabBlindSpotNet, device: torch.device, precision: str = "auto"):
    """Cast the model for inference and return the autocast dtype: "fp16"
    (the default on GPU) and "bf16" run the whole model in that precision
    (fp16 is closer to fp32: about 3e-4 relative at the 99th percentile on a
    real trial, against 3e-3 for bf16), "amp" is bf16 autocast over float32
    weights (the residual stream then stays float32), "fp32" none."""
    if precision == "auto":
        precision = "fp16" if device.type == "cuda" else "fp32"
    if precision in ("fp16", "bf16"):
        dt = torch.float16 if precision == "fp16" else torch.bfloat16
        model.to(dtype=dt)
        return dt
    if precision == "amp":
        return torch.bfloat16 if device.type != "cuda" or torch.cuda.is_bf16_supported() else torch.float16
    return None


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
    precision = "fp32" if args.no_amp else args.precision
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stats = load_stats(files, args.stats_cache or (out / "stats"), workers=args.workers)
    print(f"stats for {len(files)} trials in {time.time() - t0:.0f} s; device {device}", flush=True)
    specs, n_avg = [], 0
    for s in args.ckpt or []:
        if s.startswith("avg:"):  # unnamed average (avg1, avg2, ...): its weights' "=" are not a name
            n_avg += 1
            specs.append((f"avg{n_avg}", s))
        else:
            specs.append(tuple(s.split("=", 1)) if "=" in s else (Path(s).stem, s))
    models = []
    for name, path in specs:
        model, wcfg, ck = load_checkpoint(path, device)
        amp_dtype = prepare(model, device, precision)
        if args.compile:
            cfg = model.cfg
            model = torch.compile(model)
            model.cfg = cfg
        models.append((name, path, model, wcfg, ck, amp_dtype))
        print(f"{name}: {path} (step {ck.get('step')}, box radius {model.cfg.box_radius}, "
              f"{next(model.parameters()).dtype}, autocast {amp_dtype})", flush=True)
    for fn in files:
        rel = fn.relative_to(root)
        with TrialFile(fn) as tr:
            session = str(tr.attrs.get("label", fn.parent.parent.name))
        dest = out / session / fn.parent.name / fn.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with TrialFile(fn) as tr, h5py.File(dest, "a") as f:
            st = stats[str(fn)]
            if args.gain == "movie":
                st = dict(st, gain=movie_gain(st, tr.valid))
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
            for name, path, model, wcfg, ck, amp_dtype in models:
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
    p.add_argument("--ckpt", nargs="*", default=[],
                   help="name=path of each checkpoint, best first; the path can be avg:SPEC;SPEC;... for a weighted "
                        "average made in memory (SPEC: path, path=weight or a glob, as in evaluate)")
    p.add_argument("--out", required=True, help="output folder (one file per trial, same relative paths)")
    p.add_argument("--files", default="", help="comma-separated parts of trial paths to keep "
                                               "(e.g. 780125_2025-03-14_09-30-06/Path1/trial_002); default all")
    p.add_argument("--out_frames", type=int, default=0,
                   help="output frames per tile; 0 = the whole trial, or as many as fit in GPU memory")
    p.add_argument("--out_px", type=int, default=0, help="output pixels per tile side; 0 = the checkpoint's training size")
    p.add_argument("--batch", type=int, default=1, help="tiles per forward pass (halved when the GPU runs out of memory)")
    p.add_argument("--no_silo", action="store_true", help="skip SiLo's baseline")
    p.add_argument("--no_movie", action="store_true", help="skip the movie and activity (when adding checkpoints)")
    p.add_argument("--sd", action="store_true", help="also save each network's noise SD")
    p.add_argument("--stats_cache", default="", help="trial stats cache folder (default <out>/stats)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16", "amp", "fp32"],
                   help="auto = fp16 on GPU (the whole model cast), fp32 on CPU; amp = bf16 autocast (before 2026-11-17)")
    p.add_argument("--no_amp", action="store_true", help="same as --precision fp32")
    p.add_argument("--gain", default="stats", choices=["stats", "movie"],
                   help="the trial's variance/mean ratio: from movie - activity (stats, as in training) or from the movie "
                        "alone (movie: for trials without GIAnT activity, e.g. before localization)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (Linux; each new tile shape compiles)")
