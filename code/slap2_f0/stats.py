"""Whole-trial statistics of exported trials, computed once and cached.

Per pixel: mean and variance of the movie (the variance map is a network
input: pixels with events have more variance than photon noise explains),
mean and mean square of the activity (the simulator's event statistics), and
mean and variance of the residual movie - activity, of single frames and of
8- and 64-frame means (mean_res, var_res, var8_res, var64_res: on bright
pixels the residual varies more than photon noise explains, both frame to
frame and over seconds, and the simulator adds that excess).

`slap2-f0 stats` stores them next to each trial file (``sidecar_path``), where
training finds them first; otherwise they are cached in a stats folder.

Per trial, measured on background pixels (valid and unlabeled, where there is
no fitted activity), on the residual movie - activity:
    gain    variance / mean in counts (photon noise times detector gain,
            after the 80 Hz kernel)
    rho_t1, rho_t2   temporal correlation at lags of 1 and 2 frames
    rho_r, rho_c     correlation with the next row / next column
    F64     variance of 64-frame means relative to var / 64: how much
            correlated noise inflates long averages
    skew    skewness of single frames (photon noise at low counts is skewed)
    zero_frac   share of exact zeros among valid movie voxels
The simulator matches these, so its noise looks like the real movie's.

Also per pixel the share of frames where the movie is exactly 0 (zero_movie:
real movies are photon counts, never negative, and dim pixels are 0 in many
frames), and per frame the activity summed over footprint-support pixels
(act_trace, where the simulator may cut the activity's timeline without
splitting events).
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

MAPS = ("mean_movie", "var_movie", "mean_act", "mean_act2", "n_valid", "mean_res", "var_res", "var8_res", "var64_res",
        "zero_movie", "act_trace")  # arrays (act_trace is per frame)
SCALARS = ("gain", "rho_t1", "rho_t2", "rho_r", "rho_c", "F64", "skew", "zero_frac", "n_frames")
VERSION = 5  # bump when the stats change, to invalidate caches


def trial_stats(fn: str | Path, channel: int = 0, chunk: int = 64, block: int = 64) -> dict:
    """One pass over the trial in chunks of frames (chunk must be a multiple of
    block); chunks are float32 and sums float64, to bound memory."""
    with h5py.File(fn, "r") as f:
        _, T, R, C = f["movie"].shape
        mi = f["mean_image"][channel]
        valid = f["mask/valid"][:].astype(bool) & np.isfinite(mi)
        bg = valid & ~f["mask/labeled"][:].astype(bool)
        support = valid & f["mask/footprint_support"][:].astype(bool)
        discard = f["frame/discard"][:].astype(bool)
        act_trace = np.zeros(T, np.float64)
        z = lambda *s: np.zeros(s or (R, C), np.float64)  # noqa: E731
        n, sm, sm2, sa, sa2, nz = z(), z(), z(), z(), z(), z()
        # residual moments on background pixels: own, lag 1 and 2, next row, next column
        sx, sx2, sx3 = z(), z(), z()
        lag = {k: [z(), z(), z(), z()] for k in (1, 2)}  # sum x_t x_t+k, sum x_t, sum x_t+k, count
        nb_r = [z(R - 1, C) for _ in range(4)]
        nb_c = [z(R, C - 1) for _ in range(4)]
        bsum, bsum2, bn = z(), z(), z()
        rs, rs2, b8s, b8s2, b8n = z(), z(), z(), z(), z()  # residual on all valid pixels; 8-frame block means
        b64s, b64s2, b64n = z(), z(), z()  # and 64-frame block means
        prev = None  # last 2 frames of the previous chunk
        for t0 in range(0, T, chunk):
            t1 = min(T, t0 + chunk)
            m = f["movie"][channel, t0:t1].astype(np.float32)
            a = f["activity"][channel, t0:t1].astype(np.float32)
            m[:, ~valid] = np.nan
            m[discard[t0:t1]] = np.nan
            fin = np.isfinite(m)
            m0 = np.where(fin, m, 0)
            a0 = np.where(fin, a, 0)
            n += fin.sum(0, dtype=np.float64)
            sm += m0.sum(0, dtype=np.float64)
            sm2 += (m0 ** 2).sum(0, dtype=np.float64)
            sa += a0.sum(0, dtype=np.float64)
            sa2 += (a0 ** 2).sum(0, dtype=np.float64)
            nz += (fin & (m == 0)).sum(0, dtype=np.float64)
            act_trace[t0:t1] = np.nan_to_num(a[:, support]).sum(1, dtype=np.float64)
            xr = m0 - a0
            rs += xr.sum(0, dtype=np.float64)
            rs2 += (xr ** 2).sum(0, dtype=np.float64)
            for b0 in range(0, xr.shape[0] - 7, 8):
                cnt = fin[b0: b0 + 8].sum(0)
                good = cnt >= 6
                bm = np.where(good, xr[b0: b0 + 8].sum(0) / np.maximum(cnt, 1), 0)
                b8s += bm
                b8s2 += bm ** 2
                b8n += good
            for b0 in range(0, xr.shape[0] - block + 1, block):
                cnt = fin[b0: b0 + block].sum(0)
                good = cnt >= 0.75 * block
                bm = np.where(good, xr[b0: b0 + block].sum(0) / np.maximum(cnt, 1), 0)
                b64s += bm
                b64s2 += bm ** 2
                b64n += good
            x = np.where(fin & bg, m - a, np.nan)
            fx = np.isfinite(x)
            x0 = np.where(fx, x, 0)
            sx += x0.sum(0, dtype=np.float64)
            sx2 += (x0 ** 2).sum(0, dtype=np.float64)
            sx3 += (x0 ** 3).sum(0, dtype=np.float64)
            xe = x if prev is None else np.concatenate([prev, x])
            for k in (1, 2):
                if xe.shape[0] > k:
                    p, q = xe[:-k], xe[k:]
                    ok = np.isfinite(p) & np.isfinite(q)
                    acc = lag[k]
                    acc[0] += np.where(ok, p * q, 0).sum(0, dtype=np.float64)
                    acc[1] += np.where(ok, p, 0).sum(0, dtype=np.float64)
                    acc[2] += np.where(ok, q, 0).sum(0, dtype=np.float64)
                    acc[3] += ok.sum(0, dtype=np.float64)
            prev = x[-2:]
            for acc, p, q in ((nb_r, x[:, :-1], x[:, 1:]), (nb_c, x[:, :, :-1], x[:, :, 1:])):
                ok = np.isfinite(p) & np.isfinite(q)
                acc[0] += np.where(ok, p * q, 0).sum(0, dtype=np.float64)
                acc[1] += np.where(ok, p, 0).sum(0, dtype=np.float64)
                acc[2] += np.where(ok, q, 0).sum(0, dtype=np.float64)
                acc[3] += ok.sum(0, dtype=np.float64)
            for b0 in range(0, x.shape[0] - block + 1, block):
                xb = x[b0: b0 + block]
                cnt = np.isfinite(xb).sum(0, dtype=np.float64)
                good = cnt >= 0.75 * block
                bm = np.where(good, np.nansum(xb, 0) / np.maximum(cnt, 1), 0)
                bsum += bm
                bsum2 += bm ** 2
                bn += good
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_movie = sm / n
        var_movie = sm2 / n - mean_movie ** 2
        mean_act = sa / n
        mean_act2 = sa2 / n
        mean_res = rs / n
        zero_movie = nz / n
        var_res = rs2 / n - mean_res ** 2
        var8_res = b8s2 / np.maximum(b8n, 1) - (b8s / np.maximum(b8n, 1)) ** 2
        var8_res[b8n < 8] = np.nan
        var64_res = b64s2 / np.maximum(b64n, 1) - (b64s / np.maximum(b64n, 1)) ** 2
        var64_res[b64n < 8] = np.nan
        cnt_x = np.maximum(n * bg, 1)
        mx = sx / cnt_x
        vx = sx2 / cnt_x - mx ** 2
        use = bg & (n > 100) & (mx > 0) & (vx > 0)

        def corr_t(k):
            s, p, q, c = lag[k]
            cov = s / np.maximum(c, 1) - (p / np.maximum(c, 1)) * (q / np.maximum(c, 1))
            return float(cov[use].sum() / vx[use].sum())

        def corr_s(acc, ua, ub, va, vb):
            s, p, q, c = acc
            ok = ua & ub & (c > 100)
            cov = s / np.maximum(c, 1) - (p / np.maximum(c, 1)) * (q / np.maximum(c, 1))
            return float(cov[ok].sum() / np.sqrt(va[ok] * vb[ok]).sum())

        rho_r = corr_s(nb_r, use[:-1], use[1:], vx[:-1], vx[1:])
        rho_c = corr_s(nb_c, use[:, :-1], use[:, 1:], vx[:, :-1], vx[:, 1:])
        bvar = bsum2 / np.maximum(bn, 1) - (bsum / np.maximum(bn, 1)) ** 2
        okb = use & (bn >= 8)
        F64 = float(np.median(bvar[okb] * block / vx[okb])) if okb.any() else float("nan")
        gain = float(np.median(vx[use] / mx[use])) if use.any() else float("nan")
        m3 = sx3 / cnt_x - 3 * mx * (sx2 / cnt_x) + 2 * mx ** 3
        skew = float(np.median(m3[use] / vx[use] ** 1.5)) if use.any() else float("nan")
    out = {k: v.astype(np.float32) for k, v in
           dict(mean_movie=mean_movie, var_movie=var_movie, mean_act=mean_act, mean_act2=mean_act2, n_valid=n,
                mean_res=mean_res, var_res=var_res, var8_res=var8_res, var64_res=var64_res, zero_movie=zero_movie,
                act_trace=act_trace).items()}
    zf = float(nz[valid].sum() / max(n[valid].sum(), 1))
    out.update(gain=gain, rho_t1=corr_t(1), rho_t2=corr_t(2), rho_r=rho_r, rho_c=rho_c, F64=F64, skew=skew, zero_frac=zf,
               n_frames=float(T))
    return out


def _key(fn: Path) -> str:
    st = fn.stat()
    return hashlib.sha1(f"{fn.resolve()}|{st.st_size}|{int(st.st_mtime)}|v{VERSION}".encode()).hexdigest()[:16]


def sidecar_path(fn: str | Path) -> Path:
    """Where `slap2-f0 stats` stores a trial's stats: next to it, so they travel
    with the exported data (e.g. in its Code Ocean data asset), whatever path
    it is mounted at."""
    fn = Path(fn)
    return fn.with_name(f"{fn.stem}_stats_v{VERSION}.npz")


def _from_npz(p: Path) -> dict:
    d = np.load(p)
    return {k: (d[k] if k in MAPS else float(d[k])) for k in d.files if k != "file_size"}


def _cached(args) -> dict:
    """Stats of one file: from its sidecar (if made for this version and file
    size), else from the cache folder, else computed (and then cached, and
    written as a sidecar if asked)."""
    fn, cache, write_sidecar = args
    fn = Path(fn)
    side = sidecar_path(fn)
    size = fn.stat().st_size
    if side.exists():
        with np.load(side) as d:
            ok = "file_size" in d.files and int(d["file_size"]) == size
        if ok:
            return _from_npz(side)
    p = None
    if cache is not None:
        p = Path(cache) / f"{fn.stem}_{_key(fn)}.npz"
        if p.exists():
            s = _from_npz(p)
            if write_sidecar:
                np.savez(side, file_size=size, **s)
            return s
    s = trial_stats(fn)
    if p is not None:
        Path(cache).mkdir(parents=True, exist_ok=True)
        np.savez(p, **s)
    if write_sidecar:
        np.savez(side, file_size=size, **s)
    return s


def load_stats(files, cache_dir: str | Path | None = None, workers: int = 0, max_workers: int = 6,
               write_sidecars: bool = False) -> dict[str, dict]:
    """Stats for each file, keyed by str(path); computed in parallel, cached as
    .npz in cache_dir and/or (write_sidecars) next to each file, where they
    are found first."""
    files = [str(f) for f in files]
    jobs = [(f, None if cache_dir is None else str(cache_dir), write_sidecars) for f in files]
    if workers and len(files) > 1:
        # each worker holds a few float32 chunks of a trial (up to ~1 GB for large crops)
        with ProcessPoolExecutor(max_workers=min(workers, max_workers, len(files), os.cpu_count() or 1)) as ex:
            res = list(ex.map(_cached, jobs))
    else:
        res = [_cached(j) for j in jobs]
    return dict(zip(files, res))


def run(args) -> None:
    """`slap2-f0 stats`: compute every exported trial's stats under a folder and
    store them next to the trials (sidecars), e.g. at the end of an export run."""
    import time

    from .io import list_trials

    files = list_trials(args.root)
    t0 = time.time()
    stats = load_stats(files, None, workers=args.workers, max_workers=args.workers, write_sidecars=True)
    print(f"stats of {len(files)} trial files under {args.root} in {time.time() - t0:.0f} s; gain "
          + " ".join(f"{stats[str(f)]['gain']:.1f}" for f in files[:8]) + (" ..." if len(files) > 8 else ""), flush=True)
