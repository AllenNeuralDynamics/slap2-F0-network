"""Whole-trial statistics of exported trials, computed once and cached.

Per pixel: mean and variance of the movie (the variance map is a network
input: pixels with events have more variance than photon noise explains), and
mean and mean square of the activity (the simulator's event statistics).

Per trial, measured on background pixels (valid and unlabeled, where there is
no fitted activity), on the residual movie - activity:
    gain    variance / mean in counts (photon noise times detector gain,
            after the 80 Hz kernel)
    rho_t1, rho_t2   temporal correlation at lags of 1 and 2 frames
    rho_r, rho_c     correlation with the next row / next column
    F64     variance of 64-frame means relative to var / 64: how much
            correlated noise inflates long averages
    skew    skewness of single frames (photon noise at low counts is skewed)
The simulator matches these, so its noise looks like the real movie's.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

MAPS = ("mean_movie", "var_movie", "mean_act", "mean_act2", "n_valid")
SCALARS = ("gain", "rho_t1", "rho_t2", "rho_r", "rho_c", "F64", "skew", "n_frames")
VERSION = 2  # bump when the stats change, to invalidate caches


def trial_stats(fn: str | Path, channel: int = 0, chunk: int = 64, block: int = 64) -> dict:
    """One pass over the trial in chunks of frames (chunk must be a multiple of
    block); chunks are float32 and sums float64, to bound memory."""
    with h5py.File(fn, "r") as f:
        _, T, R, C = f["movie"].shape
        mi = f["mean_image"][channel]
        valid = f["mask/valid"][:].astype(bool) & np.isfinite(mi)
        bg = valid & ~f["mask/labeled"][:].astype(bool)
        discard = f["frame/discard"][:].astype(bool)
        z = lambda *s: np.zeros(s or (R, C), np.float64)  # noqa: E731
        n, sm, sm2, sa, sa2 = z(), z(), z(), z(), z()
        # residual moments on background pixels: own, lag 1 and 2, next row, next column
        sx, sx2, sx3 = z(), z(), z()
        lag = {k: [z(), z(), z(), z()] for k in (1, 2)}  # sum x_t x_t+k, sum x_t, sum x_t+k, count
        nb_r = [z(R - 1, C) for _ in range(4)]
        nb_c = [z(R, C - 1) for _ in range(4)]
        bsum, bsum2, bn = z(), z(), z()
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
           dict(mean_movie=mean_movie, var_movie=var_movie, mean_act=mean_act, mean_act2=mean_act2, n_valid=n).items()}
    out.update(gain=gain, rho_t1=corr_t(1), rho_t2=corr_t(2), rho_r=rho_r, rho_c=rho_c, F64=F64, skew=skew, n_frames=float(T))
    return out


def _key(fn: Path) -> str:
    st = fn.stat()
    return hashlib.sha1(f"{fn.resolve()}|{st.st_size}|{int(st.st_mtime)}|v{VERSION}".encode()).hexdigest()[:16]


def _cached(args) -> dict:
    fn, cache = args
    fn = Path(fn)
    if cache is not None:
        p = Path(cache) / f"{fn.stem}_{_key(fn)}.npz"
        if p.exists():
            d = np.load(p)
            return {k: (d[k] if k in MAPS else float(d[k])) for k in d.files}
    s = trial_stats(fn)
    if cache is not None:
        Path(cache).mkdir(parents=True, exist_ok=True)
        np.savez(p, **s)
    return s


def load_stats(files, cache_dir: str | Path | None = None, workers: int = 0) -> dict[str, dict]:
    """Stats for each file, keyed by str(path); computed in parallel, cached as .npz."""
    files = [str(f) for f in files]
    jobs = [(f, None if cache_dir is None else str(cache_dir)) for f in files]
    if workers and len(files) > 1:
        # each worker holds a few float32 chunks of a trial (up to ~1 GB for large crops)
        with ProcessPoolExecutor(max_workers=min(workers, 6, len(files), os.cpu_count() or 1)) as ex:
            res = list(ex.map(_cached, jobs))
    else:
        res = [_cached(j) for j in jobs]
    return dict(zip(files, res))
