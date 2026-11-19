"""Local (non-rigid) residual motion: per-block displacement minus the field's, free of noise.

usage: python local_motion.py TRIAL.h5 ... [--sizes 32,64,128] [--check 8]

Like residual_motion.py, every kept frame of movie - activity is fitted as
(1 + a) M - d . grad M by weighted least squares, here per block of
size x size pixels and on the whole field. The pixels are split in two
halves (checkerboard of --check x --check squares); each half gets its own
fit, per block and for the whole field. Local motion of a block = its
displacement minus the field's. The covariance of the two halves' local
motion estimates gives its variance free of noise (8 px squares keep the
80 Hz noise's spatial correlation out of it). Reported per block size,
over blocks with at least 400 fitted pixels: the rigid SD, the local SD
(RMS over blocks), the local share of the total motion variance, and the
noise SD of one block's estimate (the per-frame precision a detector of
that size can reach). Fast and slow parts split by a 17-frame running mean.

Not zero without motion: within a block, a brightness change that varies
across a structure looks like a shift of it ((x - x0) G is proportional to
dG/dx for a Gaussian blob), so the simulator's brightness fields read
0.04-0.19 px here. Compare real trials with no-motion simulations of the
same template.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from residual_motion import runmean  # noqa: E402
from slap2_f0.io import TrialFile  # noqa: E402


class Fitter:
    """Weighted LS fit of (a, d_r, d_c) per frame on subsets of the fitted pixels; data kept compressed."""

    def __init__(self, movie, M, gr, gc, ok, keep):
        self.idx = np.flatnonzero(ok.ravel())
        self.pos = np.full(ok.size, -1)
        self.pos[self.idx] = np.arange(len(self.idx))
        m = M.ravel()[self.idx].astype(np.float64)
        self.X = np.stack([m, -gr.ravel()[self.idx], -gc.ravel()[self.idx]], 1).astype(np.float64)
        self.w = 1.0 / np.maximum(m, 0.05 * np.median(m))
        self.Y = np.nan_to_num(movie.reshape(movie.shape[0], -1)[:, self.idx].astype(np.float64) - m[None])
        self.keep = keep

    def __call__(self, mask):
        j = self.pos[np.flatnonzero(mask.ravel())]
        j = j[j >= 0]
        X, w = self.X[j], self.w[j]
        A = X.T @ (X * w[:, None])
        beta = np.linalg.solve(A, ((self.Y[:, j] * w[None]) @ X).T)
        beta[:, ~self.keep] = np.nan
        return beta


def parts(x):
    s = runmean(x[None])[0]
    return {"all": x, "fast": x - s, "slow": s}


def covvar(a, b):
    """Shared variance of two estimates, and the noise variance of their mean."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float(np.mean(a * b)), float(0.25 * np.mean((a - b) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="+")
    ap.add_argument("--sizes", default="32,64,128")
    ap.add_argument("--check", type=int, default=8)
    args = ap.parse_args()
    for fn in args.trials:
        with TrialFile(fn) as tf, h5py.File(fn, "r") as f:
            movie = f["movie"][tf.channel].astype(np.float32)
            if "activity" in f:
                movie -= np.nan_to_num(f["activity"][tf.channel].astype(np.float32))
            keep, valid = ~tf.discard, tf.valid
            label = f"{Path(fn).parent.parent.name}/{Path(fn).stem}"
        M = np.where(valid, np.nanmean(movie[keep], 0), np.nan)
        gr = np.full_like(M, np.nan)
        gc = np.full_like(M, np.nan)
        gr[1:-1] = (M[2:] - M[:-2]) / 2
        gc[:, 1:-1] = (M[:, 2:] - M[:, :-2]) / 2
        ok = valid & np.isfinite(M) & np.isfinite(gr) & np.isfinite(gc)
        rr, cc = np.indices(M.shape)
        half = ((rr // args.check + cc // args.check) % 2).astype(bool)
        fit = Fitter(movie, M, gr, gc, ok, keep)
        del movie
        gA, gB = fit(ok & half)[1:], fit(ok & ~half)[1:]
        print(f"\n{label}: {int(ok.sum())} px, {int(keep.sum())} kept frames", flush=True)
        for p in ("all", "fast", "slow"):
            rig = [covvar(parts(gA[i])[p], parts(gB[i])[p]) for i in (0, 1)]
            print(f"  rigid {p:4s}: SD {np.sqrt(max(rig[0][0], 0)):.3f} / {np.sqrt(max(rig[1][0], 0)):.3f} px (rows / cols), "
                  f"noise of the field estimate {np.sqrt(rig[0][1]):.3f} / {np.sqrt(rig[1][1]):.3f}")
        print(f"  {'block':>5s} {'n':>3s} {'band':>4s} {'local SD r/c':>13s} {'local share':>11s} {'block noise SD r/c':>19s}  (median block px)")
        for size in [int(s) for s in args.sizes.split(",")]:
            res = {p: [] for p in ("all", "fast", "slow")}
            npx = []
            for r0 in range(0, M.shape[0], size):
                for c0 in range(0, M.shape[1], size):
                    blk = np.zeros_like(ok)
                    blk[r0:r0 + size, c0:c0 + size] = True
                    if (ok & blk).sum() < 400 or (ok & blk & half).sum() < 150 or (ok & blk & ~half).sum() < 150:
                        continue
                    try:
                        bA = fit(ok & blk & half)[1:]
                        bB = fit(ok & blk & ~half)[1:]
                    except np.linalg.LinAlgError:
                        continue
                    npx.append(int((ok & blk).sum()))
                    for p in res:
                        loc = [covvar(parts(bA[i] - gA[i])[p], parts(bB[i] - gB[i])[p]) for i in (0, 1)]
                        rig = [covvar(parts(gA[i])[p], parts(gB[i])[p])[0] for i in (0, 1)]
                        res[p].append([loc[0][0], loc[1][0], rig[0], rig[1], loc[0][1], loc[1][1]])
            if not npx:
                continue
            for p, v in res.items():
                v = np.array(v)
                lv = v[:, :2].mean(0)  # mean shared local variance over blocks
                rv = v[:, 2:4].mean(0)
                share = lv.sum() / max(lv.sum() + max(rv.sum(), 0), 1e-12)
                nz = np.sqrt(np.median(v[:, 4:6], 0))
                print(f"  {size:5d} {len(npx):3d} {p:>4s} {np.sign(lv[0]) * np.sqrt(abs(lv[0])):6.3f} / {np.sign(lv[1]) * np.sqrt(abs(lv[1])):5.3f}"
                      f" {share:11.2f} {nz[0]:9.3f} / {nz[1]:6.3f}  ({int(np.median(npx))})", flush=True)


if __name__ == "__main__":
    main()
