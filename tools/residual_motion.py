"""Residual motion in exported (registered) trials: how large, how fast.

usage: python residual_motion.py TRIAL.h5 [TRIAL.h5 ...] [--block 8] [--blur]

Each kept frame of movie - activity (GIAnT's fitted activity removed) is fit
at the valid pixels as the trial's mean image, scaled and shifted:
    frame ~ (1 + a) * M - d_r * dM/drow - d_c * dM/dcol
by least squares weighted by 1 / M (photon noise). d is a sub-pixel shift
(rows, columns, in pixels) and a the frame's fractional brightness change.
Noise in single frames makes each estimate noisy, so the pixels are split
into two halves (a checkerboard of --block x --block squares) and each half
gets its own estimate: rigid motion is shared by the halves, noise is not,
so their covariance estimates the motion's variance free of noise. Reported
per axis: the motion SD from that covariance, the noise SD of one half's
estimate, and the same for the fast part (minus a 17-frame running mean)
and the slow part. Brightness a likewise, as a check (2-5% per frame).
In simulated trials with truth/motion the true SDs are printed alongside.

--blur adds second derivatives of M to the fit, + b_r d2M/drow2 + b_c d2M/dcol2
(px^2; positive = blurred): motion within the frame-formation kernel (frames
pool 200 Hz samples over about two 80 Hz frames) blurs the frame along the
motion, by half the displacement's variance within the kernel. The shift's
own second-order term adds d^2 / 2 (SD s^2 / sqrt(2) for Gaussian d with SD s).
Also printed: the correlation of each blur coefficient with the squared
speed of the frame's displacement (from the two halves: v_A * v_B, an
unbiased estimate of v^2) and with d^2 (d_A * d_B), and what the shift and
the blur do to the baseline on the quarter of pixels with the steepest
mean image (RMS in % of their mean). Per band (slow, fast): the SDs of the
isotropic ((b_r + b_c) / 2, focus) and anisotropic parts, and noise-free
correlations (covariance across the two halves over the shared SDs) of
b_r with b_c and of the isotropic blur with the brightness a and the
displacement; and its correlation with GIAnT's activity summed over the
field (whether the blur is misfit of the activity).
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from slap2_f0.io import TrialFile  # noqa: E402

RUN = 17


def runmean(x):
    """Centered running mean over RUN samples of x (n, T), ignoring NaN."""
    h = RUN // 2
    v = np.isfinite(x)
    cs = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(np.where(v, x, 0), 1)], 1)
    cn = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(v, 1)], 1)
    T = x.shape[1]
    lo, hi = np.clip(np.arange(T) - h, 0, T), np.clip(np.arange(T) + h + 1, 0, T)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (cs[:, hi] - cs[:, lo]) / (cn[:, hi] - cn[:, lo])
    return np.where(v, out, np.nan)


def fit(movie, M, gr, gc, pix, keep, second=()):
    """Per-frame (a, d_r, d_c[, b_r, b_c]) from the pixels pix (bool R x C);
    NaN at discarded frames. second: second derivatives (d2M/drow2, d2M/dcol2)."""
    X = np.stack([M[pix], -gr[pix], -gc[pix]] + [g[pix] for g in second], 1).astype(np.float64)  # (P, k)
    w = 1.0 / np.maximum(M[pix], 0.05 * np.nanmedian(M[pix])).astype(np.float64)
    A = X.T @ (X * w[:, None])
    Y = movie[:, pix].astype(np.float64) - M[pix][None]  # (T, P)
    Y = np.nan_to_num(Y)
    b = (Y * w[None]) @ X  # (T, 3)
    beta = np.linalg.solve(A, b.T)  # (k, T)
    beta[:, ~keep] = np.nan
    return beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="+")
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--blur", action="store_true")
    args = ap.parse_args()
    for fn in args.trials:
        with TrialFile(fn) as tf, h5py.File(fn, "r") as f:
            movie = f["movie"][tf.channel].astype(np.float32)
            keep = ~tf.discard
            valid = tf.valid
            asum = None  # GIAnT's activity summed over the field, per frame
            if "activity" in f:
                act = np.nan_to_num(f["activity"][tf.channel].astype(np.float32))
                movie -= act
                asum = act[:, valid].sum(1).astype(np.float64)
                del act
            M = np.where(valid, np.nanmean(movie[keep], 0), np.nan)
            # central differences where both neighbours are valid
            gr = np.full_like(M, np.nan)
            gc = np.full_like(M, np.nan)
            gr[1:-1] = (M[2:] - M[:-2]) / 2
            gc[:, 1:-1] = (M[:, 2:] - M[:, :-2]) / 2
            second = ()
            if args.blur:
                grr = np.full_like(M, np.nan)
                gcc = np.full_like(M, np.nan)
                grr[1:-1] = M[2:] - 2 * M[1:-1] + M[:-2]
                gcc[:, 1:-1] = M[:, 2:] - 2 * M[:, 1:-1] + M[:, :-2]
                second = (grr, gcc)
            ok = valid & np.isfinite(M) & np.isfinite(gr) & np.isfinite(gc)
            for g in second:
                ok &= np.isfinite(g)
            rr, cc = np.indices(M.shape)
            half = ((rr // args.block + cc // args.block) % 2).astype(bool)
            bA, bB = fit(movie, M, gr, gc, ok & half, keep, second), fit(movie, M, gr, gc, ok & ~half, keep, second)
            truth = f["truth/motion"][()] if "truth/motion" in f else None
            label = f"{Path(fn).parent.parent.name}/{Path(fn).stem}"
        print(f"\n{label}: {int(ok.sum())} px, {int(keep.sum())} kept frames"
              + ("" if truth is None else " (simulated, true motion known)"))
        print(f"  {'':12s} {'motion SD':>10s} {'noise SD':>9s} {'fast motion':>12s} {'fast noise':>11s} {'slow motion':>12s} {'slow noise':>11s}"
              + ("" if truth is None else f" {'true: all':>10s} {'fast':>6s} {'slow':>6s}"))
        sl_A, sl_B = runmean(bA), runmean(bB)
        parts = {"all": (bA, bB), "fast": (bA - sl_A, bB - sl_B), "slow": (sl_A, sl_B)}
        if truth is not None:
            tt = np.where(keep[None], truth, np.nan)
            ts = runmean(tt)
            tparts = {"all": tt, "fast": tt - ts, "slow": ts}
        names = ["brightness", "rows (px)", "cols (px)"] + (["blur r (px2)", "blur c (px2)"] if second else [])
        shared = {}
        for i, name in enumerate(names):
            cells = []
            for p in ("all", "fast", "slow"):
                a, b = parts[p][0][i], parts[p][1][i]
                m = np.isfinite(a) & np.isfinite(b)
                a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
                cov = float(np.mean(a * b))
                noise = np.sqrt(max(0.5 * np.mean((a - b) ** 2), 0))  # SD of one half's noise
                cells += [np.sign(cov) * np.sqrt(abs(cov)), noise]
                shared[(i, p)] = max(cov, 0.0)
            line = f"  {name:12s} {cells[0]:10.3f} {cells[1]:9.3f} {cells[2]:12.3f} {cells[3]:11.3f} {cells[4]:12.3f} {cells[5]:11.3f}"
            if truth is not None and 0 < i < 3:
                line += " " + " ".join(f"{np.nanstd(tparts[p][i - 1]):{w}.3f}" for p, w in (("all", 10), ("fast", 6), ("slow", 6)))
            print(line, flush=True)
        if second:
            steep = ok & (np.hypot(gr, gc) >= np.nanquantile(np.hypot(gr, gc)[ok], 0.75))
            Mm = M[steep]
            for j, (g1, g2) in enumerate(((gr, grr), (gc, gcc))):
                sd_d, sd_b = np.sqrt(shared[(1 + j, "all")]), np.sqrt(shared[(3 + j, "all")])
                eff_d = 100 * sd_d * np.sqrt(np.mean((g1[steep] / Mm) ** 2))
                eff_b = 100 * sd_b * np.sqrt(np.mean((g2[steep] / Mm) ** 2))
                dA, dB, b = bA[1 + j], bB[1 + j], 0.5 * (bA[3 + j] + bB[3 + j])
                vA, vB = np.full_like(dA, np.nan), np.full_like(dB, np.nan)
                vA[1:-1], vB[1:-1] = (dA[2:] - dA[:-2]) / 2, (dB[2:] - dB[:-2]) / 2
                def corr(x, y):
                    m = np.isfinite(x) & np.isfinite(y)
                    return float(np.corrcoef(x[m], y[m])[0, 1])
                print(f"  {'rows' if j == 0 else 'cols'}: shift's 2nd-order term SD {shared[(1 + j, 'all')] / np.sqrt(2):.4f} px2; "
                      f"blur vs speed^2 r = {corr(b, vA * vB):+.2f}, vs d^2 r = {corr(b, dA * dB):+.2f}; "
                      f"on the steepest quarter: shift {eff_d:.2f}%, blur {eff_b:.2f}% of the baseline", flush=True)

            def cv(x, y):
                m = np.isfinite(x) & np.isfinite(y)
                return float(np.mean((x[m] - x[m].mean()) * (y[m] - y[m].mean())))

            def tc(xA, xB, yA, yB):  # noise-free correlation: covariance across halves over the shared SDs
                return 0.5 * (cv(xA, yB) + cv(xB, yA)) / np.sqrt(max(cv(xA, xB), 1e-12) * max(cv(yA, yB), 1e-12))

            for p in ("slow", "fast"):
                A_, B_ = parts[p]
                iA, iB = 0.5 * (A_[3] + A_[4]), 0.5 * (B_[3] + B_[4])
                nA, nB = 0.5 * (A_[3] - A_[4]), 0.5 * (B_[3] - B_[4])
                line = (f"  {p} blur: isotropic SD {np.sqrt(max(cv(iA, iB), 0)):.3f}, anisotropic {np.sqrt(max(cv(nA, nB), 0)):.3f} px2; "
                        f"corr b_r~b_c {tc(A_[3], B_[3], A_[4], B_[4]):+.2f}; isotropic blur ~ brightness {tc(iA, iB, A_[0], B_[0]):+.2f}, "
                        f"~ d_r {tc(iA, iB, A_[1], B_[1]):+.2f}, ~ d_c {tc(iA, iB, A_[2], B_[2]):+.2f}")
                if asum is not None and np.std(asum) > 0:
                    af = np.where(keep, asum, np.nan)
                    sa = runmean(af[None])[0]
                    af = af - sa if p == "fast" else sa
                    line += f", ~ field activity {np.corrcoef(*[x[np.isfinite(x) & np.isfinite(af)] for x in (0.5 * (iA + iB), af)])[0, 1]:+.2f}"
                print(line, flush=True)


if __name__ == "__main__":
    main()
