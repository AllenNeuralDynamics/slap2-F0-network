"""Do the motion-correction traces predict baseline changes?

usage: python motion_covariates.py EXPORT_DIR PRED_DIR [--model v9_finetuned] [--fast 33]

EXPORT_DIR holds exported trials (<session>/Path1/trial_*.h5), PRED_DIR the
network's predictions for them (``slap2-f0 predict``, same layout).

Per diag trial (exports carry frame/onlineMotion{X,Y,Z}shift, frame/motion_{row,col},
frame/rec_neg_err):

A. Global: per-frame brightness a, residual shift d and blur b of movie - activity
   (residual_motion.fit on two checkerboard halves). Noise-free correlation of each
   with each motion covariate, and the noise-corrected R^2 of a regression on all of
   them, in the fast band (minus a 17-frame running mean) and the slow band.

B. Local: residual r = movie - activity - network baseline, 8-frame mean, as a
   fraction of the baseline. Per pixel, a regression on the motion covariates
   (8-frame means, standardized), fitted on alternate 128-frame blocks and
   tested on the others. Reported per pixel class: the variance explained out of
   sample minus the same with the covariates circularly shifted by a third of the
   trial (the overfitting null), as RMS in % of the baseline, and as a share of
   the residual's variance. The same for the frame's own whole-field fit
   (brightness, shift, blur from A) as covariates, and for both together.
   --fast N: B on the fast band only (minus an N-frame running mean).
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from residual_motion import fit, runmean  # noqa: E402


def box(x, n=8):
    """Running mean over n frames along axis 0 (T, ...), NaN-aware, centered."""
    v = np.isfinite(x)
    z = np.where(v, x, 0.0)
    cs = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(z, 0)], 0)
    cn = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(v, 0)], 0)
    T = x.shape[0]
    lo = np.clip(np.arange(T) - n // 2, 0, T)
    hi = np.clip(np.arange(T) - n // 2 + n, 0, T)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (cs[hi] - cs[lo]) / np.maximum(cn[hi] - cn[lo], 1)
    out[(cn[hi] - cn[lo]) < n // 2] = np.nan
    return out


def deriv(x):
    d = np.full_like(x, np.nan)
    d[1:-1] = (x[2:] - x[:-2]) / 2
    return d


def covariates(f):
    fr = f["frame"]
    ox, oy, oz = (fr[f"onlineMotion{k}shift"][()].astype(np.float64) for k in "XYZ")
    mr, mc = fr["motion_row"][()].astype(np.float64), fr["motion_col"][()].astype(np.float64)
    err = fr["rec_neg_err"][()].astype(np.float64)
    vr, vc = deriv(mr), deriv(mc)
    C = {"online x": ox, "online y": oy, "online z": oz,
         "d/dt online x": deriv(ox), "d/dt online y": deriv(oy), "d/dt online z": deriv(oz),
         "offline row": mr, "offline col": mc, "d/dt offline row": vr, "d/dt offline col": vc,
         "offline speed^2": vr ** 2 + vc ** 2, "registration error": err}
    return C


def nfcorr(A, B, c):
    """Noise-free correlation of a parameter measured twice (A, B) with covariate c."""
    m = np.isfinite(A) & np.isfinite(B) & np.isfinite(c)
    if m.sum() < 50:
        return np.nan
    a, b, c = A[m] - A[m].mean(), B[m] - B[m].mean(), c[m] - c[m].mean()
    sab = np.mean(a * b)
    if sab <= 0 or c.std() == 0:
        return np.nan
    return float(np.mean(0.5 * (a + b) * c) / np.sqrt(sab) / c.std())


def nfr2(A, B, Cm):
    """Noise-corrected R^2 of the halves' mean on covariates (in-sample, adjusted)."""
    y = 0.5 * (A + B)
    m = np.isfinite(y) & np.isfinite(A) & np.isfinite(B) & np.all(np.isfinite(Cm), 1)
    if m.sum() < 100:
        return np.nan
    X = np.column_stack([np.ones(m.sum()), Cm[m]])
    yy = y[m] - y[m].mean()
    beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    pred = X @ beta - y[m].mean()
    n, k = len(yy), X.shape[1] - 1
    expl = np.var(pred) - k / n * np.var(yy - pred)  # adjust for chance fit
    a, b = A[m] - A[m].mean(), B[m] - B[m].mean()
    sig = np.mean(a * b)
    return float(expl / sig) if sig > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exports")
    ap.add_argument("preds")
    ap.add_argument("--model", default="v9_finetuned")
    ap.add_argument("--fast", type=int, default=0, help="B on the fast band only: minus a running mean of this many frames")
    args = ap.parse_args()
    for fn in sorted(Path(args.exports).glob("*/Path1/trial_*.h5")):
        sess = fn.parent.parent.name
        pf = Path(args.preds) / sess / "Path1" / fn.name
        with h5py.File(fn, "r") as f:
            movie = f["movie"][0].astype(np.float32)
            act = np.nan_to_num(f["activity"][0].astype(np.float32)) if "activity" in f else 0
            movie -= act
            valid = f["mask/valid"][()].astype(bool)
            sup = f["mask/footprint_support"][()].astype(bool)
            keep = f["frame/discard"][()] == 0
            gain = float(f.attrs.get("photonScale", [25])[0])
            C = covariates(f)
        with h5py.File(pf, "r") as g:
            base = g[f"baseline/{args.model}"][()].astype(np.float32)
            rows, cols = g["rows"][()], g["cols"][()]
            pgain = float(g.attrs["gain"])
        T = movie.shape[0]
        movie[~keep] = np.nan
        print(f"\n{sess}/{fn.stem}: {T} frames, {keep.sum()} kept; covariate SDs (kept frames): "
              + ", ".join(f"{k} {np.nanstd(v[keep]):.3g}" for k, v in C.items() if not k.startswith("d/dt")), flush=True)

        # ---- A: global parameters vs covariates
        M = np.where(valid, np.nanmean(movie[keep], 0), np.nan)
        gr, gc = np.full_like(M, np.nan), np.full_like(M, np.nan)
        gr[1:-1], gc[:, 1:-1] = (M[2:] - M[:-2]) / 2, (M[:, 2:] - M[:, :-2]) / 2
        grr, gcc = np.full_like(M, np.nan), np.full_like(M, np.nan)
        grr[1:-1], gcc[:, 1:-1] = M[2:] - 2 * M[1:-1] + M[:-2], M[:, 2:] - 2 * M[:, 1:-1] + M[:, :-2]
        ok = valid & np.isfinite(gr) & np.isfinite(gc) & np.isfinite(grr) & np.isfinite(gcc)
        rr, cc = np.indices(M.shape)
        half = ((rr // 8 + cc // 8) % 2).astype(bool)
        bA = fit(movie, M, gr, gc, ok & half, keep, (grr, gcc))
        bB = fit(movie, M, gr, gc, ok & ~half, keep, (grr, gcc))
        names = ["brightness", "shift r", "shift c", "blur r", "blur c"]
        Cm = np.column_stack([v for v in C.values()])
        Cm[~keep] = np.nan
        for band in ("fast", "slow"):
            def bp(x):
                s = runmean(x[None])[0]
                return x - s if band == "fast" else s
            Cb = np.column_stack([bp(Cm[:, j]) for j in range(Cm.shape[1])])
            print(f"  A {band}: noise-free corr with each covariate; last column: noise-corrected R^2 on all")
            print("    " + " " * 11 + " ".join(f"{k[:9]:>9s}" for k in C) + "      R^2")
            for i, nm in enumerate(names):
                A_, B_ = bp(bA[i]), bp(bB[i])
                cs = [nfcorr(A_, B_, Cb[:, j]) for j in range(Cb.shape[1])]
                print(f"    {nm:11s}" + " ".join(f"{c:+9.2f}" for c in cs) + f" {nfr2(A_, B_, Cb):8.2f}", flush=True)

        # ---- B: per-pixel residual after the network baseline vs covariates
        P = len(rows)
        mv = movie[:, rows, cols]
        r = (mv - base) / np.maximum(base, 1e-3)
        r8 = box(r, 8)
        if args.fast:
            r8 = r8 - box(r8, args.fast)
        noise8 = pgain / np.maximum(np.nanmean(base, 0), 1e-3) / 8  # photon-noise variance of r8 (fractional), ignoring correlation
        Cs = box(Cm, 8)
        if args.fast:
            Cs = Cs - box(Cs, args.fast)
        Cs = (Cs - np.nanmean(Cs, 0)) / np.nanstd(Cs, 0)
        blk = (np.arange(T) // 128) % 2
        rng = np.random.default_rng(0)
        steep = np.hypot(gr, gc)[rows, cols] / np.maximum(M[rows, cols], 1e-3)
        classes = {"support": sup[rows, cols], "background": ~sup[rows, cols],
                   "steepest quarter": steep >= np.nanpercentile(steep, 75)}

        def cv_explained(X):
            okf = np.all(np.isfinite(X), 1) & np.isfinite(r8).mean(1).astype(bool)
            out = np.zeros(P)
            tot = np.zeros(P)
            Y = np.nan_to_num(r8)
            W = np.isfinite(r8) & okf[:, None]
            for k in (0, 1):
                tr, te = okf & (blk == k), okf & (blk != k)
                Xa = np.column_stack([np.ones(tr.sum()), X[tr]])
                Xe = np.column_stack([np.ones(te.sum()), X[te]])
                lam = 1e-3 * tr.sum() * np.eye(Xa.shape[1]); lam[0, 0] = 0
                beta = np.linalg.solve(Xa.T @ Xa + lam, Xa.T @ Y[tr])  # (K+1, P)
                pred = Xe @ beta
                ye = Y[te]
                mu = Xe[:, :1] @ beta[:1]
                we = W[te]
                out += np.sum(we * ((ye - mu) ** 2 - (ye - pred) ** 2), 0)  # variance explained beyond the mean
                tot += np.sum(we * (ye - (ye * we).sum(0) / np.maximum(we.sum(0), 1)) ** 2, 0)
            n = W.sum(0)
            return out / np.maximum(n, 1), tot / np.maximum(n, 1)

        img = 0.5 * (bA + bB).T  # (T, 5): the frame's own brightness, shift and blur from the whole field
        img[~keep] = np.nan
        imgs = box(img, 8)
        if args.fast:
            imgs = imgs - box(imgs, args.fast)
        imgs = (imgs - np.nanmean(imgs, 0)) / np.nanstd(imgs, 0)
        Cs = np.column_stack([Cs, imgs])
        names_all = list(C.keys()) + [f"image {n}" for n in names]
        sets = {"motion covariates (12)": list(C.keys()),
                "online only (6)": [k for k in C if "online" in k],
                "offline + error (6)": [k for k in C if "online" not in k],
                "image fit (5)": [f"image {n}" for n in names],
                "image fit + motion (17)": names_all}
        print("  B: residual after the network baseline (8-frame, % of baseline): out-of-sample explained RMS, real - shifted null;"
              " share of the residual variance")
        for sname, keys in sets.items():
            X = Cs[:, [names_all.index(k) for k in keys]]
            ex, tot = cv_explained(X)
            nulls = [cv_explained(np.roll(X, int(T * s), 0))[0] for s in (1 / 3, 2 / 3)]
            exn = 0.5 * (nulls[0] + nulls[1])
            line = f"    {sname:24s}"
            for cname, m in classes.items():
                d = np.mean(ex[m] - exn[m])
                line += f"  {cname}: {100 * np.sign(d) * np.sqrt(abs(d)):5.2f}% ({100 * d / np.mean(tot[m]):+.1f}% of var)"
            print(line, flush=True)
        tot = cv_explained(Cs[:, :1])[1]
        print("    residual RMS (8-frame) / photon-noise RMS: " + ", ".join(
            f"{c} {100 * np.sqrt(np.mean(tot[m])):.1f}% / {100 * np.sqrt(np.mean(noise8[m])):.1f}%" for c, m in classes.items()), flush=True)


if __name__ == "__main__":
    main()
