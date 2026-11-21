"""Same-frame neighbour predictability of a baseline's residual, by distance.

usage: python neighbor_predictability.py LABEL EXPORT.h5 (PRED.h5 KEY | truth)

(docs/plan.md, "Same-frame neighbour correlations" 2026-11-21.)

Residual r = (movie - activity - B) / B, B = a network baseline from a
prediction file, or the simulation's truth/baseline ("truth": r is then the
noise alone). Predictors: the same frame's sum(movie - B) / sum(B) over
  d1:   each of the 8 nearest pixels,
  d2:   8 angular sectors of the Chebyshev ring 2,
  d3-4: 8 sectors of rings 3-4 (d1-d4 are inside v9's hidden 9x9 box),
  d5-7: 8 sectors of rings 5-7 (visible to v9 in the output frame).
Models, cross-fitted on alternate 128-frame blocks, chance level (predictors
circularly shifted by T/3 and 2T/3) subtracted:
  pp:     coefficients per pixel, fitted to r (oracle);
  glob:   one coefficient vector for all sampled pixels;
  movie:  coefficients per pixel fitted to the pixel's own movie deviation
          (activity included; what zero-lag movie correlations supply), applied to r.
Bands: single frame; 8-frame mean minus a 33-frame mean ("fast").
Not scored: discarded frames, and frames within 2 s of the trial's edges.
Output: % of the residual's variance explained, per pixel class.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion_covariates import box  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
RINGS = {"d1": None, "d2": (2, 2), "d3-4": (3, 4), "d5-7": (5, 7)}


def kernels():
    ks = []
    dr, dc = np.mgrid[-7:8, -7:8]
    ch = np.maximum(abs(dr), abs(dc))
    sec = np.floor((np.arctan2(dr, dc) + np.pi) / (np.pi / 4) + 0.5).astype(int) % 8
    for name, rr in RINGS.items():
        if rr is None:
            for a, b in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                k = np.zeros((15, 15), np.float32)
                k[7 + a, 7 + b] = 1
                ks.append(k)
        else:
            ks += [((ch >= rr[0]) & (ch <= rr[1]) & (sec == s)).astype(np.float32) for s in range(8)]
    return np.stack(ks)  # (32, 15, 15)


def features(num, den, rs, cs):
    K = torch.from_numpy(kernels())[:, None].to(dev)
    T = num.shape[0]
    out = np.empty((T, len(rs), K.shape[0]), np.float32)
    ri, ci = torch.from_numpy(rs).to(dev), torch.from_numpy(cs).to(dev)
    for a in range(0, T, 32):
        n = torch.from_numpy(num[a:a + 32])[:, None].to(dev)
        d = torch.from_numpy(den[a:a + 32])[:, None].to(dev)
        cn = torch.nn.functional.conv2d(n, K, padding=7)[:, :, ri, ci]
        cd = torch.nn.functional.conv2d(d, K, padding=7)[:, :, ri, ci]
        f = torch.where(cd > 1e-3, cn / cd.clamp_min(1e-3), torch.full_like(cn, float("nan")))
        out[a:a + 32] = f.permute(0, 2, 1).cpu().numpy()
    return out


def cv(X, Y, W, blk, mode, Yfit=None, ridge=1e-2):
    """Out-of-sample variance explained beyond each pixel's mean, and the total
    variance, per pixel. X (T, P, K), Y (T, P), W (T, P) bool."""
    T, P, K = X.shape
    Xt = torch.from_numpy(np.nan_to_num(X)).to(dev)
    Yt = torch.from_numpy(np.nan_to_num(Y)).to(dev)
    Ft = Yt if Yfit is None else torch.from_numpy(np.nan_to_num(Yfit)).to(dev)
    Wt = torch.from_numpy(W).to(dev).float()
    ex, tot = torch.zeros(P, device=dev), torch.zeros(P, device=dev)
    for k in (0, 1):
        b = torch.from_numpy(blk == k).to(dev).float()[:, None]
        wtr, wte = Wt * b, Wt * (1 - b)
        ntr = wtr.sum(0).clamp_min(1)
        mx = (Xt * wtr[..., None]).sum(0) / ntr[:, None]
        mf = (Ft * wtr).sum(0) / ntr
        my = (Yt * wtr).sum(0) / ntr
        Xc = (Xt - mx) * Wt[..., None]
        Xw = Xc * wtr[..., None]
        if mode == "glob":
            A = torch.einsum("tpk,tpl->kl", Xw, Xc)
            c = torch.einsum("tpk,tp->k", Xw, Ft - mf)
            A = A + ridge * torch.diag(torch.diagonal(A))
            beta = torch.linalg.solve(A, c)[None].expand(P, K)
        else:
            A = torch.einsum("tpk,tpl->pkl", Xw, Xc)
            c = torch.einsum("tpk,tp->pk", Xw, Ft - mf)
            A = A + ridge * torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2)) + 1e-6 * torch.eye(K, device=dev)
            beta = torch.linalg.solve(A, c[..., None])[..., 0]
        pred = torch.einsum("tpk,pk->tp", Xc, beta)
        y = Yt - my
        ex += (wte * (y ** 2 - (y - pred) ** 2)).sum(0)
        yte = (Yt * wte).sum(0) / wte.sum(0).clamp_min(1)
        tot += (wte * (Yt - yte) ** 2).sum(0)
    n = Wt.sum(0).clamp_min(1)
    return (ex / n).cpu().numpy(), (tot / n).cpu().numpy()


def explained(Xs, rb, Wm, blk, mode, yb):
    T = Xs.shape[0]
    fit = yb if mode == "movie" else None
    m = "glob" if mode == "glob" else "pp"
    ex, tot = cv(Xs, rb, Wm, blk, m, Yfit=fit)
    if m == "glob":
        return ex, tot
    nul = [cv(np.roll(Xs, int(T * s), 0), rb, Wm, blk, m, Yfit=fit)[0] for s in (1 / 3, 2 / 3)]
    return ex - np.mean(nul, 0), tot


def run(label, exp, pred, key, rng):
    with h5py.File(exp, "r") as f:
        movie = f["movie"][0].astype(np.float32)
        act = np.nan_to_num(f["activity"][0].astype(np.float32))
        valid = f["mask/valid"][()].astype(bool)
        sup = f["mask/footprint_support"][()].astype(bool)
        keep = f["frame/discard"][()] == 0
        hz = float(np.ravel(f.attrs.get("alignHz", 80.0))[0])
        T, H, W_ = movie.shape
        if pred == "truth":
            B = f["truth/baseline"][()].astype(np.float32)
    if pred != "truth":
        with h5py.File(pred, "r") as g:
            Bp = g[f"baseline/{key}"][()].astype(np.float32)
            rows, cols = g["rows"][()], g["cols"][()]
        B = np.full((T, H, W_), np.nan, np.float32)
        B[:, rows, cols] = Bp
        del Bp
    B[:, ~valid] = np.nan
    movie[~keep] = np.nan
    ok = np.isfinite(movie) & np.isfinite(B) & (B > 0)
    num = np.where(ok, movie - B, 0).astype(np.float32)
    den = np.where(ok, B, 0).astype(np.float32)
    with np.errstate(invalid="ignore"):
        M = np.nanmean(np.where(ok, movie, np.nan), 0)
    cand = valid & np.isfinite(M) & (M > 0)
    sidx, bidx = np.flatnonzero(cand & sup), np.flatnonzero(cand & ~sup)
    idx = np.concatenate([rng.choice(sidx, min(5000, len(sidx)), replace=False),
                          rng.choice(bidx, min(5000, len(bidx)), replace=False)])
    rs, cs = (a.astype(np.int64) for a in np.unravel_index(idx, (H, W_)))
    X = features(num, den, rs, cs)
    del num, den
    Bs = B[:, rs, cs]
    y = (movie[:, rs, cs] - Bs) / Bs
    r = (movie[:, rs, cs] - act[:, rs, cs] - Bs) / Bs
    del movie, act, B
    gr, gc = np.zeros_like(M), np.zeros_like(M)
    gr[1:-1], gc[:, 1:-1] = (M[2:] - M[:-2]) / 2, (M[:, 2:] - M[:, :-2]) / 2
    steep = np.nan_to_num((np.hypot(gr, gc) / np.maximum(M, 1e-3))[rs, cs])
    classes = {"support": sup[rs, cs], "backgr": ~sup[rs, cs], "steep q": steep >= np.percentile(steep, 75)}
    e = int(round(2.0 * hz))
    tok = keep.copy()
    tok[:e] = False
    tok[T - e:] = False
    blk = (np.arange(T) // 128) % 2
    print(f"\n{label}: {T} frames, {tok.sum()} scored, {len(idx)} pixels ({classes['support'].sum()} support)", flush=True)
    X8, y8, r8 = (box(a.reshape(T, -1), 8).reshape(a.shape) for a in (X, y, r))
    bands = {"1 frame": (X, y, r),
             "8f fast": tuple(a - box(a.reshape(T, -1), 33).reshape(a.shape) for a in (X8, y8, r8))}
    hdr = "  " + " " * 15 + "".join(f"| {c:>7s}: pp   glob  movie " for c in classes)
    print(hdr)
    for bname, (Xb, yb, rb) in bands.items():
        Wm = tok[:, None] & np.isfinite(rb) & np.isfinite(yb)
        for j, ring in enumerate(RINGS):
            Xs = Xb[:, :, 8 * j: 8 * j + 8]
            res = {m: explained(Xs, rb, Wm, blk, m, yb) for m in ("pp", "glob", "movie")}
            line = f"  {bname:8s} {ring:5s} "
            for cname, cm in classes.items():
                line += "|         " + " ".join(f"{100 * res[m][0][cm].mean() / res[m][1][cm].mean():+5.1f}"
                                            for m in ("pp", "glob", "movie")) + " "
            print(line, flush=True)
        # all rings inside the box together, and everything
        for nm, sl in (("d1-4", slice(0, 24)), ("all", slice(0, 32))):
            ex, tot = explained(Xb[:, :, sl], rb, Wm, blk, "pp", yb)
            print(f"  {bname:8s} {nm:5s} " + "".join(f"|         {100 * ex[cm].mean() / tot[cm].mean():+5.1f}  (pp)       "
                                                  for cm in classes.values()), flush=True)


if __name__ == "__main__":
    label, exp = sys.argv[1], sys.argv[2]
    pred = sys.argv[3]
    key = sys.argv[4] if pred != "truth" else None
    run(label, exp, pred, key, np.random.default_rng(0))
