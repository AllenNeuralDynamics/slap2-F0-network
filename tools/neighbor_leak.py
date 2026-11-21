"""If a baseline used the same frame's nearby pixels to predict v9's residual,
how much of the pixel's own event would it follow?

usage: python neighbor_leak.py EXPORT.h5 PRED.h5

Fits r = (movie - activity - B) / B on the same-frame features of
neighbor_corr2 (d1, and d1-4) by least squares on all scored frames (one
coefficient vector for all sampled pixels, and per pixel). Then applies the
coefficients to the same features computed from GIAnT's activity alone,
sum(activity) / sum(B), and regresses that on the pixel's own activity / B over
support pixels: slope 1 = the baseline would follow the whole event, 0 = none.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neighbor_predictability import features  # noqa: E402

exp, pred = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
with h5py.File(exp, "r") as f:
    movie = f["movie"][0].astype(np.float32)
    act = np.nan_to_num(f["activity"][0].astype(np.float32))
    valid = f["mask/valid"][()].astype(bool)
    sup = f["mask/footprint_support"][()].astype(bool)
    keep = f["frame/discard"][()] == 0
T, H, W = movie.shape
with h5py.File(pred, "r") as g:
    B = np.full((T, H, W), np.nan, np.float32)
    B[:, g["rows"][()], g["cols"][()]] = g["baseline/v9_finetuned"][()].astype(np.float32)
B[:, ~valid] = np.nan
movie[~keep] = np.nan
ok = np.isfinite(movie) & np.isfinite(B) & (B > 0)
den = np.where(ok, B, 0).astype(np.float32)
idx = np.flatnonzero(valid & sup & ok.any(0))
idx = rng.choice(idx, min(6000, len(idx)), replace=False)
rs, cs = (a.astype(np.int64) for a in np.unravel_index(idx, (H, W)))
X = features(np.where(ok, movie - B, 0).astype(np.float32), den, rs, cs)
Xa = features(np.where(ok, act, 0).astype(np.float32), den, rs, cs)
Bs = B[:, rs, cs]
r = (movie[:, rs, cs] - act[:, rs, cs] - Bs) / Bs
a = act[:, rs, cs] / Bs
e = 160
tok = keep.copy(); tok[:e] = False; tok[T - e:] = False
Wm = tok[:, None] & np.isfinite(r) & np.isfinite(a) & np.all(np.isfinite(X[:, :, :24]), 2)
good = Wm.sum(0) > 200  # pixels with enough scored frames
idx, X, Xa, r, a, Wm = idx[good], X[:, good], Xa[:, good], r[:, good], np.nan_to_num(a[:, good]), Wm[:, good]
print(f"{Path(exp).parent.parent.name}: {len(idx)} support pixels; "
      f"activity / baseline at the pixel: mean {np.mean(a[Wm]):.3f}, 99th pct {np.percentile(a[Wm], 99):.2f}")
for nm, sl in (("d1", slice(0, 8)), ("d1-4", slice(0, 24))):
    x, xa = np.nan_to_num(X[:, :, sl]), np.nan_to_num(Xa[:, :, sl])
    w = Wm.astype(np.float64)
    n = w.sum(0)
    mx = (x * w[..., None]).sum(0) / n[:, None]
    mr = (np.nan_to_num(r) * w).sum(0) / n
    xc = (x - mx) * w[..., None]
    rc = (np.nan_to_num(r) - mr) * w
    K = x.shape[2]
    # one vector for all pixels
    A = np.einsum("tpk,tpl->kl", xc, xc); c = np.einsum("tpk,tp->k", xc, rc)
    bg = np.linalg.solve(A + 1e-2 * np.diag(np.diag(A)), c)
    # per pixel
    Ap = torch.einsum("tpk,tpl->pkl", torch.from_numpy(xc), torch.from_numpy(xc)).numpy()
    cp = np.einsum("tpk,tp->pk", xc, rc)
    bp = np.linalg.solve(Ap + 1e-2 * np.einsum("pkk->pk", Ap)[..., None] * np.eye(K) + 1e-9 * np.eye(K), cp[..., None])[..., 0]
    for mname, beta in (("one vector", np.broadcast_to(bg, (len(idx), K))), ("per pixel", bp)):
        pa = np.einsum("tpk,pk->tp", xa, beta)  # what the activity alone puts into the prediction
        ac = a - (a * w).sum(0) / n
        sl_all = np.sum(w * pa * ac) / np.sum(w * ac * ac)
        big = Wm & (a > np.percentile(a[Wm], 99))
        print(f"  {nm:5s} {mname:10s}: sum of coefficients {beta.sum(1).mean():+.2f}; the prediction follows "
              f"{100 * sl_all:.0f}% of the pixel's own activity (slope); in the top 1% of activity frames, "
              f"{100 * np.mean(pa[big]) / np.mean(a[big]):.0f}% of it")
