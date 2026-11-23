"""S4 held-out evaluation (docs/plan.md, "S4 results"): balanced scores per
checkpoint and paired differences on the test mice, with 95% intervals from a
hierarchical bootstrap (mice, then windows).

usage: python tools/s4_eval.py RESULTS_DIR

RESULTS_DIR holds one folder per evaluation run (eval.csv, windows.csv from
`slap2-f0 evaluate --per_window`). Checkpoints are named <arm>_<pre|ft|soup|wise>
(and S3_pre, S3_ft). All runs score the same windows (same seed and files), so
differences between runs are paired by window.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout_eval as he  # noqa: E402

ROOT = Path(sys.argv[1])
TEST = ["776270", "810196", "829704", "876767"]
VAL = ["878030", "794237", "834788", "804730"]
rows, W = he.load(ROOT)
cks = sorted({k[0] for k in W})
kinds = sorted({k[2] for k in W})
nw = len(next(iter(W.values()))["sup_n8"])
print("checkpoints:", cks); print("kinds:", kinds, "windows per mouse:", nw)

def arr(kind, pred, mice):
    a = np.zeros((len(cks), len(mice), 2, nw))
    for i, c in enumerate(cks):
        for j, m in enumerate(mice):
            d = W.get((c, m, kind))
            if d is None:
                a[i, j] = np.nan; continue
            for q, cl in enumerate(he.CLASSES):
                a[i, j, q] = d[f"{cl}_se8_{pred}"]
    return a

def scores(kind, mice, ref, wts=None, mm=None):
    num, den = arr(kind, "net", mice), arr(kind, ref, mice)
    if wts is None:
        wts = np.ones((1, len(mice), nw)); mm = np.ones((1, len(mice)))
    a = np.einsum("bmw,cmkw->bcmk", wts, num); b = np.einsum("bmw,cmkw->bcmk", wts, den)
    r = a / b  # (B, ck, mouse, class)
    per_mouse = r.mean(3)  # balanced: mean of classes
    return (per_mouse * mm[:, None, :]).sum(2) / mm.sum(1)[:, None]  # (B, ck)

REF = {k: ("mean_image" if k == "real" else "oracle") for k in kinds}
rng = np.random.default_rng(0)
B = 2000
ww, mm = he.boot_weights(B, len(TEST), nw, rng)
print(f"\nBalanced score on the test mice, % vs reference (real: mean image; simulated: oracle); lower is better")
print(f"{'checkpoint':16s}" + "".join(f"{k:>10s}" for k in kinds))
point, boot = {}, {}
for k in kinds:
    point[k] = scores(k, TEST, REF[k])[0]
    boot[k] = scores(k, TEST, REF[k], ww, mm)
for i, c in enumerate(cks):
    print(f"{c:16s}" + "".join(f"{100 * (point[k][i] - 1):+10.2f}" for k in kinds))

def diff(a, b, kinds_=("real", "fast", "free", "sim", "fastdim", "freedim")):
    ia, ib = cks.index(a), cks.index(b)
    out = []
    for k in kinds_:
        if k not in point: continue
        d = 100 * (point[k][ia] - point[k][ib]); bd = 100 * (boot[k][:, ia] - boot[k][:, ib])
        lo, hi = np.nanpercentile(bd, [2.5, 97.5])
        out.append(f"{k} {d:+.2f} [{lo:+.2f},{hi:+.2f}]")
    print(f"  {a} - {b}: " + "; ".join(out))

print("\nPaired differences (points of the ratio x100, 95% hierarchical bootstrap: mice, then windows)")
for arm in ("seed1", "lowcount", "edge", "deep"):
    for t in ("wise", "ft"):
        if f"{arm}_{t}" in cks: diff(f"{arm}_{t}", f"ctrl_{t}")
for arm in ("ctrl", "seed1", "lowcount", "edge", "deep"):
    if f"{arm}_wise" in cks:
        diff(f"{arm}_wise", f"{arm}_ft"); diff(f"{arm}_soup", f"{arm}_ft")
if "S3_ft" in cks:
    diff("ctrl_ft", "S3_ft"); diff("ctrl_pre", "S3_pre"); diff("ctrl_wise", "S3_ft")
