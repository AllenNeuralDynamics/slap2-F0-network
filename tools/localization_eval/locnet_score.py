"""Score locnet_run results: sources found with each baseline against the
template's GIAnT sources (simulations: the true sources) at 2 px, one-to-one (closest pairs first).

usage: python locnet_score.py RESULT.json [RESULT.json ...] [--radius 2] [--csv out.csv]
Prints, per trial and baseline, the number of sources, precision, recall and
F1 at the default peak threshold, and the best F1 over the threshold sweep.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def match(r1, c1, r2, c2, radius):
    """One-to-one matches within radius, closest pairs first."""
    if not len(r1) or not len(r2):
        return 0
    d = np.hypot(np.subtract.outer(r1, r2), np.subtract.outer(c1, c2))
    i, j = np.nonzero(d <= radius)
    used1, used2, m = set(), set(), 0
    for k in np.argsort(d[i, j], kind="stable"):
        if i[k] not in used1 and j[k] not in used2:
            used1.add(i[k]); used2.add(j[k]); m += 1
    return m


def prf(found, gt, m):
    p = m / found if found else float("nan")
    r = m / gt if gt else float("nan")
    f = 2 * m / (found + gt) if found + gt else float("nan")
    return p, r, f


ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+")
ap.add_argument("--radius", type=float, default=2.0)
ap.add_argument("--csv", default="")
args = ap.parse_args()
rows = []
pooled = {}  # (baseline, kind) -> threshold -> [found, true, matched] summed over trials


def kind_of(stem):
    """Simulation kind from the folder name (<session>_<kind><seed>), else "real"."""
    tail = stem.rsplit("_", 1)[-1]
    return tail.rstrip("0123456789") if tail.rstrip("0123456789") in ("sim", "fast") else "real"


for fn in args.results:
    J = json.loads(Path(fn).read_text())
    gr, gc = np.atleast_1d(J.get("gt_row", [])).astype(float), np.atleast_1d(J.get("gt_col", [])).astype(float)
    ths = np.atleast_1d(J["peakths"]).astype(float)
    names = J["names"] if isinstance(J["names"], list) else [J["names"]]
    for name in names:
        key = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
        e = J[key]
        sweep = e["sweep"] if isinstance(e["sweep"], list) else [e["sweep"]]
        best = (-1, None)
        out = {}
        for th, sw in zip(ths, sweep):
            r, c = np.atleast_1d(sw["row"]).astype(float), np.atleast_1d(sw["col"]).astype(float)
            m = match(r, c, gr, gc, args.radius)
            acc = pooled.setdefault((name, kind_of(Path(fn).stem)), {}).setdefault(float(th), [0, 0, 0])
            acc[0] += len(r); acc[1] += len(gr); acc[2] += m
            p, rc, f = prf(len(r), len(gr), m)
            if th == J["peakth_default"]:
                out = dict(n=len(r), matched=m, precision=p, recall=rc, f1=f)
            if f == f and f > best[0]:
                best = (f, th)
        rows.append(dict(trial=Path(fn).stem, baseline=name, n_gt=len(gr), **out,
                         best_f1=best[0] if best[1] is not None else float("nan"),
                         best_peakth=best[1] if best[1] is not None else float("nan"),
                         seconds=e.get("seconds")))
w = max(len(r["trial"]) for r in rows)
print(f"{'trial':{w}} {'baseline':>14} {'n_gt':>5} {'n':>5} {'match':>5} {'prec':>6} {'rec':>6} {'F1':>6} {'bestF1':>6} {'@th':>4}")
for r in rows:
    print(f"{r['trial']:{w}} {r['baseline']:>14} {r['n_gt']:5d} {r['n']:5d} {r['matched']:5d} {r['precision']:6.3f} "
          f"{r['recall']:6.3f} {r['f1']:6.3f} {r['best_f1']:6.3f} {r['best_peakth']:4.0f}")
if args.csv:
    with open(args.csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

print()
print("pooled over trials (sum of found, true, matched): at the default threshold, and at each baseline's best single threshold")
print(f"{'kind':>5} {'baseline':>14} {'prec':>6} {'rec':>6} {'F1':>6} | {'th':>4} {'prec':>6} {'rec':>6} {'F1':>6}")
default = None
for fn in args.results[:1]:
    default = float(json.loads(Path(fn).read_text())["peakth_default"])
for (name, kind), byth in sorted(pooled.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    if not any(v[1] for v in byth.values()):
        continue
    stats = {th: prf(*v[:2], v[2]) for th, v in byth.items()}
    p0, r0, f0 = stats.get(default, (float("nan"),) * 3)
    th_best = max(stats, key=lambda t: stats[t][2] if stats[t][2] == stats[t][2] else -1)
    pb, rb, fb = stats[th_best]
    print(f"{kind:>5} {name:>14} {p0:6.3f} {r0:6.3f} {f0:6.3f} | {th_best:4.1f} {pb:6.3f} {rb:6.3f} {fb:6.3f}")
