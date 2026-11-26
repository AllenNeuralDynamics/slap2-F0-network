"""Scores where the target does not rest on GIAnT's activity fit, from an evaluation's eval.csv (docs/plan.md, "S6",
2026-11-26): labeled pixels outside every footprint in real windows (target = the movie itself), and
activity-free simulations. 8-frame squared error of the net over the mean image's (%, lower is better) and bias
as % of the target (positive = baseline below the target). "free all" and "fdim all" are the mean of the support,
labeled and background classes.

usage: python tools/trusted_scores.py EVAL_DIR [--ckpts a,b,...] [--pairs a-b,...]
                                      [--val 878030,...] [--test 776270,...]

EVAL_DIR holds eval.csv from `slap2-f0 evaluate` with the kinds real, free and freedim. The checkpoints are those
in eval.csv, in the order they first appear (--ckpts picks and orders them). --pairs lists the paired differences
of the labeled-pixel score to print (default: each checkpoint against each later one). The split defaults to
tools/heldout_eval.py's.
"""
import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("root", type=Path, metavar="EVAL_DIR")
    p.add_argument("--ckpts", default="", help="checkpoints to report, in this order (default: all in eval.csv)")
    p.add_argument("--pairs", default="", help="paired differences a-b,... (default: each against each later one)")
    p.add_argument("--val", default="878030,794237,834788,804730", help="validation mice (as tools/heldout_eval.py)")
    p.add_argument("--test", default="776270,810196,829704,876767", help="held-out mice (as tools/heldout_eval.py)")
    a = p.parse_args()
    with open(a.root / "eval.csv") as fh:
        rows = list(csv.DictReader(fh))
    test, val = a.test.split(","), a.val.split(",")
    R = {(r["checkpoint"], r["session"].split("_")[0], r["kind"]): r for r in rows}
    cks = a.ckpts.split(",") if a.ckpts else list(dict.fromkeys(r["checkpoint"] for r in rows))
    pairs = [tuple(x.split("-", 1)) for x in a.pairs.split(",")] if a.pairs else list(combinations(cks, 2))

    def score(ck, m, kind, cl, ref="mean_image"):
        r = R[(ck, m, kind)]
        return 100 * (float(r[f"{cl}_mse8_net"]) / float(r[f"{cl}_mse8_{ref}"]) - 1)

    def bias(ck, m, kind, cl):
        r = R[(ck, m, kind)]
        return 100 * float(r[f"{cl}_bias_net"]) / float(r[f"{cl}_mean_target"])

    for label, mice in (("held-out", test), ("validation", val), (f"all {len(test + val)}", test + val)):
        print(f"\n{label} mice: mean over mice")
        print(f"{'':12s}{'lab real':>10s}{'bg real':>10s}{'free sup':>10s}{'free lab':>10s}{'free bg':>10s}{'fdim sup':>10s}{'fdim bg':>10s}   | bias: {'lab real':>9s}{'bg real':>9s}{'free all':>9s}{'fdim all':>9s}")
        for ck in cks:
            s = [np.mean([score(ck, m, k, c) for m in mice]) for k, c in
                 (("real", "lab"), ("real", "bg"), ("free", "sup"), ("free", "lab"), ("free", "bg"), ("freedim", "sup"), ("freedim", "bg"))]
            b = [np.mean([bias(ck, m, "real", "lab") for m in mice]), np.mean([bias(ck, m, "real", "bg") for m in mice]),
                 np.mean([np.mean([bias(ck, m, "free", c) for c in ("sup", "lab", "bg")]) for m in mice]),
                 np.mean([np.mean([bias(ck, m, "freedim", c) for c in ("sup", "lab", "bg")]) for m in mice])]
            print(f"{ck:12s}" + "".join(f"{v:+10.2f}" for v in s) + "   |       " + "".join(f"{v:+9.2f}" for v in b))
    mice = test + val
    print("\nlab real score per mouse (% vs mean image)")
    print(f"{'':12s}" + "".join(f"{m:>9s}" for m in mice))
    for ck in cks:
        print(f"{ck:12s}" + "".join(f"{score(ck, m, 'real', 'lab'):+9.2f}" for m in mice))
    print(f"paired differences over the {len(mice)} mice, lab real score: mean, and number of mice where the first is better")
    for x, y in pairs:
        d = np.array([score(x, m, "real", "lab") - score(y, m, "real", "lab") for m in mice])
        print(f"  {x} - {y}: {d.mean():+.2f} (range {d.min():+.2f} to {d.max():+.2f}); better in {int((d < 0).sum())} of {len(mice)}")


if __name__ == "__main__":
    main()
