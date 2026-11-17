"""Compare the sources of two SILo runs on the same data (e.g. moving-median vs
network localization baselines): counts, one-to-one matches at 1.5 px, and the
extracted sources' SNR (sources/temporal/SNR) for matched and unmatched ones.

usage: python compare_silo_runs.py A/experiment_summary.h5 B/experiment_summary.h5 [--names a,b] [--radius 1.5]
"""
import argparse

import h5py
import numpy as np


def match(a, b, radius):
    """Index pairs (i, j) of one-to-one matches within radius, closest first."""
    if not len(a) or not len(b):
        return []
    d = np.hypot(*(np.subtract.outer(a[:, k], b[:, k]) for k in range(2)))
    i, j = np.nonzero(d <= radius)
    pairs, ua, ub = [], set(), set()
    for k in np.argsort(d[i, j], kind="stable"):
        if i[k] not in ua and j[k] not in ub:
            ua.add(i[k]); ub.add(j[k]); pairs.append((i[k], j[k]))
    return pairs


def load(fn, path):
    with h5py.File(fn, "r") as f:
        g = f[path]
        xy = np.asarray(g["sources/spatial/coords"][()]).T[:, 1:3]  # MATLAB order: rows [z, y, x]
        snr = np.asarray(g["sources/temporal/SNR"][()]).ravel()
        seeds = np.asarray(g["visualizations/act_im_peaks"][()]).T[:, 1:3]
    return xy, snr, seeds


ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--names", default="A,B")
ap.add_argument("--radius", type=float, default=1.5)
args = ap.parse_args()
na, nb = args.names.split(",")
with h5py.File(args.a, "r") as f:
    paths = [k for k in f.keys() if k.startswith("Path")]
for path in paths:
    xa, sa, pa = load(args.a, path)
    xb, sb, pb = load(args.b, path)
    pairs = match(xa, xb, args.radius)
    ia = np.array([p[0] for p in pairs], int)
    ib = np.array([p[1] for p in pairs], int)
    only_a = np.setdiff1d(np.arange(len(xa)), ia)
    only_b = np.setdiff1d(np.arange(len(xb)), ib)
    seeds = match(pa, pb, args.radius)
    med = lambda x: float(np.median(x)) if len(x) else float("nan")  # noqa: E731
    print(f"{path}: sources {na} {len(xa)}, {nb} {len(xb)}, matched {len(pairs)} "
          f"(seeds {len(pa)} / {len(pb)}, matched {len(seeds)})")
    print(f"  SNR median: {na} {med(sa):.2f}, {nb} {med(sb):.2f}; matched pairs {na} {med(sa[ia]):.2f} {nb} {med(sb[ib]):.2f} "
          f"(ratio {nb}/{na} {med(sb[ib] / sa[ia]) if len(ia) else float('nan'):.3f}); "
          f"{na} only {len(only_a)} (SNR {med(sa[only_a]):.2f}), {nb} only {len(only_b)} (SNR {med(sb[only_b]):.2f})")
