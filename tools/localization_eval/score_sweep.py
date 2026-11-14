"""Precision/recall/F1 vs peakth for both settings, pooled over simulations.

The grid shift per simulation is taken from score_sims.py's movmedian result
at the default peakth (so every threshold and setting uses the same frame).

usage: python score_sweep.py <capsule code dir> <score_sims.json> <out.json> <name>=<sweep_flat.mat>,<groundtruth.h5>,<nFramesKept> ...
"""
import json
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, sys.argv[1])
from giant_eval import metrics as M  # noqa: E402

base = json.load(open(sys.argv[2]))
SETTINGS = ["movmedian", "splitFreq"]
out = {"per_sim": {}, "pooled": {}}
pooled = {}
for arg in sys.argv[4:]:
    name, rest = arg.split("=", 1)
    fn, gtfn, nk = rest.split(",")
    flat = sio.loadmat(fn)
    gt = M.load_ground_truth(gtfn, int(nk))
    b = base[name]["movmedian"]
    rc = np.stack([gt["R"], gt["C"]], 1).astype(float)
    gt_rc = rc[:, ::-1] if b["transposed"] else rc
    off = np.array(b["offset"])
    ths = np.ravel(flat["ths"])
    res = {}
    for s in SETTINGS:
        for th in ths:
            key = f"Path1_{s}_th{int(round(10 * th))}_RCV"
            rcv = np.atleast_2d(flat[key]).reshape(-1, 3)
            det = rcv[:, :2] - 1.0
            pairs, dist = M.match_points(det, gt_rc + off, 2.0)
            p, r, f1 = M.detection_scores(len(det), len(gt_rc), len(pairs))
            res[f"{s}@{th:g}"] = {"n": int(len(det)), "matched": int(len(pairs)), "precision": p, "recall": r, "f1": f1,
                                  "loc_err": M._nanmedian(dist)}
            acc = pooled.setdefault((s, float(th)), [0, 0, 0])
            acc[0] += len(det); acc[1] += len(gt_rc); acc[2] += len(pairs)
    out["per_sim"][name] = res
for (s, th), (npred, ntrue, nm) in sorted(pooled.items()):
    p, r, f1 = M.detection_scores(npred, ntrue, nm)
    out["pooled"][f"{s}@{th:g}"] = {"n": npred, "n_gt": ntrue, "matched": nm, "precision": p, "recall": r, "f1": f1}
    print(f"{s:10s} peakth {th:5.1f}: n {npred:4d} matched {nm:3d}/{ntrue}  P {p:.3f} R {r:.3f} F1 {f1:.3f}")
json.dump(out, open(sys.argv[3], "w"), indent=1)
