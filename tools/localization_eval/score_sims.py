"""Score movmedian vs splitFreq localization against simulation ground truth.

Uses the integration-test capsule's giant_eval.metrics: gt_frame (grid shift
between simulated and registered frames, from the activity image since there
are no extracted footprints, refined from matched positions), match_points
(Hungarian) at 2 px, detection_scores and the median localization error.

usage: python score_sims.py <capsule code dir> <out.json> <name>=<results_flat.mat>,<groundtruth.h5>,<nFramesKept> ...
Optional: <name>=...,<nFramesKept>,<experiment_summary.h5> also scores that SILo run's seeds (act_im_peaks).
"""
import json
import sys

import h5py
import numpy as np
import scipy.io as sio

sys.path.insert(0, sys.argv[1])
from giant_eval import metrics as M  # noqa: E402

MATCH_PX = 2.0
SETTINGS = ["movmedian", "splitFreq"]


def score(det_rc, act_im, gt, offset=None):
    src = {"coords": np.c_[np.zeros(len(det_rc)), det_rc], "n": len(det_rc), "profiles": None}
    summary = {"Path1": {"visualizations": {"act_im": act_im}}}
    fr = M.gt_frame(src, summary, "Path1", gt, MATCH_PX)
    gt_rc = fr["gt_rc"]
    off = fr["offset"] if offset is None else offset
    pairs, dist = M.match_points(det_rc, gt_rc + off, MATCH_PX)
    p, r, f1 = M.detection_scores(len(det_rc), gt_rc.shape[0], len(pairs))
    return {"n_sources": int(len(det_rc)), "n_gt": int(gt_rc.shape[0]), "n_matched": int(len(pairs)),
            "precision": p, "recall": r, "f1": f1, "loc_err_px_median": M._nanmedian(dist),
            "offset": [float(x) for x in off], "transposed": bool(fr["transposed"]),
            "matched_gt": sorted(int(j) for j in pairs[:, 1]) if len(pairs) else []}


out = {}
for arg in sys.argv[3:]:
    name, rest = arg.split("=", 1)
    parts = rest.split(",")
    flat = sio.loadmat(parts[0])
    gt = M.load_ground_truth(parts[1], int(parts[2]))
    res = {}
    fr_offsets = {}
    for s in SETTINGS:
        rcv = np.atleast_2d(flat[f"Path1_{s}_RCV"]).reshape(-1, 3)
        det = rcv[:, :2] - 1.0  # MATLAB 1-indexed -> 0-indexed (as act_im_peaks)
        act = np.nan_to_num(flat[f"Path1_{s}_actIM"].astype(float))
        res[s] = score(det, act, gt)
        res[s]["normFac"] = float(np.ravel(flat[f"Path1_{s}_normFac"])[0])
        res[s]["peak_values_median"] = float(np.median(rcv[:, 2])) if len(rcv) else None
        fr_offsets[s] = (det, act)
    # the same grid shift for both settings (movmedian's), as a check
    det, act = fr_offsets["splitFreq"]
    res["splitFreq_with_movmedian_offset"] = score(det, act, gt, offset=np.array(res["movmedian"]["offset"]))
    if len(parts) > 3:
        with h5py.File(parts[3], "r") as f:
            seeds = np.asarray(f["Path1/visualizations/act_im_peaks"][()]).T[:, 1:3]
            act = np.asarray(f["Path1/visualizations/act_im"][()]).T
            act = np.nan_to_num(act.reshape(act.shape[-2], act.shape[-1]).astype(float))
        res["silo_seeds"] = score(seeds, act, gt)
    out[name] = res
    print(name, {k: (v["n_sources"], v["n_matched"], round(v["f1"], 3), round(v["loc_err_px_median"], 3), v["offset"])
                 for k, v in res.items()})

with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=1)
