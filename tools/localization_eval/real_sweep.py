"""Real data: sources per peakth for both settings and overlap with the final GIAnT seeds (1.5 px)."""
import sys
import h5py, numpy as np, scipy.io as sio
sys.path.insert(0, sys.argv[3])
from giant_eval.metrics import match_points
flat = sio.loadmat(sys.argv[1])
es = h5py.File(sys.argv[2], "r")
for p in ["Path1", "Path2"]:
    seeds = np.asarray(es[f"{p}/visualizations/act_im_peaks"][()]).T[:, 1:3]
    for th in np.ravel(flat["ths"]):
        row = []
        rcs = {}
        for s in ["movmedian", "splitFreq"]:
            rc = np.atleast_2d(flat[f"{p}_{s}_th{int(round(10*th))}_RCV"]).reshape(-1, 3)[:, :2] - 1
            rcs[s] = rc
            row.append(f"{s} n={len(rc):3d} inRef={len(match_points(rc, seeds, 1.5)[0]):3d}")
        both = len(match_points(rcs["movmedian"], rcs["splitFreq"], 1.5)[0])
        print(f"{p} (ref {len(seeds)}) peakth {th:4.1f}: " + " | ".join(row) + f" | shared {both}")
