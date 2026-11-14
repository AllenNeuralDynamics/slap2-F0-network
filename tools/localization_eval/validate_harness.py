"""Check that the harness's movmedian seeds reproduce a SILo run's act_im_peaks and act_im.

usage: python validate_harness.py <results_flat.mat> <experiment_summary.h5> [Path1 ...]
"""
import sys

import h5py
import numpy as np
import scipy.io as sio

flat = sio.loadmat(sys.argv[1])
paths = sys.argv[3:] or ["Path1"]
with h5py.File(sys.argv[2], "r") as f:
    for p in paths:
        seeds = np.asarray(f[f"{p}/visualizations/act_im_peaks"][()]).T[:, 1:3] + 1  # 1-indexed
        act = np.asarray(f[f"{p}/visualizations/act_im"][()]).T
        act = act.reshape(act.shape[-2], act.shape[-1])
        rcv = np.atleast_2d(flat[f"{p}_movmedian_RCV"]).reshape(-1, 3)
        mine = flat[f"{p}_movmedian_actIM"]
        a = {tuple(x) for x in seeds.astype(int)}
        b = {tuple(x) for x in rcv[:, :2].astype(int)}
        same_shape = act.shape == mine.shape
        diff = np.nanmax(np.abs(act - mine)) if same_shape else float("nan")
        print(f"{p}: SILo seeds {len(a)}, harness {len(b)}, identical positions {len(a & b)}; "
              f"act_im shapes {act.shape} vs {mine.shape}, max |diff| {diff:.3g}, "
              f"NaN pattern equal {bool(np.array_equal(np.isnan(act), np.isnan(mine))) if same_shape else 'n/a'}")
