"""Compare movmedian vs splitFreq localization on 780125 with the final GIAnT sources.

usage: python analyze_real.py <results_flat.mat> <experiment_summary.h5> <capsule code dir> [out.json]
"""
import json
import sys

import h5py
import numpy as np
import scipy.io as sio
from scipy.signal import correlate

sys.path.insert(0, sys.argv[3])
from giant_eval.metrics import match_points  # noqa: E402  (Hungarian, as the capsule)

flat = sio.loadmat(sys.argv[1])
es = h5py.File(sys.argv[2], "r")
SETTINGS = ["movmedian", "splitFreq"]
MATCH = 1.5


def ref_points(path):
    # MATLAB-order file: h5py arrays are transposed; rows are [z, y, x], 0-indexed
    seeds = np.asarray(es[f"{path}/visualizations/act_im_peaks"][()]).T[:, 1:3]
    coords = np.asarray(es[f"{path}/sources/spatial/coords"][()]).T[:, 1:3]
    mean_im = np.asarray(es[f"{path}/visualizations/mean_im"][()]).T  # C x 1 x H x W in MATLAB order
    mean_im = np.nansum(mean_im.reshape(mean_im.shape[0], -1, mean_im.shape[-2], mean_im.shape[-1])[:, 0], 0)
    return seeds, coords, mean_im


def xcorr_offset(a, b, maxs=15):
    """(dy, dx) such that a[y + dy, x + dx] ~ b[y, x] (integer, |shift| <= maxs)."""
    a = np.nan_to_num(a - np.nanmean(a)); b = np.nan_to_num(b - np.nanmean(b))
    h = max(a.shape[0], b.shape[0]); w = max(a.shape[1], b.shape[1])
    A = np.zeros((h, w)); A[:a.shape[0], :a.shape[1]] = a
    B = np.zeros((h, w)); B[:b.shape[0], :b.shape[1]] = b
    c = correlate(A, B, mode="full", method="fft")
    cy, cx = h - 1, w - 1
    sub = c[cy - maxs:cy + maxs + 1, cx - maxs:cx + maxs + 1]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    return np.array([iy - maxs, ix - maxs], float)


def consistency(points, peaks, valid_trials, radius=MATCH):
    """Fraction of points with a per-trial peak within radius in >= half the valid trials."""
    if not len(points):
        return float("nan"), np.zeros(0)
    counts = np.zeros(len(points))
    for t in valid_trials:
        pk = peaks[peaks[:, 0] == t][:, 1:3]
        if not len(pk):
            continue
        d = np.sqrt(((points[:, None, :] - pk[None, :, :]) ** 2).sum(-1))
        counts += d.min(1) <= radius
    frac = counts / len(valid_trials)
    return float(np.mean(frac >= 0.5)), frac


def robust_z(ai, rc):
    """(actIM - median) / (MAD/0.6745) at the nearest pixel of each point, as getActImPeaks thresholds."""
    v = ai[np.isfinite(ai)]
    mu = np.median(v); sig = np.median(np.abs(v - mu)) / 0.6741891400433162
    ij = np.round(rc).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < ai.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < ai.shape[1])
    z = np.full(len(rc), np.nan)
    z[ok] = (ai[ij[ok, 0], ij[ok, 1]] - mu) / sig
    return z


def summ(v):
    v = np.asarray(v, float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)), "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)), "min": float(v.min()), "max": float(v.max())}


out = {}
for path in ["Path1", "Path2"]:
    if f"{path}_meanIM" not in flat:
        continue
    seeds, coords, ref_mean = ref_points(path)
    mean_im = flat[f"{path}_meanIM"]
    mean_im = np.nansum(mean_im, 2) if mean_im.ndim == 3 else mean_im
    R = {}
    for s in SETTINGS:
        rcv = np.atleast_2d(flat[f"{path}_{s}_RCV"]).reshape(-1, 3)
        R[s] = {"rc": rcv[:, :2] - 1.0, "v": rcv[:, 2],
                "peaks": np.atleast_2d(flat[f"{path}_{s}_peaks"]).reshape(-1, 4),
                "valid": np.atleast_1d(flat[f"{path}_{s}_validTrials"]).ravel().astype(int),
                "normFac": float(np.ravel(flat[f"{path}_{s}_normFac"])[0]),
                "actIM": flat[f"{path}_{s}_actIM"],
                "actIMraw": flat[f"{path}_{s}_actIMraw"],
                "nPre": int(np.ravel(flat[f"{path}_{s}_nPeaksBeforePrune"])[0])}
        R[s]["peaks"][:, 1:3] -= 1.0  # 0-indexed

    # frame offset: harness frame -> reference frame (ref ~ harness + off)
    img_off = xcorr_offset(ref_mean, mean_im)
    pairs, _ = match_points(R["movmedian"]["rc"] + img_off, seeds, 3.0)
    off = img_off.copy()
    if len(pairs) >= 3:
        off = off + np.median(seeds[pairs[:, 1]] - (R["movmedian"]["rc"][pairs[:, 0]] + img_off), 0)
    po = {"ref_n_seeds": int(len(seeds)), "harness_to_ref_offset_img": img_off.tolist(),
          "harness_to_ref_offset": off.tolist(), "ref_mean_im_shape": list(ref_mean.shape),
          "harness_mean_im_shape": list(mean_im.shape)}

    for s in SETTINGS:
        r = R[s]
        rc_ref = r["rc"] + off
        p_seed, d_seed = match_points(rc_ref, seeds, MATCH)
        p_cent, d_cent = match_points(rc_ref, coords, MATCH)
        own_cons, own_frac = consistency(r["rc"], r["peaks"], r["valid"])
        ref_cons, ref_frac = consistency(seeds - off, r["peaks"], r["valid"])
        n_trial_peaks = [int((r["peaks"][:, 0] == t).sum()) for t in r["valid"]]
        # activity-image value at the reference seeds (harness frame, nearest pixel)
        ai = r["actIM"]
        ij = np.round(seeds - off).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < ai.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < ai.shape[1])
        v_at_ref = robust_z(ai, seeds - off)[ok]
        r["z"] = robust_z(ai, r["rc"])
        po[s] = {
            "n_sources": int(len(r["rc"])), "n_peaks_before_prune": r["nPre"],
            "n_valid_trials": int(len(r["valid"])), "normFac": r["normFac"],
            "actIM_raw_max": float(np.nanmax(r["actIMraw"])),
            "match_ref_seeds": int(len(p_seed)), "match_ref_seeds_err_median": float(np.median(d_seed)) if len(d_seed) else None,
            "match_ref_coords": int(len(p_cent)),
            "self_consistency_frac": own_cons, "ref_seed_consistency_frac": ref_cons,
            "self_trial_frac_mean": float(np.mean(own_frac)), "ref_trial_frac_mean": float(np.mean(ref_frac)),
            "per_trial_peaks_median": float(np.median(n_trial_peaks)) if n_trial_peaks else None,
            "peak_values": summ(r["v"]), "peak_values_raw": summ(r["v"] * r["normFac"]),
            "actIM_z_at_ref_seeds": summ(v_at_ref[np.isfinite(v_at_ref)]),
            "peak_z": summ(r["z"][np.isfinite(r["z"])]),
            "frac_ref_seeds_z_above_8": float(np.mean(v_at_ref[np.isfinite(v_at_ref)] >= 8)),
        }
    # settings vs each other
    a, b = R["movmedian"], R["splitFreq"]
    pairs, dist = match_points(a["rc"], b["rc"], MATCH)
    only_a = np.setdiff1d(np.arange(len(a["rc"])), pairs[:, 0])
    only_b = np.setdiff1d(np.arange(len(b["rc"])), pairs[:, 1])
    po["between"] = {"matched": int(len(pairs)), "only_movmedian": int(len(only_a)), "only_splitFreq": int(len(only_b)),
                     "shared_peak_z_movmedian": summ(a["z"][pairs[:, 0]]),
                     "shared_peak_z_splitFreq": summ(b["z"][pairs[:, 1]]),
                     "only_movmedian_peak_z": summ(a["z"][only_a]),
                     "only_splitFreq_peak_z": summ(b["z"][only_b])}
    # are the unique sources reference sources?
    for name, r, idx in (("only_movmedian", a, only_a), ("only_splitFreq", b, only_b)):
        if len(idx):
            p, _ = match_points(r["rc"][idx] + off, seeds, MATCH)
            po["between"][f"{name}_in_ref"] = int(len(p))
        else:
            po["between"][f"{name}_in_ref"] = 0
    out[path] = po

txt = json.dumps(out, indent=1)
print(txt)
if len(sys.argv) > 4:
    with open(sys.argv[4], "w") as f:
        f.write(txt)
