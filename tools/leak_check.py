"""How much activity leaks into network baselines.

usage:
  python leak_check.py --sim PRED_DIR TRUTH_DIR [--sim PRED_DIR TRUTH_DIR ...] [--models a,b]
  python leak_check.py --real VIEWER_ROOT [--models a,b]

Simulated trials (``--sim``: ``slap2-f0 predict`` output and the simulated
trial files it came from, with ``truth/baseline`` and ``activity``): the leak
is the slope of (baseline - true baseline) on the true activity at footprint-
support pixels, both as 8-frame means (the scale SiLo Z-scores): 0 = none,
1 = the baseline follows the activity. Comparing simulations whose activity
keeps the recordings' stimulus rhythm with the same simulations with
shuffled timing (``locnet_sims.py`` kind "simshuf") shows how much of the leak
comes from the rhythm.

Real trials (``--real``: the viewer's prediction folder; each session file
has movie, activity and baselines, model subfolders add baselines): the true
baseline is unknown, so each baseline's rise during the strongest population
responses (the 10% of frames with the largest summed activity, against the
quieter half; fast parts: 8-frame means minus their 4 s running mean) is
compared, in % of its mean, with the movie's own rise at background pixels
(a real shared brightening, e.g. neuropil or stimulus light, which a baseline
may follow) and with movie - activity at the support pixels. A baseline that
rises more than the shared brightening at support pixels follows the
synapses' own activity. Also the slope of the baseline's fast part on the
activity's fast part at support pixels.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d


def mean8(x):
    return uniform_filter1d(x, 8, axis=0, mode="nearest")


def fast(x):
    x = np.where(np.isfinite(x), x, np.nanmean(x, 0, keepdims=True))  # discarded frames
    x8 = mean8(x)
    return x8 - uniform_filter1d(x8, 320, axis=0, mode="nearest")


def slope(y, x):
    ok = np.isfinite(y) & np.isfinite(x)
    x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    return float((x * y).sum() / max((x * x).sum(), 1e-12))


def sim_leak(pred_fn, truth_fn, models):
    out = {}
    with h5py.File(pred_fn, "r") as p, h5py.File(truth_fn, "r") as t:
        rows, cols = p["rows"][()], p["cols"][()]
        sup = p["mask/footprint_support"][()][rows, cols].astype(bool)
        keep = p["frame/discard"][()] == 0
        truth = t["truth/baseline"][()][:, rows, cols][:, sup]
        act = t["activity"][0][:, rows, cols][:, sup]
        a8, tr8 = mean8(act)[keep], mean8(truth)[keep]
        for m in models or list(p["baseline"].keys()):
            if m in p["baseline"]:
                b = p[f"baseline/{m}"][()].astype(np.float32)[:, sup]
                out[m] = slope(mean8(b)[keep] - tr8, a8)
    return out


def rise(x, hi, lo):
    """Mean fast part over frames hi minus over frames lo, in % of the mean."""
    fb = np.nanmean(fast(x), 1)
    return 100 * (np.nanmean(fb[hi]) - np.nanmean(fb[lo])) / np.nanmean(x)


def real_leak(fn, pred_files, models):
    with h5py.File(fn, "r") as f:
        rows, cols = f["rows"][()], f["cols"][()]
        sup = f["mask/footprint_support"][()][rows, cols].astype(bool)
        bg = ~f["mask/labeled"][()][rows, cols].astype(bool)
        keep = f["frame/discard"][()] == 0
        act = f["activity"][()].astype(np.float32)
        mov = f["movie"][()].astype(np.float32)
    if not sup.any():
        return {}
    pop = fast(np.nansum(act[:, sup], 1)[:, None])[:, 0]
    pop[~keep] = np.nan
    hi, lo = pop >= np.nanquantile(pop, 0.9), pop <= np.nanquantile(pop, 0.5)
    out = {"movie, background: rise in strong responses (%)": rise(mov[:, bg], hi, lo),
           "movie, support (%)": rise(mov[:, sup], hi, lo),
           "movie - activity, support (%)": rise((mov - act)[:, sup], hi, lo)}
    a_s = fast(act[:, sup])[keep]
    for pf in pred_files:
        with h5py.File(pf, "r") as g:
            for m in g["baseline"].keys():
                if (models and m not in models) or f"{m}, support (%)" in out:
                    continue
                b = g[f"baseline/{m}"][()].astype(np.float32)
                out[f"{m}, support (%)"] = rise(b[:, sup], hi, lo)
                out[f"{m}, background (%)"] = rise(b[:, bg], hi, lo)
                out[f"{m}, support: slope on activity"] = slope(fast(b[:, sup])[keep], a_s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", nargs=2, action="append", default=[], metavar=("PRED_DIR", "TRUTH_DIR"))
    ap.add_argument("--real", default="")
    ap.add_argument("--models", default="")
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    if args.sim:
        rows = {}
        for pred, truth in args.sim:
            for fn in sorted(Path(pred).rglob("*.h5")):
                tf = Path(truth) / fn.relative_to(pred)
                if tf.exists():
                    rows[f"{Path(truth).name}/{fn.relative_to(pred)}"] = sim_leak(fn, tf, models)
        allm = sorted({m for r in rows.values() for m in r})
        print(f"{'simulated trial':58s} " + " ".join(f"{m:>15s}" for m in allm))
        for k, r in rows.items():
            print(f"{k:58s} " + " ".join(f"{r.get(m, np.nan):15.3f}" for m in allm))
    if args.real:
        root = Path(args.real)
        subs = [p for p in root.iterdir() if p.is_dir() and not p.name[:6].isdigit()]
        for sess in sorted(p for p in root.iterdir() if p.is_dir() and p.name[:6].isdigit()):
            for fn in sorted(sess.rglob("*.h5")):
                rel = fn.relative_to(root)
                preds = [fn] + [s / rel for s in subs if (s / rel).exists()]
                print(f"{rel}:")
                for k, v in real_leak(fn, preds, models).items():
                    print(f"   {k:58s} {v:8.3f}")


if __name__ == "__main__":
    main()
