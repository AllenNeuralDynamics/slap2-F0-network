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
baseline is unknown, so the leak is the slope of the baseline's fast part
(8-frame mean minus its 4 s running mean) on the activity's fast part at
support pixels. For reference, the slope of the background pixels' fast part
on the summed activity, for the movie itself (stimulus-locked brightness, such
as stimulus light, that a correct baseline should follow) and for the
baseline (x1000).
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d


def mean8(x):
    return uniform_filter1d(x, 8, axis=0, mode="nearest")


def fast(x):
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


def real_leak(fn, pred_files, models):
    out = {}
    with h5py.File(fn, "r") as f:
        rows, cols = f["rows"][()], f["cols"][()]
        sup = f["mask/footprint_support"][()][rows, cols].astype(bool)
        bg = ~f["mask/labeled"][()][rows, cols].astype(bool)
        keep = f["frame/discard"][()] == 0
        act = f["activity"][()].astype(np.float32)
        mov = f["movie"][()].astype(np.float32)
    a_s = fast(act[:, sup])[keep]
    pop = np.repeat(fast(np.nansum(act[:, sup], 1, keepdims=True))[keep], bg.sum(), 1)
    out["movie: background on summed activity (x1e3)"] = 1e3 * slope(fast(mov[:, bg])[keep], pop)
    for pf in pred_files:
        with h5py.File(pf, "r") as g:
            for m in g["baseline"].keys():
                if models and m not in models or m in out:
                    continue
                b = g[f"baseline/{m}"][()].astype(np.float32)
                out[m] = slope(fast(b[:, sup])[keep], a_s)
                out[m + ": background on summed activity (x1e3)"] = 1e3 * slope(fast(b[:, bg])[keep], pop)
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
