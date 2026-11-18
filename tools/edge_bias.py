"""Baseline bias near the start and end of trials.

usage:
  python edge_bias.py --sim PRED_DIR TRUTH_DIR [--sim ...] [--models a,b]
  python edge_bias.py --registered MOCO_DIR BASELINE_DIR [--registered ...] [--files DMD2]

Simulated trials (``slap2-f0 predict`` output and the simulated trials, with
``truth/baseline``): mean (baseline - truth) / mean truth, in %, over valid
pixels (support and background separately), in bins of frames counted from
the trial's first and from its last frame, next to the trial's middle.

Real trials (``slap2-f0 predict-registered`` baselines and GIAnT's registered
movies): the true baseline is unknown, so the residual movie - baseline, in %
of the mean baseline, averaged over all valid pixels and trials, per bin;
away from the edges it is the mean activity (plus any bias), so a bias near
the edges shows as a step against the middle. Shared brightness changes
(2-5% per frame) average out over trials.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

BINS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256)]


def bin_means(x, T):
    """x (T,) per-frame values -> dict of bin -> mean, from the start ('s'), the end ('e') and the middle."""
    out = {}
    for a, b in BINS:
        out[f"s{a}-{b}"] = np.nanmean(x[a:b])
        out[f"e{a}-{b}"] = np.nanmean(x[T - b:T - a])
    out["mid"] = np.nanmean(x[256:T - 256])
    return out


def sim_rows(pred_fn, truth_fn, models):
    rows = {}
    with h5py.File(pred_fn, "r") as p, h5py.File(truth_fn, "r") as t:
        rr, cc = p["rows"][()], p["cols"][()]
        sup = p["mask/footprint_support"][()][rr, cc].astype(bool)
        bg = ~p["mask/labeled"][()][rr, cc].astype(bool)
        keep = p["frame/discard"][()] == 0
        truth = t["truth/baseline"][()][:, rr, cc]
        T = truth.shape[0]
        for m in models or list(p["baseline"].keys()):
            if m not in p["baseline"]:
                continue
            b = p[f"baseline/{m}"][()].astype(np.float32)
            for cname, cm in (("sup", sup), ("bg", bg)):
                d = np.nanmean(b[:, cm] - truth[:, cm], 1) / np.nanmean(truth[:, cm])
                d[~keep] = np.nan
                rows[(m, cname)] = bin_means(100 * d, T)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", nargs=2, action="append", default=[], metavar=("PRED_DIR", "TRUTH_DIR"))
    ap.add_argument("--registered", nargs=2, action="append", default=[], metavar=("MOCO_DIR", "BASELINE_DIR"))
    ap.add_argument("--files", default="")
    ap.add_argument("--models", default="")
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    cols = [f"s{a}-{b}" for a, b in BINS] + ["mid"] + [f"e{a}-{b}" for a, b in reversed(BINS)]
    if args.sim:
        acc = {}
        for pred, truth in args.sim:
            for fn in sorted(Path(pred).rglob("*.h5")):
                tf = Path(truth) / fn.relative_to(pred)
                if tf.exists():
                    for k, v in sim_rows(fn, tf, models).items():
                        acc.setdefault(k, []).append(v)
        print("bias (baseline - truth) / truth in %, mean over simulated trials; frames from the start (s) and end (e)")
        print(f"{'model':16s} {'px':4s} " + " ".join(f"{c:>8s}" for c in cols))
        for (m, cname), vs in sorted(acc.items()):
            print(f"{m:16s} {cname:4s} " + " ".join(f"{np.nanmean([v[c] for v in vs]):8.2f}" for c in cols))
    if args.registered:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
        from slap2_f0.registered import load_registered
        acc = []
        for moco, bdir in args.registered:
            for bf in sorted(Path(bdir).glob("*_BASELINE-80Hz.h5")):
                stem = bf.name.replace("_BASELINE-80Hz.h5", "")
                if args.files and args.files not in stem:
                    continue
                movie, valid, discard, _, _ = load_registered(Path(moco) / f"{stem}_REGISTERED_DOWNSAMPLED-80Hz.tif",
                                                              Path(moco) / f"{stem}_ALIGNMENTDATA.h5")
                with h5py.File(bf, "r") as f:
                    b = f["baseline"][:].astype(np.float32).transpose(0, 2, 1)
                m, bb = movie[:, valid], b[:, valid]
                res = np.nanmean(m - bb, 1) / np.nanmean(bb)
                res[discard] = np.nan
                acc.append(bin_means(100 * res, len(res)))
                print(f"  {stem}: middle residual {acc[-1]['mid']:.2f}%", flush=True)
        print(f"residual (movie - baseline) / baseline in %, mean over {len(acc)} trials; minus the middle:")
        print(" ".join(f"{c:>8s}" for c in cols))
        print(" ".join(f"{np.nanmean([v[c] - v['mid'] for v in acc]):8.2f}" for c in cols))
        print("SD over trials of (bin - middle):")
        print(" ".join(f"{np.nanstd([v[c] - v['mid'] for v in acc]):8.2f}" for c in cols))


if __name__ == "__main__":
    main()
