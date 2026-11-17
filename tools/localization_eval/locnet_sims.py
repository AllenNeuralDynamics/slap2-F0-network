"""Simulated trials with known sources for the localization test of network baselines.

usage: python locnet_sims.py OUT_DIR TRIAL.h5 [TRIAL.h5 ...] [--kinds sim,fast] [--seeds 1] [--stats DIR]

Each exported trial is a template (slap2_f0.simulate): its activity, shifted in
time and scaled, is added to a drifting baseline with matched noise, so the
true sources are the template's GIAnT sources (``sources/peak_row``,
``peak_col``, copied into each simulated file) and the true baseline is
``truth/baseline``. Kinds: "sim" (the default simulator) and "fast" (with the
fast shared brightness changes v6 was trained on). Writes
OUT_DIR/<session>_<kind><seed>/Path1/trial_001.h5, so ``slap2-f0 predict``
names them by folder.
"""
import argparse
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from slap2_f0.io import TrialFile  # noqa: E402
from slap2_f0.simulate import SimConfig, Template, write_trial  # noqa: E402
from slap2_f0.stats import load_stats  # noqa: E402
from slap2_f0.train import FAST_SIM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("trials", nargs="+")
ap.add_argument("--kinds", default="sim,fast")
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--stats", default="")
args = ap.parse_args()
out = Path(args.out)
stats = load_stats(args.trials, args.stats or (out / "stats"))
for fn in args.trials:
    with TrialFile(fn) as tf, h5py.File(fn, "r") as src:
        if "sources" not in src or int(tf.attrs.get("nSources", 0) or 0) == 0:
            print(f"{fn}: no sources, skipped")
            continue
        session = str(tf.attrs.get("label", Path(fn).parent.parent.name))
        tp = Template(tf, stats[str(fn)])
        for kind in args.kinds.split(","):
            cfg = SimConfig(**FAST_SIM) if kind == "fast" else SimConfig()
            for seed in range(args.seeds):
                dest = out / f"{session}_{kind}{seed}" / "Path1" / "trial_001.h5"
                dest.parent.mkdir(parents=True, exist_ok=True)
                r = write_trial(tp, dest, cfg, seed=777_000 + seed)
                with h5py.File(dest, "a") as f:
                    for k in ("peak_row", "peak_col"):
                        f[f"sources/{k}"] = src[f"sources/{k}"][()]
                    f.attrs["label"] = f"{session}_{kind}{seed}"
                print(f"{dest}: amp {r['amp']:.2f}, {len(src['sources/peak_row'])} sources", flush=True)
