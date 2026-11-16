"""Command line: ``slap2-f0 inspect|train|evaluate|predict|view|probe|ceiling|simcheck ...``."""

from __future__ import annotations

import argparse

import numpy as np

from . import ceiling as ceiling_mod
from . import predict as predict_mod
from . import simulate as simulate_mod
from . import train as train_mod
from . import view as view_mod
from .io import TrialFile, list_trials


def inspect(args: argparse.Namespace) -> None:
    files = list_trials(args.root)
    print(f"{len(files)} trial files under {args.root}")
    for fn in files:
        with TrialFile(fn) as tr:
            a = tr.attrs
            print(
                f"{fn.relative_to(args.root)}: {tr.n_frames} frames, {tr.n_rows}x{tr.n_cols} px, "
                f"{tr.valid.sum()} valid, {int(a.get('nSources', 0))} sources | gain {a.get('activityGain', np.nan):.2f} "
                f"peak slope {a.get('peakSlope', np.nan):.2f} disk slope {a.get('diskSlope', np.nan):.2f} "
                f"event residual {a.get('eventResidualFraction', np.nan):.2f} blur {a.get('blurSigma', np.nan):.2f}"
            )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="slap2-f0")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inspect", help="summarize exported trial files and their QC")
    pi.add_argument("root")
    pi.set_defaults(func=inspect)
    pt = sub.add_parser("train", help="train the slab blind-spot network")
    train_mod.add_arguments(pt)
    pt.set_defaults(func=train_mod.train)
    pe = sub.add_parser("evaluate", help="score saved checkpoints on each session's windows (real and simulated)")
    train_mod.add_eval_arguments(pe)
    pe.set_defaults(func=train_mod.evaluate_checkpoints)
    pr = sub.add_parser("predict", help="full-trial baselines of saved checkpoints and SiLo, for viewing")
    predict_mod.add_arguments(pr)
    pr.set_defaults(func=predict_mod.run)
    pv = sub.add_parser("view", help="browser viewer of predict output: data, baselines and residuals over time")
    pv.add_argument("root", help="folder with predict output (any depth)")
    pv.add_argument("--port", type=int, default=8765)
    pv.add_argument("--host", default="127.0.0.1")
    pv.add_argument("--no_browser", action="store_true", help="do not open a browser tab")
    pv.add_argument("--cache_trials", type=int, default=2, help="trials kept in memory")
    pv.add_argument("--models", default="", help="comma-separated models to list first, in this order (best first)")
    pv.set_defaults(func=view_mod.serve)
    pp = sub.add_parser("probe", help="peak GPU memory and time per step over a grid of model settings")
    train_mod.add_probe_arguments(pp)
    pp.set_defaults(func=train_mod.probe)
    pc = sub.add_parser("ceiling", help="reference baselines: how much there is for the network to learn")
    ceiling_mod.add_arguments(pc)
    pc.set_defaults(func=ceiling_mod.run)
    ps = sub.add_parser("simcheck", help="simulate trials from templates and compare their noise statistics")
    simulate_mod.add_check_arguments(ps)
    ps.set_defaults(func=simulate_mod.check)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
