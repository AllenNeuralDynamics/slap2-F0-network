"""Command line: ``slap2-f0 inspect <folder>`` and ``slap2-f0 train <folder> ...``."""

from __future__ import annotations

import argparse

import numpy as np

from . import ceiling as ceiling_mod
from . import train as train_mod
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
    pp = sub.add_parser("probe", help="peak GPU memory and time per step over a grid of model settings")
    train_mod.add_probe_arguments(pp)
    pp.set_defaults(func=train_mod.probe)
    pc = sub.add_parser("ceiling", help="reference baselines: how much there is for the network to learn")
    ceiling_mod.add_arguments(pc)
    pc.set_defaults(func=ceiling_mod.run)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
