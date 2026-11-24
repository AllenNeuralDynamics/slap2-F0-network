"""Command line: ``slap2-f0 inspect|train|evaluate|predict|predict-registered|view|probe|bench|ceiling|simcheck|stats ...``."""

from __future__ import annotations

import argparse

import numpy as np

from . import average as average_mod
from . import ceiling as ceiling_mod
from . import gpu_bench as bench_mod
from . import predict as predict_mod
from . import registered as registered_mod
from . import simulate as simulate_mod
from . import stats as stats_mod
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
                f"{tr.valid.sum()} valid, {int(a.get('nSources', 0))} sources | gain applied {a.get('activityGain', np.nan):.2f} "
                f"peak slope {a.get('peakSlope', np.nan):.2f} disk slope {a.get('diskSlope', np.nan):.2f} "
                f"event residual {a.get('eventResidualFraction', np.nan):.2f} blur {a.get('blurSigma', np.nan):.2f} "
                f"ring-event ratio core {a.get('eventRatioCore', np.nan):.2f} 2 px {a.get('eventRatio2px', np.nan):.2f}"
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
    pa = sub.add_parser("average", help="weighted average of checkpoints: soups of kept steps, WiSE-FT interpolation")
    average_mod.add_arguments(pa)
    pa.set_defaults(func=average_mod.run)
    pr = sub.add_parser("predict", help="full-trial baselines of saved checkpoints and SiLo, for viewing")
    predict_mod.add_arguments(pr)
    pr.set_defaults(func=predict_mod.run)
    pg = sub.add_parser("predict-registered",
                        help="network baselines for GIAnT's registered 80 Hz movies (*_BASELINE-80Hz.h5 for SiLo)")
    registered_mod.add_arguments(pg)
    pg.set_defaults(func=registered_mod.run)
    pv = sub.add_parser("view", help="browser viewer of the training data (movie, GIAnT fit, target) or of "
                                      "predict output (baselines and residuals), frame by frame")
    view_mod.add_arguments(pv)
    pv.set_defaults(func=view_mod.serve)
    pp = sub.add_parser("probe", help="peak GPU memory and time per step over a grid of model settings")
    train_mod.add_probe_arguments(pp)
    pp.set_defaults(func=train_mod.probe)
    pb = sub.add_parser("bench", help="machine benchmark: a trained model's step, forward and data-loader rates")
    bench_mod.add_arguments(pb)
    pb.set_defaults(func=bench_mod.grid)
    pc = sub.add_parser("ceiling", help="reference baselines: how much there is for the network to learn")
    ceiling_mod.add_arguments(pc)
    pc.set_defaults(func=ceiling_mod.run)
    pq = sub.add_parser("stats", help="compute trial stats and store them next to the trial files (end of an export)")
    pq.add_argument("root")
    pq.add_argument("--workers", type=int, default=8)
    pq.add_argument("--out", default="", help="write new sidecars into this overlay folder (<session>/<path>/...) "
                                            "instead of next to the trials (e.g. read-only data assets)")
    pq.set_defaults(func=stats_mod.run)
    ps = sub.add_parser("simcheck", help="simulate trials from templates and compare their noise statistics")
    simulate_mod.add_check_arguments(ps)
    ps.set_defaults(func=simulate_mod.check)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
