"""Command line: ``slap2-f0 inspect <folder>`` and ``slap2-f0 train <folder> ...``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import WindowConfig, WindowDataset, loss_region
from .io import TrialFile, list_trials
from .losses import beta_nll, student_t_nll
from .model import ModelConfig, SlabBlindSpotNet, make_inputs


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


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    files = list_trials(args.root)
    if not files:
        raise SystemExit(f"no trial files under {args.root}")
    mcfg = ModelConfig(gap=args.gap, width=args.width, spatial_kernel=args.spatial_kernel)
    wcfg = WindowConfig(out_frames=args.out_frames, out_px=args.out_px,
                        t_reach=mcfg.gap + 1 + mcfg.temporal_reach, s_reach=mcfg.spatial_reach)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SlabBlindSpotNet(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ds = WindowDataset(files, wcfg, length=args.steps * args.batch, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, persistent_workers=args.workers > 0)
    region = loss_region(wcfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"model": vars(mcfg), "window": vars(wcfg), "args": vars(args)}, indent=2, default=str))
    t0 = time.time()
    for step, b in enumerate(dl, 1):
        b = {k: v.to(device) for k, v in b.items()}
        x = make_inputs(b["movie"], b["mean_image"], b["varfac"], b["scale"])
        mu, var = model(x)
        sel = (slice(None),) + region
        y, w = b["target"][sel], b["weight"][sel]
        if args.loss == "student":
            loss = student_t_nll(y, mu[sel], var[sel], w, nu=args.nu)
        else:
            loss = beta_nll(y, mu[sel], var[sel], w, beta=args.beta)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            print(f"step {step} loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "config": vars(mcfg), "step": step}, out / "model.pt")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="slap2-f0")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inspect", help="summarize exported trial files and their QC")
    pi.add_argument("root")
    pi.set_defaults(func=inspect)
    pt = sub.add_parser("train", help="train the slab blind-spot network")
    pt.add_argument("root", help="folder with exported trial files")
    pt.add_argument("--out", default="results/train")
    pt.add_argument("--gap", type=int, default=8)
    pt.add_argument("--width", type=int, default=32)
    pt.add_argument("--spatial_kernel", type=int, default=3)
    pt.add_argument("--out_frames", type=int, default=64)
    pt.add_argument("--out_px", type=int, default=32)
    pt.add_argument("--batch", type=int, default=4)
    pt.add_argument("--steps", type=int, default=20000)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--loss", choices=["beta", "student"], default="student")
    pt.add_argument("--beta", type=float, default=0.5)
    pt.add_argument("--nu", type=float, default=5.0)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--device", default="auto")
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--log_every", type=int, default=50)
    pt.add_argument("--save_every", type=int, default=1000)
    pt.set_defaults(func=train)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
