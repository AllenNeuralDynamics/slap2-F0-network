"""Weighted averages of saved checkpoints' weights.

Two uses (docs/plan.md, "Fine-tuning trade-off", 2026-11-21):
    soup     the uniform average of a run's kept steps (``train --keep_every``),
             stochastic weight averaging after the fact:
             ``slap2-f0 average soup.pt "run/finetune/steps/step_00[01]*.pt"``
    WiSE-FT  an interpolation between the pretrained and the fine-tuned model:
             ``slap2-f0 average wise.pt pre.pt=0.5 ft.pt=0.5``

The network has no normalization statistics or other buffers, so the averaged
weights are a complete model. All checkpoints must share the model
configuration; the window settings are the first checkpoint's.
"""

from __future__ import annotations

import glob
import math
from pathlib import Path

import torch


def average_states(states: list[dict], weights: list[float]) -> dict:
    """The weighted mean of state dicts (weights normalized to sum 1), each
    tensor in float64 and cast back to its dtype."""
    if not states or len(states) != len(weights):
        raise ValueError("need one weight per state dict")
    keys = list(states[0])
    if any(list(s) != keys for s in states[1:]):
        raise ValueError("state dicts hold different tensors")
    weights = [float(w) for w in weights]
    tot = sum(weights)
    if not all(math.isfinite(w) for w in weights) or not (math.isfinite(tot) and tot > 0):
        raise ValueError(f"weights must be finite and sum to a positive number, not {weights}")
    out = {}
    for k in keys:
        acc = sum(w / tot * s[k].double() for w, s in zip(weights, states))
        out[k] = acc.to(states[0][k].dtype)
    return out


def parse_specs(specs: list[str]) -> list[tuple[str, float]]:
    """path, path=weight, or a glob (each match with the glob's weight)."""
    out = []
    for spec in specs:
        path, w = spec, 1.0
        if "=" in spec:
            head, tail = spec.rsplit("=", 1)
            try:
                path, w = head, float(tail)
            except ValueError:
                pass
        if any(c in path for c in "*?["):
            matches = sorted(glob.glob(path))  # wildcards in folder names too (runs/S*/steps/step_*.pt)
            if not matches:
                raise SystemExit(f"no checkpoints match {path}")
            out += [(str(m), w) for m in matches]
        else:
            out.append((path, w))
    return out


def run(args) -> None:
    specs = parse_specs(args.ckpt)
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p, _ in specs]
    for (p, _), ck in zip(specs, cks):
        if args.key not in ck:
            raise SystemExit(f"{p} has no '{args.key}' weights")
        if ck["config"] != cks[0]["config"]:
            raise SystemExit(f"{p}: model configuration differs from {specs[0][0]}")
    model = average_states([ck[args.key] for ck in cks], [w for _, w in specs])
    tot = sum(w for _, w in specs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model, "config": cks[0]["config"], "window": cks[0]["window"], "step": None,
                "averaged": [{"path": p, "weight": w / tot, "step": ck.get("step")} for (p, w), ck in zip(specs, cks)]},
               out)
    print(f"{out}: " + ", ".join(f"{w / tot:.3g} x {p} (step {ck.get('step')})" for (p, w), ck in zip(specs, cks)))


def add_arguments(p) -> None:
    p.add_argument("out", help="the averaged checkpoint (.pt)")
    p.add_argument("ckpt", nargs="+", help="checkpoints: path, path=weight (default 1; normalized to sum 1) or a "
                                           "glob such as run/steps/step_*.pt")
    p.add_argument("--key", default="model",
                   help="weights to average: model, or model_raw for the raw weights of an --ema run")
