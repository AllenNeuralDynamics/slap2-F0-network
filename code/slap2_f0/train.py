"""Training loop for the blind-spot baseline network.

Files are split into training and validation by session when there are at
least three sessions (so validation measures generalization to unseen
sessions), otherwise by trial. Validation uses a fixed set of windows and
reports the likelihood plus calibration of the predicted noise: z = (target -
mu) / sqrt(var) should have mean 0 and variance 1 (about 95% within +-2 for a
Gaussian), which is what Z-scoring events in SiLo relies on.

Outputs in --out: config.json, metrics.csv (one row per validation),
last.pt and best.pt (lowest validation loss).
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import WindowConfig, WindowDataset, loss_region
from .io import TrialFile, list_trials
from .losses import beta_nll, student_t_nll
from .model import ModelConfig, SlabBlindSpotNet, make_inputs


def split_files(files: list[Path], val_fraction: float, seed: int = 0) -> tuple[list[Path], list[Path], str]:
    """Hold out whole sessions if there are at least 3, else whole trials."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        with TrialFile(f) as tr:
            groups[str(tr.attrs.get("label", f.parent.parent.name))].append(f)
    rng = np.random.default_rng(seed)
    if len(groups) >= 3:
        names = sorted(groups)
        rng.shuffle(names)
        n_val = max(1, round(val_fraction * len(names)))
        val = [f for n in names[:n_val] for f in groups[n]]
        train = [f for n in names[n_val:] for f in groups[n]]
        return train, val, f"by session (validation: {', '.join(sorted(names[:n_val]))})"
    files = sorted(files)
    idx = rng.permutation(len(files))
    n_val = max(1, round(val_fraction * len(files))) if len(files) > 1 else 0
    val = [files[i] for i in idx[:n_val]]
    train = [files[i] for i in idx[n_val:]] or val
    return train, val, f"by trial ({len(groups)} session(s))"


def _loss(args, y, mu, var, w):
    if args.loss == "student":
        return student_t_nll(y, mu, var, w, nu=args.nu)
    return beta_nll(y, mu, var, w, beta=args.beta)


def _forward(model, b, device, amp_dtype, region):
    """Model outputs on the loss region only (mu, var: (B, frames, rows, cols))."""
    b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
    x = make_inputs(b["movie"], b["mean_image"], b["varfac"], b["scale"])
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        mu, var = model(x, b["scale"], region)
    return b, mu.float(), var.float()


@torch.no_grad()
def evaluate(model, loader, region, device, amp_dtype, args) -> dict[str, float]:
    model.eval()
    sel = (slice(None),) + region
    tot = defaultdict(float)
    for b in loader:
        b, m, v = _forward(model, b, device, amp_dtype, region)
        y, w = b["target"][sel], b["weight"][sel]
        n = w.sum().item()
        if n == 0:
            continue
        tot["loss"] += _loss(args, y, m, v, w).item() * n
        z = (y - m) / v.sqrt()
        tot["z"] += (z * w).sum().item()
        tot["z2"] += (z ** 2 * w).sum().item()
        tot["in2"] += ((z.abs() < 2).float() * w).sum().item()
        tot["n"] += n
    model.train()
    n = max(tot["n"], 1.0)
    zm = tot["z"] / n
    return {"val_loss": tot["loss"] / n, "z_mean": zm, "z_var": tot["z2"] / n - zm ** 2, "within_2": tot["in2"] / n}


def train(args) -> None:
    torch.manual_seed(args.seed)
    files = list_trials(args.root)
    if not files:
        raise SystemExit(f"no trial files under {args.root}")
    train_files, val_files, how = split_files(files, args.val_fraction, args.seed)
    mcfg = ModelConfig(gap=args.gap, width=args.width, spatial_kernel=args.spatial_kernel,
                       downsample=args.downsample, full_res_blocks=args.full_res_blocks,
                       checkpoint=not args.no_checkpoint)
    if args.temporal_dilations:
        mcfg.temporal_dilations = [int(d) for d in args.temporal_dilations.split(",")]
    if args.spatial_dilations:
        mcfg.spatial_dilations = [int(d) for d in args.spatial_dilations.split(",")]
    wcfg = WindowConfig(out_frames=args.out_frames, out_px=args.out_px,
                        t_reach=mcfg.gap + 1 + mcfg.temporal_reach, s_reach=mcfg.spatial_reach,
                        w_unfitted=args.w_unfitted)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = None
    if device.type == "cuda" and not args.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = SlabBlindSpotNet(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    warm = min(500, max(1, args.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))

    kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")
    dl = DataLoader(WindowDataset(train_files, wcfg, length=args.steps * args.batch, seed=args.seed),
                    batch_size=args.batch, **kw)
    vl = DataLoader(WindowDataset(val_files, wcfg, length=args.val_windows, seed=10_000 + args.seed),
                    batch_size=args.batch, **kw) if val_files else None
    region = loss_region(wcfg)
    sel = (slice(None),) + region

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "model": asdict(mcfg), "window": asdict(wcfg), "args": vars(args), "n_params": n_params,
        "device": str(device), "amp": str(amp_dtype), "split": how,
        "train_files": [str(f) for f in train_files], "val_files": [str(f) for f in val_files],
    }, indent=2, default=str))
    print(f"{len(files)} trial files; split {how}: {len(train_files)} train, {len(val_files)} validation")
    print(f"model {n_params:,} parameters on {device} (amp {amp_dtype}); window {wcfg.out_frames + 2 * wcfg.t_reach} frames x "
          f"{wcfg.out_px + 2 * wcfg.s_reach}^2 px, loss on {wcfg.out_frames} x {wcfg.out_px}^2", flush=True)

    fields = ["step", "train_loss", "val_loss", "z_mean", "z_var", "within_2", "lr", "elapsed_s"]
    with open(out / "metrics.csv", "w", newline="") as fh:
        csv.writer(fh).writerow(fields)
    best = math.inf
    ema = None
    t0 = time.time()

    def save(name, step):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": asdict(mcfg),
                    "window": asdict(wcfg), "step": step}, out / name)

    step = 0
    for step, b in enumerate(dl, 1):
        b, mu, var = _forward(model, b, device, amp_dtype, region)
        loss = _loss(args, b["target"][sel], mu, var, b["weight"][sel])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % args.log_every == 0 or step == 1:
            mem = f" gpu peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if device.type == "cuda" else ""
            print(f"step {step} loss {loss.item():.4f} (avg {ema:.4f}) lr {sched.get_last_lr()[0]:.2e} "
                  f"{time.time() - t0:.0f} s{mem}", flush=True)
        last_step = step == args.steps or (args.max_hours and time.time() - t0 > 3600 * args.max_hours)
        if vl is not None and (step % args.val_every == 0 or last_step):
            m = evaluate(model, vl, region, device, amp_dtype, args)
            row = {"step": step, "train_loss": ema, **m, "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0}
            with open(out / "metrics.csv", "a", newline="") as fh:
                csv.writer(fh).writerow([row[k] for k in fields])
            print("validation " + " ".join(f"{k} {v:.4f}" for k, v in m.items()), flush=True)
            if m["val_loss"] < best:
                best = m["val_loss"]
                save("best.pt", step)
        if step % args.save_every == 0 or last_step:
            save("last.pt", step)
        if last_step:
            break
    print(f"done after {step} steps, {time.time() - t0:.0f} s; best validation loss {best:.4f}")


def probe(args) -> None:
    """Peak GPU memory and time per training step over a grid of settings, on
    random inputs of the real window size; writes probe.csv to --out."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    grid = [(b, w, fr, md) for md in [int(v) for v in args.max_dilations.split(",")]
            for fr in [int(v) for v in args.full_res_list.split(",")]
            for w in [int(v) for v in args.widths.split(",")]
            for b in [int(v) for v in args.batches.split(",")]]
    for b, w, fr, md in grid:
        dil = [2 ** k for k in range(int(math.log2(md)) + 1)]
        mcfg = ModelConfig(width=w, full_res_blocks=fr, temporal_dilations=dil)
        T = args.out_frames + 2 * (mcfg.gap + 1 + mcfg.temporal_reach)
        S = args.out_px + 2 * mcfg.spatial_reach
        region = (slice(T // 2 - args.out_frames // 2, T // 2 + args.out_frames // 2),
                  slice(mcfg.spatial_reach, mcfg.spatial_reach + args.out_px),
                  slice(mcfg.spatial_reach, mcfg.spatial_reach + args.out_px))
        row = {"batch": b, "width": w, "full_res_blocks": fr, "max_dilation": md, "window_frames": T,
               "window_px": S, "peak_gb": float("nan"), "s_per_step": float("nan"), "status": "ok"}
        model = opt = None
        try:
            model = SlabBlindSpotNet(mcfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            movie = torch.rand(b, T, S, S, device=device) * 20
            mean = torch.full((b, S, S), 10.0, device=device)
            vf = torch.ones(b, S, S, device=device)
            scale = torch.full((b,), 10.0, device=device)
            y = torch.rand(b, args.out_frames, args.out_px, args.out_px, device=device)
            w8 = torch.ones_like(y)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(3):
                t1 = time.time()
                x = make_inputs(movie, mean, vf, scale)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    mu, var = model(x, scale, region)
                loss = student_t_nll(y, mu.float(), var.float(), w8)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - t1)
            row["s_per_step"] = float(np.mean(times[1:]))
            if device.type == "cuda":
                row["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        except torch.OutOfMemoryError:
            row["status"] = "out of memory"
        finally:
            del model, opt
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)
        print(" ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    with open(out / "probe.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def add_probe_arguments(p) -> None:
    p.add_argument("--out", default="results/probe")
    p.add_argument("--batches", default="1,2,4,8")
    p.add_argument("--widths", default="16,32")
    p.add_argument("--full_res_list", default="0,1,2")
    p.add_argument("--max_dilations", default="32,64")
    p.add_argument("--out_frames", type=int, default=64)
    p.add_argument("--out_px", type=int, default=32)


def add_arguments(p) -> None:
    p.add_argument("root", help="folder with exported trial files (searched recursively)")
    p.add_argument("--out", default="results/train")
    p.add_argument("--gap", type=int, default=8, help="frames hidden on each side of the output frame")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--spatial_kernel", type=int, default=3, help="1 = each pixel's own trace only (v0)")
    p.add_argument("--temporal_dilations", default="", help="comma-separated, default 1,2,4,...,64")
    p.add_argument("--spatial_dilations", default="", help="comma-separated, default 1,1,2,2,4,4,8")
    p.add_argument("--downsample", type=int, default=2, help="spatial downsampling after the full-resolution blocks")
    p.add_argument("--full_res_blocks", type=int, default=2)
    p.add_argument("--no_checkpoint", action="store_true", help="store activations instead of recomputing them")
    p.add_argument("--out_frames", type=int, default=64)
    p.add_argument("--out_px", type=int, default=32)
    p.add_argument("--w_unfitted", type=float, default=0.0, help="loss weight of labeled pixels outside every footprint")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--max_hours", type=float, default=0.0, help="stop (and validate, save) after this long; 0 = no limit")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--loss", choices=["beta", "student"], default="student")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--nu", type=float, default=5.0)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--val_windows", type=int, default=256)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--no_amp", action="store_true", help="disable mixed precision on the GPU")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1000)
