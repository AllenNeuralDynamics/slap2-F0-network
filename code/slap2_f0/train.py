"""Training loop for the blind-spot baseline network.

Two kinds of data (``--data``):
    sim   simulated windows built on real trials as templates, with the true,
          noise-free baseline as the target (``simulate``): pretraining
    real  real windows, with movie - activity as the (noisy) target:
          fine-tuning, usually from a simulation-pretrained checkpoint
          (``--init``)

Loss (``_losses``): squared error of the baseline weighted by 1 / photon-
noise variance, which is minimized by the mean (unbiased at any brightness),
plus ``w_var`` times a Gaussian NLL for the noise head around the detached
baseline.

Files are split into training and validation by session when there are at
least three sessions (so validation measures generalization to unseen
sessions), otherwise by trial. Every validation evaluates fixed windows from
the held-out files, both real (``real_*``) and simulated (``sim_*``), and
from the training files (``tr_*``); see ``evaluate`` for the metrics.

best.pt is chosen on held-out windows of the training data kind (``--select``,
default ``balanced``): the network's 8-frame MSE relative to the mean image's,
averaged over footprint-support and background pixels. The per-frame loss
can't pick checkpoints on real data: photon noise buries the 1-3% that can be
learned (v2 run). The untrained network (step 0, which predicts the mean
image, score 1) is a candidate, so best.pt is never worse than the start.

Outputs in --out: config.json, metrics.csv (one row per validation),
last.pt, best.pt and stats/ (cached whole-trial statistics).
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import WindowConfig, WindowDataset, loss_region
from .io import TrialFile, list_trials
from .losses import gaussian_nll, student_t_nll
from .model import ModelConfig, SlabBlindSpotNet, blind_window_stats, make_inputs
from .simulate import SimConfig
from .stats import load_stats


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


def _ref_var(b, region):
    """Photon-noise variance at the mean image (scale units), a fixed per-pixel
    weight for the mean loss that does not depend on the model."""
    s = b["scale"].view(-1, 1, 1)
    mi = (b["mean_image"] / s)[(slice(None),) + region[1:]].clamp_min(1e-3)
    gs = (b["gain"] / b["scale"]).view(-1, 1, 1)
    return torch.nan_to_num(gs * mi + 1.0 / s ** 2, nan=1.0).unsqueeze(1)


def _losses(args, b, mu, var, region):
    """Mean loss: squared error against the target weighted by 1 / photon-noise
    variance, which is minimized by the conditional mean, so it is unbiased at
    any brightness (unlike a robust likelihood on skewed low counts). Noise
    loss: Gaussian NLL of movie - activity around the detached mean, so the
    variance head learns the actual spread around the predicted baseline."""
    sel = (slice(None),) + region
    w = b["weight"][sel]
    y, yn = b["target"][sel], b["noisy"][sel]
    mu_loss = (w * (y - mu) ** 2 / _ref_var(b, region)).sum() / w.sum().clamp_min(1.0)
    var_loss = gaussian_nll(yn, mu.detach(), var, w)
    return args.w_mu * mu_loss + args.w_var * var_loss, mu_loss, var_loss


def _forward(model, b, device, amp_dtype, region, aux=False):
    """Model outputs on the loss region only (mu, var: (B, frames, rows, cols))."""
    b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
    x = make_inputs(b["movie"], b["mean_image"], b["varfac"], b["scale"], b["var_map"], b["gain"])
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        out = model(x, b["scale"], b["gain"], region, aux=aux)
    if aux:
        return b, out[0].float(), out[1].float(), out[2]
    return b, out[0].float(), out[1].float()


def movmean_t(x: torch.Tensor, w: torch.Tensor, k: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted moving mean over k frames along dim 1 (SiLo's IMs for k = 8);
    returns the mean and the summed weight."""
    T = x.shape[1]
    cs = F.pad((x * w).cumsum(1), (0, 0, 0, 0, 1, 0))
    cw = F.pad(w.cumsum(1), (0, 0, 0, 0, 1, 0))
    lo, hi = k // 2, k - k // 2
    t = torch.arange(T, device=x.device)
    a, b = (t - lo).clamp(0, T), (t + hi).clamp(0, T)
    n = cw[:, b] - cw[:, a]
    return (cs[:, b] - cs[:, a]) / n.clamp_min(1e-9), n


def silo_baseline(movie: torch.Tensor, scale: torch.Tensor, region, smooth: int = 8, window: int = 321) -> torch.Tensor:
    """Today's SiLo localization baseline on the loss region: a moving mean over
    `smooth` frames, then a moving median over `window` frames
    (localizeSources_vIM at 80 Hz: ceil(0.2 s * 80 / 2) and 4 s * 80 + 1),
    NaN-aware, in units of the trial scale. It is not blind: each frame's
    baseline includes that frame and its events."""
    tsl, rsl, csl = region
    half = window // 2
    m = movie[:, :, rsl, csl] / scale.view(-1, 1, 1, 1)
    t0, t1 = tsl.start - half - smooth, tsl.stop + half + smooth
    pad_l, pad_r = max(0, -t0), max(0, t1 - m.shape[1])
    m = F.pad(m.permute(0, 2, 3, 1), (pad_l, pad_r), value=float("nan")).permute(0, 3, 1, 2)
    m = m[:, t0 + pad_l: t1 + pad_l]
    valid = torch.isfinite(m)
    cs = F.pad(torch.where(valid, m, torch.zeros_like(m)).cumsum(1), (0, 0, 0, 0, 1, 0))
    cn = F.pad(valid.float().cumsum(1), (0, 0, 0, 0, 1, 0))
    lo, hi = smooth // 2, smooth - smooth // 2
    idx = torch.arange(lo, m.shape[1] - hi, device=m.device)
    mm = (cs[:, idx + hi] - cs[:, idx - lo]) / (cn[:, idx + hi] - cn[:, idx - lo])  # NaN where no data
    n_out = tsl.stop - tsl.start
    first = half + smooth - lo  # index in mm of the first output frame
    win = mm[:, first - half: first - half + n_out + window - 1].unfold(1, window, 1)  # (B, n_out, H, W, window)
    return torch.nanmedian(win, dim=-1).values


@torch.no_grad()
def evaluate(model, loader, region, device, amp_dtype, args) -> dict[str, float]:
    """Mean loss, noise calibration and baseline errors on fixed windows.

    Baselines compared against the target (the true baseline for simulated
    windows, movie - activity for real ones): the network, the mean image (its
    starting point), the blind window mean (its other built-in reference) and
    today's SiLo baseline (moving median). Two oracles that know the activity,
    as in ``ceiling``: the mean image minus the trial's mean activity (oracle,
    the best constant) and the blind window mean of movie - activity
    (oracle_blind, which also follows drift but is noisier). Errors per frame
    (mse_*) and on
    8-frame averages (mse8_*, what SiLo Z-scores), on all loss pixels and
    split into footprint support and unlabeled background. For real targets
    the photon noise adds the same amount to every MSE; for simulated ones the
    MSE is the baseline error itself. bias_* = mean(target - baseline).
    Calibration: z = (movie - activity - mu) / sqrt(var) should have mean 0 and
    variance 1; r8 is the variance of 8-frame averages of the residual over
    the mean predicted per-frame variance (the factor that converts the
    network's sigma to SiLo's)."""
    model.eval()
    sel = (slice(None),) + region
    tot = defaultdict(float)
    for b in loader:
        b, m, v, aux = _forward(model, b, device, amp_dtype, region, aux=True)
        y, yn, w = b["target"][sel], b["noisy"][sel], b["weight"][sel]
        s = b["scale"].view(-1, 1, 1)
        mi = (b["mean_image"] / s)[(slice(None),) + region[1:]].unsqueeze(1).expand_as(y)
        silo = silo_baseline(b["movie"], b["scale"], region)
        w = w * torch.isfinite(silo) * torch.isfinite(mi)
        if w.sum() == 0:
            continue
        oracle = torch.nan_to_num(((b["mean_image"] - b["mean_act"]) / s)[(slice(None),) + region[1:]])
        oracle = oracle.unsqueeze(1).expand_as(y)
        wv = (b["weight"] > 0).float()
        ob, _, ob_n = blind_window_stats(b["noisy"] * wv, wv, model.cfg.gap, model.cfg.window_reach)
        ob = torch.where(ob_n[sel] > 0, ob[sel], oracle)
        preds = {"net": m, "mean_image": torch.nan_to_num(mi), "window": aux["window_mean"],
                 "silo": torch.nan_to_num(silo), "oracle": oracle, "oracle_blind": ob}
        sup = b["support"][(slice(None),) + region[1:]].bool().unsqueeze(1)
        bg = ~b["labeled"][(slice(None),) + region[1:]].bool().unsqueeze(1)
        classes = {"": w, "sup_": w * sup, "bg_": w * bg}
        y8, n8 = movmean_t(y, w)
        w8 = w * (n8 >= 4)
        for cname, cw in classes.items():
            n = cw.sum().item()
            if n == 0:
                continue
            tot[cname + "n"] += n
            cw8 = cw * w8
            tot[cname + "n8"] += cw8.sum().item()
            for pname, p in preds.items():
                d = y - p
                tot[f"{cname}mse_{pname}"] += (cw * d ** 2).sum().item()
                p8, _ = movmean_t(p, w)
                tot[f"{cname}mse8_{pname}"] += (cw8 * (y8 - p8) ** 2).sum().item()
                if cname == "" or pname == "net":
                    tot[f"{cname}bias_{pname}"] += (cw * d).sum().item()
        n = w.sum().item()
        tot["loss"] += (w * (y - m) ** 2 / _ref_var(b, region)).sum().item()
        tot["var_loss"] += gaussian_nll(yn, m, v, w).item() * n
        z = (yn - m) / v.sqrt()
        tot["z"] += (z * w).sum().item()
        tot["z2"] += (z ** 2 * w).sum().item()
        tot["in2"] += ((z.abs() < 2).float() * w).sum().item()
        r8, _ = movmean_t(yn - m, w)
        tot["r8_num"] += (w8 * r8 ** 2).sum().item()
        tot["r8_den"] += (w8 * v).sum().item()
        tot["mix"] += (w * aux["mix"]).sum().item()
    model.train()
    n = max(tot["n"], 1.0)
    out = {"loss": tot["loss"] / n, "var_loss": tot["var_loss"] / n}
    zm = tot["z"] / n
    out.update(z_mean=zm, z_var=tot["z2"] / n - zm ** 2, within_2=tot["in2"] / n,
               r8=tot["r8_num"] / max(tot["r8_den"], 1e-9), mix=tot["mix"] / n)
    for k, v in tot.items():
        if "_mse" in k or k.startswith("mse") or "bias_" in k:
            cname = k.split("mse")[0] if "mse" in k else k.split("bias_")[0]
            denom = tot.get(cname + ("n8" if "mse8_" in k else "n"), 0.0)
            out[k] = v / max(denom, 1.0)
    return out


def selection_score(row: dict, prefix: str, how: str) -> float:
    """Checkpoint score (lower is better) from a validation row: the loss, the
    8-frame MSE relative to the mean image's, or that ratio averaged over
    footprint-support and background pixels (balanced; the overall ratio
    when neither class has pixels)."""
    if how == "loss":
        return row.get(f"{prefix}_loss", math.inf)

    def ratio(c):
        a, b = row.get(f"{prefix}_{c}mse8_net", math.nan), row.get(f"{prefix}_{c}mse8_mean_image", math.nan)
        return a / b if b > 0 and math.isfinite(a) else math.nan

    if how == "balanced":
        r = [x for x in (ratio("sup_"), ratio("bg_")) if math.isfinite(x)]
        if r:
            return float(np.mean(r))
    r = ratio("")
    return r if math.isfinite(r) else math.inf


def _model_config(args) -> ModelConfig:
    mcfg = ModelConfig(gap=args.gap, width=args.width, spatial_kernel=args.spatial_kernel,
                       downsample=args.downsample, full_res_blocks=args.full_res_blocks,
                       checkpoint=not args.no_checkpoint, window_reach=args.window_reach,
                       max_log_correction=args.max_log_correction)
    if args.temporal_dilations:
        mcfg.temporal_dilations = [int(d) for d in args.temporal_dilations.split(",")]
    if args.spatial_dilations:
        mcfg.spatial_dilations = [int(d) for d in args.spatial_dilations.split(",")]
    return mcfg


def train(args) -> None:
    torch.manual_seed(args.seed)
    files = list_trials(args.root)
    if not files:
        raise SystemExit(f"no trial files under {args.root}")
    train_files, val_files, how = split_files(files, args.val_fraction, args.seed)
    mcfg = _model_config(args)
    if mcfg.gap + mcfg.window_reach > mcfg.gap + 1 + mcfg.temporal_reach:
        raise SystemExit("window_reach must not exceed the temporal reach + 1 (the window would run out)")
    wcfg = WindowConfig(out_frames=args.out_frames, out_px=args.out_px,
                        t_reach=mcfg.gap + 1 + mcfg.temporal_reach, s_reach=mcfg.spatial_reach,
                        w_unfitted=args.w_unfitted)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = None
    if device.type == "cuda" and not args.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stats = load_stats(files, args.stats_cache or out / "stats", workers=args.workers)
    print(f"trial stats for {len(files)} files in {time.time() - t0:.0f} s; gain "
          + " ".join(f"{stats[str(f)]['gain']:.1f}" for f in files[:8]) + (" ..." if len(files) > 8 else ""), flush=True)

    model = SlabBlindSpotNet(mcfg).to(device)
    if args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"initialized from {args.init} (step {ck.get('step')})")
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    warm = min(1000, max(1, args.steps // 5))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))

    sim = SimConfig() if args.data == "sim" else None
    kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")
    mk = lambda fl, n, seed, s: DataLoader(WindowDataset(fl, wcfg, stats, length=n, seed=seed, sim=s,  # noqa: E731
                                                         flip=seed == args.seed and not args.no_flip),
                                           batch_size=args.batch, **kw)
    dl = mk(train_files, args.steps * args.batch, args.seed, sim)
    evals = {}
    if val_files:
        evals["real"] = mk(val_files, args.val_windows, 10_000 + args.seed, None)
        evals["sim"] = mk(val_files, args.val_windows, 20_000 + args.seed, SimConfig())
    evals["tr"] = mk(train_files, args.val_windows, 30_000 + args.seed, sim)  # did it learn anything at all?
    region = loss_region(wcfg)
    sel_prefix = args.data if val_files else "tr"
    sel_key = f"{sel_prefix}_{args.select}"

    (out / "config.json").write_text(json.dumps({
        "model": asdict(mcfg), "window": asdict(wcfg), "args": vars(args), "n_params": n_params,
        "device": str(device), "amp": str(amp_dtype), "split": how, "sim": asdict(SimConfig()),
        "train_files": [str(f) for f in train_files], "val_files": [str(f) for f in val_files],
    }, indent=2, default=str))
    print(f"{len(files)} trial files; split {how}: {len(train_files)} train, {len(val_files)} validation; "
          f"training on {args.data} data, selecting on {sel_key}")
    print(f"model {n_params:,} parameters on {device} (amp {amp_dtype}); window {wcfg.out_frames + 2 * wcfg.t_reach} frames x "
          f"{wcfg.out_px + 2 * wcfg.s_reach}^2 px, loss on {wcfg.out_frames} x {wcfg.out_px}^2", flush=True)

    fields = None
    best = math.inf
    ema = None

    def save(name, step):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": asdict(mcfg),
                    "window": asdict(wcfg), "step": step}, out / name)

    def validate(step):
        nonlocal fields, best
        row = {"step": step, "train_loss": ema if ema is not None else float("nan")}
        for name, loader in evals.items():
            for k, v in evaluate(model, loader, region, device, amp_dtype, args).items():
                row[f"{name}_{k}"] = v
        row[sel_key] = selection_score(row, sel_prefix, args.select)
        row.update(lr=sched.get_last_lr()[0], elapsed_s=time.time() - t0)
        if fields is None:
            fields = list(row)
            with open(out / "metrics.csv", "w", newline="") as fh:
                csv.writer(fh).writerow(fields)
        with open(out / "metrics.csv", "a", newline="") as fh:
            csv.writer(fh).writerow([row.get(k, float("nan")) for k in fields])
        show = [k for k in fields if any(k.endswith(s) for s in
                ("_loss", "_z_mean", "_z_var", "_r8", "_mix", "mse8_net", "mse8_mean_image", "mse8_window", "mse8_silo",
                 "mse8_oracle", "mse8_oracle_blind", "_bias_net", f"_{args.select}"))]
        print(f"validation at step {step}: " + " ".join(f"{k} {row[k]:.4g}" for k in show), flush=True)
        if row[sel_key] < best:  # step 0 counts: never keep a model worse than the start
            best = row[sel_key]
            save("best.pt", step)

    validate(0)
    step = 0
    for step, b in enumerate(dl, 1):
        b, mu, var = _forward(model, b, device, amp_dtype, region)
        loss, mu_loss, var_loss = _losses(args, b, mu, var, region)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema = mu_loss.item() if ema is None else 0.98 * ema + 0.02 * mu_loss.item()
        if step % args.log_every == 0 or step == 1:
            mem = f" gpu peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if device.type == "cuda" else ""
            print(f"step {step} mu_loss {mu_loss.item():.4f} (avg {ema:.4f}) var_loss {var_loss.item():.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f} s{mem}", flush=True)
        last_step = step == args.steps or (args.max_hours and time.time() - t0 > 3600 * args.max_hours)
        if step % args.val_every == 0 or last_step:
            validate(step)
        if step % args.save_every == 0 or last_step:
            save("last.pt", step)
        if last_step:
            break
    print(f"done after {step} steps, {time.time() - t0:.0f} s; best {sel_key} {best:.5g}")


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
                    mu, var = model(x, scale, region=region)
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
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--data", choices=["sim", "real"], default="sim",
                   help="train on simulated windows (true baseline) or real residuals")
    p.add_argument("--init", default="", help="start from this checkpoint (e.g. a simulation-pretrained best.pt)")
    p.add_argument("--stats_cache", default="", help="folder for cached trial stats (default: <out>/stats)")
    p.add_argument("--w_mu", type=float, default=1.0, help="weight of the mean loss")
    p.add_argument("--w_var", type=float, default=0.1, help="weight of the noise (variance) loss")
    p.add_argument("--window_reach", type=int, default=254, help="frames beyond the gap for the blind window stats")
    p.add_argument("--max_log_correction", type=float, default=0.7, help="soft cap of |log| of the baseline correction")
    p.add_argument("--select", choices=["balanced", "mse8", "loss"], default="balanced",
                   help="best.pt criterion on held-out windows: 8-frame MSE vs the mean image, averaged over "
                        "support and background (balanced) or over all pixels (mse8), or the mean loss")
    p.add_argument("--no_flip", action="store_true", help="no random flips of training windows")
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--val_windows", type=int, default=256)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--no_amp", action="store_true", help="disable mixed precision on the GPU")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1000)
