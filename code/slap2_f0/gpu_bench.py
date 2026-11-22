"""Machine benchmark for choosing the GPU capsule's machine (docs/plan.md, "GPU machine").

One configuration (run as ``python -m slap2_f0.gpu_bench CKPT key=value ...``):
  mode=step    time one training step (forward, backward, AdamW) of the checkpoint's
               model on random inputs of its window size (default);
  mode=eval    the forward pass alone, as in evaluation;
  mode=loader  windows per second from the data loaders (needs data=ROOT).
keys: out_px, out_frames, batch, width, hp_width, extra_blocks (repeat every
branch's dilation lists this many more times), checkpoint (0/1, activation
checkpointing), bench (0/1, cuDNN autotuning), cl (0/1, channels-last),
compile (0/1, torch.compile), steps, profile (0/1, top CUDA kernels); for
mode=loader: data, kind (real, sim, fast), workers, windows. out=FILE appends a
CSV row.

The grid (``slap2-f0 bench CKPT --out DIR [--data ROOT]``) runs each
configuration in its own process, so one running out of memory or a compiled
graph does not affect the next.
"""
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

DEFAULT_GRID = [
    # training step at the trained window (192 x 48^2 loss region)
    "bench=0", "bench=1", "bench=1 checkpoint=0", "bench=1 compile=1", "bench=1 checkpoint=0 compile=1",
    "bench=1 batch=2", "bench=1 batch=4", "bench=1 checkpoint=0 batch=2",
    # larger loss regions
    "bench=1 out_px=96", "bench=1 out_px=96 checkpoint=0", "bench=1 out_frames=448",
    "bench=1 out_px=96 out_frames=448",
    # deeper and wider networks
    "bench=1 extra_blocks=1", "bench=1 extra_blocks=1 checkpoint=0", "bench=1 extra_blocks=1 compile=1",
    "bench=1 width=48 hp_width=36", "bench=1 width=64 hp_width=48", "bench=1 extra_blocks=1 width=48 hp_width=36",
    "bench=1 extra_blocks=1 width=48 hp_width=36 checkpoint=0",
    # evaluation's forward pass
    "mode=eval bench=0", "mode=eval bench=1", "mode=eval bench=1 batch=4", "mode=eval bench=1 compile=1",
]
LOADER_GRID = ["mode=loader kind=fast workers=7", "mode=loader kind=fast workers=12", "mode=loader kind=real workers=7"]


def _one(ckpt: str, kv: dict) -> dict:
    from .data import WindowConfig, loss_region
    from .model import SlabBlindSpotNet, config_from_dict, make_inputs

    g = lambda k, d: type(d)(kv.get(k, d))  # noqa: E731
    mode = kv.get("mode", "step")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = config_from_dict(ck["config"])
    reps = g("extra_blocks", 0)
    rep = (lambda a: list(a) * (1 + reps)) if reps else (lambda a: a)
    mcfg = replace(mcfg, checkpoint=bool(g("checkpoint", 1)), width=g("width", mcfg.width),
                   hp_width=g("hp_width", mcfg.hp_width), temporal_dilations=rep(mcfg.temporal_dilations),
                   spatial_dilations=rep(mcfg.spatial_dilations), hp_temporal_dilations=rep(mcfg.hp_temporal_dilations),
                   hp_spatial_dilations=rep(mcfg.hp_spatial_dilations))
    w = WindowConfig(**ck["window"])
    w = replace(w, out_px=g("out_px", w.out_px), out_frames=g("out_frames", w.out_frames))
    row = {"mode": mode, **{k: v for k, v in kv.items() if k not in ("mode", "out", "data")}}
    if mode == "loader":
        return {**row, **_loader(kv, w)}
    torch.backends.cudnn.benchmark = bool(g("bench", 0))
    dev = torch.device("cuda")
    model = SlabBlindSpotNet(mcfg).to(dev)
    if g("cl", 0):
        model = model.to(memory_format=torch.channels_last_3d)
    region = loss_region(w)
    B = g("batch", 1)
    T, S = w.out_frames + 2 * w.t_reach, w.out_px + 2 * w.s_reach
    movie = torch.rand(B, T, S, S, device=dev) * 20
    mean = torch.full((B, S, S), 10.0, device=dev)
    vf = torch.ones(B, S, S, device=dev)
    var_map = torch.full((B, S, S), 30.0, device=dev)
    gain = torch.full((B,), 8.0, device=dev)
    scale = torch.full((B,), 10.0, device=dev)
    y = torch.rand(B, w.out_frames, w.out_px, w.out_px, device=dev)
    fwd = torch.compile(model) if g("compile", 0) else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def inputs():
        x = make_inputs(movie, mean, vf, scale, var_map=var_map, gain=gain)
        return x.contiguous(memory_format=torch.channels_last_3d) if g("cl", 0) else x

    def step():
        if mode == "eval":
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                fwd(inputs(), scale, region=region)
            return
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, var = fwd(inputs(), scale, region=region)
        loss = ((y - mu.float()) ** 2).mean() + 0.1 * (torch.log(var.float()) + (y - mu.float().detach()) ** 2 / var.float()).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    row.update(window=f"{T}x{S}x{S}", out=f"{w.out_frames}x{w.out_px}x{w.out_px}",
               params=sum(p.numel() for p in model.parameters()))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        step()
        step()
        torch.cuda.synchronize()
        row["warmup_s"] = time.time() - t0
        n = g("steps", 4)
        t0 = time.time()
        for _ in range(n):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n
        row.update(s_per_step=dt, s_per_window=dt / B, peak_gb=torch.cuda.max_memory_allocated() / 2**30,
                   out_voxels_per_s=B * w.out_frames * w.out_px ** 2 / dt, status="ok")
    except torch.OutOfMemoryError:
        row.update(status="out of memory", peak_gb=torch.cuda.max_memory_allocated() / 2**30)
    if g("profile", 0) and row.get("status") == "ok":
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        ev = prof.key_averages()
        tot = sum(e.self_device_time_total for e in ev)
        for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:15]:
            print(f"  {100 * e.self_device_time_total / tot:5.1f}%  {e.count:6d}  {e.key[:110]}")
    return row


def _loader(kv: dict, w) -> dict:
    """Windows per second from DataLoader workers, as in training or evaluation."""
    from torch.utils.data import DataLoader

    from .data import WindowDataset
    from .io import list_trials
    from .stats import load_stats
    from .train import EVAL_SEEDS, eval_sim

    kind, workers, n = kv.get("kind", "fast"), int(kv.get("workers", 7)), int(kv.get("windows", 96))
    files = list_trials(kv["data"])
    t0 = time.time()
    stats = load_stats(files, None, workers=min(workers, 6))
    files = [f for f in files if str(f) in stats]
    st_s = time.time() - t0
    ds = WindowDataset(files, w, stats, length=n + 2 * workers, seed=EVAL_SEEDS.get(kind, 0),
                       sim=eval_sim(kind) if kind != "real" else None, flip=False)
    dl = DataLoader(ds, batch_size=1, num_workers=workers)
    it = iter(dl)
    for _ in range(2 * workers):  # warm up: every worker opens trials and builds templates
        next(it)
    t0 = time.time()
    for _ in range(n):
        next(it)
    dt = time.time() - t0
    return {"status": "ok", "windows_per_s": n / dt, "stats_s": st_s, "files": len(files), "cpus": os.cpu_count()}


def grid(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configs = [c for c in (args.configs.split(";") if args.configs else DEFAULT_GRID) if c.strip()]
    if args.data:
        configs += [f"{c} data={args.data}" for c in LOADER_GRID]
    print(json.dumps({"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      "cpus": os.cpu_count(), "torch": torch.__version__}), flush=True)
    for c in configs:
        cmd = [sys.executable, "-u", "-m", "slap2_f0.gpu_bench", args.ckpt, *c.split(), f"out={out / 'bench.csv'}"]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, timeout=args.timeout, capture_output=True, text=True)
            tail = (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0 else (r.stderr.strip().splitlines() or [""])[-1]
        except subprocess.TimeoutExpired:
            tail = "timeout"
        print(f"[{time.time() - t0:5.0f} s] {c}: {tail[:300]}", flush=True)


def main(argv: list[str]) -> None:
    ckpt, kv = argv[0], dict(a.split("=", 1) for a in argv[1:])
    row = _one(ckpt, kv)
    if "out" in kv:
        p = Path(kv["out"])
        new = not p.exists()
        with open(p, "a", newline="") as fh:
            wr = csv.writer(fh)
            if new:
                wr.writerow(["json"])
            wr.writerow([json.dumps(row)])
    print(json.dumps(row), flush=True)


def add_arguments(p) -> None:
    p.add_argument("ckpt", help="a trained checkpoint (its model and window configuration are used)")
    p.add_argument("--out", default="results/bench")
    p.add_argument("--data", default="", help="exported trials, for the data-loader rates (optional)")
    p.add_argument("--configs", default="", help="';'-separated configurations of key=value pairs (default: the grid)")
    p.add_argument("--timeout", type=float, default=900, help="seconds per configuration")


if __name__ == "__main__":
    main(sys.argv[1:])
