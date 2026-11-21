"""Time one training step (forward + backward + AdamW) of a trained checkpoint's
model on random inputs of its real window size (docs/plan.md, "GPU machine").

usage: python gpu_bench.py CKPT.pt [key=value ...]
keys: out_px, out_frames, batch, width, hp_width, checkpoint (0/1, activation
checkpointing), bench (0/1, cuDNN autotuning), cl (0/1, channels-last), compile
(0/1, torch.compile; needs Triton), extra_blocks (repeat every branch's
dilation lists this many more times), profile (0/1, top CUDA kernels), steps."""
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from slap2_f0.data import WindowConfig, loss_region  # noqa: E402
from slap2_f0.model import SlabBlindSpotNet, config_from_dict, make_inputs  # noqa: E402
from slap2_f0.train import student_t_nll  # noqa: E402

kv = dict(a.split("=") for a in sys.argv[2:])
g = lambda k, d: type(d)(kv.get(k, d))  # noqa: E731
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
mcfg = config_from_dict(ck["config"])
reps = g("extra_blocks", 0)
mcfg = replace(mcfg, checkpoint=bool(g("checkpoint", 1)), width=g("width", mcfg.width), hp_width=g("hp_width", mcfg.hp_width),
               temporal_dilations=mcfg.temporal_dilations * (1 + reps) if reps else mcfg.temporal_dilations,
               spatial_dilations=mcfg.spatial_dilations * (1 + reps) if reps else mcfg.spatial_dilations,
               hp_temporal_dilations=mcfg.hp_temporal_dilations * (1 + reps) if reps else mcfg.hp_temporal_dilations,
               hp_spatial_dilations=mcfg.hp_spatial_dilations * (1 + reps) if reps else mcfg.hp_spatial_dilations)
w = WindowConfig(**ck["window"])
w = replace(w, out_px=g("out_px", w.out_px), out_frames=g("out_frames", w.out_frames))
region = loss_region(w)
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = bool(g("bench", 0))
model = SlabBlindSpotNet(mcfg).to(dev)
if g("cl", 0):
    model = model.to(memory_format=torch.channels_last_3d)
nparam = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
B = g("batch", 1)
T, S = w.out_frames + 2 * w.t_reach, w.out_px + 2 * w.s_reach
movie = torch.rand(B, T, S, S, device=dev) * 20
mean = torch.full((B, S, S), 10.0, device=dev)
vf = torch.ones(B, S, S, device=dev)
var_map = torch.full((B, S, S), 30.0, device=dev)
gain = torch.full((B,), 8.0, device=dev)
scale = torch.full((B,), 10.0, device=dev)
y = torch.rand(B, w.out_frames, w.out_px, w.out_px, device=dev)
wt = torch.ones_like(y)
fwd = torch.compile(model) if g("compile", 0) else model


def step():
    x = make_inputs(movie, mean, vf, scale, var_map=var_map, gain=gain)
    if g("cl", 0):
        x = x.contiguous(memory_format=torch.channels_last_3d)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mu, var = fwd(x, scale, region=region)
    loss = student_t_nll(y, mu.float(), var.float(), wt)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


torch.cuda.reset_peak_memory_stats()
for _ in range(2):
    step()
torch.cuda.synchronize()
n = g("steps", 4)
t0 = time.time()
for _ in range(n):
    step()
torch.cuda.synchronize()
dt = (time.time() - t0) / n
print(f"window {T}x{S}^2 out {w.out_frames}x{w.out_px}^2 batch {B} params {nparam:,} ckpt {int(mcfg.checkpoint)} "
      f"compile {g("compile", 0)} cudnn_bench {g("bench", 0)} cl {g("cl", 0)} extra_blocks {reps} width {mcfg.width}/{mcfg.hp_width}: {dt:.3f} s/step, "
      f"{dt / B:.3f} s/window, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
if g("profile", 0):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step(); torch.cuda.synchronize()
    ev = prof.key_averages()
    tot = sum(e.device_time_total for e in ev if e.device_type.name == "CUDA") or sum(e.self_device_time_total for e in ev)
    rows = sorted(ev, key=lambda e: -e.self_device_time_total)[:25]
    tot = sum(e.self_device_time_total for e in ev)
    for e in rows:
        print(f"  {100 * e.self_device_time_total / tot:5.1f}%  {e.count:6d}  {e.key[:110]}")
