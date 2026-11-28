"""Profile the reference predictor on one trial-path (slap2-F0-network#33).

usage: prof_tile.py MOCO STEM CKPT OUT_JSON [--out_frames N] [--tiles K] [--bench 0|1]
Loads the trial as predict-registered does (times load and stats), then runs
the first K tiles of predict_trial's tiling with out_frames N and records:
CUDA time per top-level module (forward hooks with CUDA events), the top
kernels (torch.profiler), and the share of wall time the GPU is busy.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from slap2_f0 import predict as P
from slap2_f0 import registered as R
from slap2_f0.model import make_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("moco")
    ap.add_argument("stem")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--out_frames", type=int, default=512)
    ap.add_argument("--tiles", type=int, default=3)
    ap.add_argument("--bench", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = bool(a.bench)
    rec = {"args": vars(a), "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
    moco = Path(a.moco)
    t0 = time.time()
    if (moco / a.stem / "done").exists():  # prep_cache.py's files (load and stats timed there)
        from bench_engines import load_cache
        tr, st, movie, meta = load_cache(moco / a.stem)
        t1 = t2 = time.time()
        rec["cache_meta"] = meta
    else:
        movie_path = R.registered_movies(moco)
        movie_path = [m for m in movie_path if m.name.startswith(a.stem + "_")][0]
        movie, valid, discard, varfac, hz = R.load_registered(movie_path, moco / f"{a.stem}_ALIGNMENTDATA.h5")
        t1 = time.time()
        tr = R.ArrayTrial(movie, valid, discard, varfac)
        st = R.movie_stats(movie, valid, discard)
        t2 = time.time()
    valid = tr.valid
    rec.update(load_s=t1 - t0, stats_s=t2 - t1, shape=list(movie.shape), valid=int(valid.sum()))
    print(rec, flush=True)
    model, wcfg, ck = P.load_checkpoint(a.ckpt, dev)
    amp = P.prepare(model, dev, "auto")
    T, Rr, C = movie.shape
    tR, sR, op = wcfg.t_reach, wcfg.s_reach, wcfg.out_px
    ns = op + 2 * sR
    rows, cols = P.valid_index(tr)
    index = np.full((Rr, C), -1, np.int64)
    index[rows, cols] = np.arange(len(rows))
    origins = P.tile_origins(index >= 0, op)
    tiles = [(r0, c0) for r0, cs in origins for c0 in cs]
    rec["n_tiles_48"] = len(tiles)
    nf = a.out_frames
    rec["time_tiles"] = -(-T // nf)
    # one input of the reference's geometry (a tile in the middle of the trial)
    pad = np.full((T + 2 * tR, Rr + 2 * sR + op, C + 2 * sR + op), np.nan, np.float32)
    pad[tR:tR + T, sR:sR + Rr, sR:sR + C] = movie
    mean_p = np.full(pad.shape[1:], np.nan, np.float32)
    mean_p[sR:sR + Rr, sR:sR + C] = np.where(tr.valid, tr.mean_image, np.nan)
    vf_p = np.zeros_like(mean_p)
    vf_p[sR:sR + Rr, sR:sR + C] = np.nan_to_num(tr.varfac)
    var_p = np.full_like(mean_p, np.nan)
    var_p[sR:sR + Rr, sR:sR + C] = st["var_movie"]
    var_p = np.where(np.isfinite(mean_p), var_p, np.nan).astype(np.float32)
    s = float(tr.scale)

    def make(r0, c0, tt0):
        m = pad[tt0:tt0 + nf + 2 * tR, r0:r0 + ns, c0:c0 + ns][None]
        sp = lambda q: torch.from_numpy(np.ascontiguousarray(q[r0:r0 + ns, c0:c0 + ns][None])).to(dev)  # noqa: E731
        sc = torch.full((1,), s, device=dev)
        g = torch.full((1,), float(st["gain"]), device=dev)
        x = make_inputs(torch.from_numpy(np.ascontiguousarray(m)).to(dev), sp(mean_p), sp(vf_p), sc, sp(var_p), g,
                        varfac_norm=model.cfg.varfac_norm)
        return x, sc, g

    region = (slice(tR, tR + nf), slice(sR, sR + op), slice(sR, sR + op))
    # module timing with CUDA events
    events = {}

    def pre(name):
        def f(mod, inp):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            events.setdefault(name, []).append([e, None])
        return f

    def post(name):
        def f(mod, inp, out):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            events[name][-1][1] = e
        return f

    mods = {"past": model.past, "future": model.future, "fuse": model.fuse}
    for k, m in enumerate(model.hp_views or []):
        mods[f"hp{k}"] = m
    for b in model.fuse_more or []:
        mods.setdefault("fuse_more", b)
    hooks = []
    for name, m in mods.items():
        hooks += [m.register_forward_pre_hook(pre(name)), m.register_forward_hook(post(name))]
    wall, fwd = [], []
    tt0 = min(nf, max(0, T - nf))  # a middle time tile
    with torch.no_grad():
        for i, (r0, c0) in enumerate(tiles[:a.tiles + 1]):
            torch.cuda.synchronize()
            w0 = time.time()
            x, sc, g = make(r0, c0, tt0)
            torch.cuda.synchronize()
            w1 = time.time()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            with torch.autocast("cuda", dtype=amp):
                mu, var = model(x, sc, g, region)
            e1.record()
            (mu.float() * s).cpu().numpy()  # the result's copy to the host, as predict_trial
            torch.cuda.synchronize()
            w2 = time.time()
            if i:  # the first is warm-up
                wall.append([w1 - w0, w2 - w1])
                fwd.append(e0.elapsed_time(e1) / 1000)
    for h in hooks:
        h.remove()
    torch.cuda.synchronize()
    per = {k: float(np.mean([a.elapsed_time(b) for a, b in v[1:]]) / 1000) for k, v in events.items()}
    rec.update(prep_s=float(np.mean([w[0] for w in wall])), fwd_wall_s=float(np.mean([w[1] for w in wall])),
               fwd_cuda_s=float(np.mean(fwd)), modules_s=per,
               peak_gb=torch.cuda.max_memory_allocated() / 2**30)
    rec["modules_s"]["rest"] = rec["fwd_cuda_s"] - sum(per.values())
    Path(a.out).write_text(json.dumps(rec, indent=1))
    from torch.profiler import ProfilerActivity, profile
    x, sc, g = make(*tiles[1], tt0)
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.autocast("cuda", dtype=amp):
            mu, var = model(x, sc, g, region)
        torch.cuda.synchronize()
    ev = prof.key_averages()
    tot = sum(e.self_device_time_total for e in ev) or 1
    rec["kernels"] = [[round(100 * e.self_device_time_total / tot, 1), e.count, e.key[:120]]
                      for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:25]]
    rec["kernel_total_s"] = tot / 1e6
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
