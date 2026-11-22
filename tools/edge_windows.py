"""Edge bias on simulated windows, against the true baseline, in minutes.

usage:
  python edge_windows.py OUT.npz --trials T1.h5 [T2.h5 ...] --ckpt name=path [name=path ...]
      [--kind fastpz] [--pz_legacy] [--n 96] [--stats DIR] [--sim amp=0,0 excess=0 ...]
      [--no_trial_edges] [--device cpu --threads N]
  python edge_windows.py OUT.npz --summary

Each draw is a new virtual trial on a template (``simulate``, as in training)
and one window whose loss region (``out_frames`` frames) lies in the trial's
middle, so its context reaches every way. The same window is predicted three
times: whole ("mid"), with every frame after the loss region removed (NaN, as
past a trial's last frame: "end"), and with every frame before it removed
("start"). Only the truncation differs (same noise, same activity, same
whole-trial maps), so end - mid and start - mid are the edge effect alone, as
in ``edge_pieces.py`` but on many short windows; each is also compared with the
true baseline. Two more windows of the same virtual trial sit at its real
first and last frames ("trial_start", "trial_end"), where the drift
(bleaching) and the simulated activity are those of a real trial edge; their
bias against the truth is not paired, so it needs many more draws.

Reported per pixel class (footprint support, background = valid and not
labeled, all valid), by frames from the edge: (mu - truth) / truth in %
pooled over the class's pixels and frames, mean and SE over draws. For the
mid-trial windows the same frames' bias in the whole window is subtracted
(paired), and the model's own split of the baseline is given: mu = base *
exp(correction), base = mean image + mix * (window mean - mean image), so the
edge effect is (base_edge - base_mid) / truth [the window-mean path] plus the
change of the correction. Path swaps ("by trunk", "by stats", "by extra"): the
truncated window's baseline is recomputed with one input of the fusion layers
taken from the whole window (``model.parts`` / ``model.head``): the trunk
branch on the missing side, the window statistics, or the same-frame context;
edge - swapped is that path's share of the edge effect.
"""
import argparse
import sys
import time
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from slap2_f0.io import TrialFile  # noqa: E402
from slap2_f0.model import make_inputs  # noqa: E402
from slap2_f0.predict import load_checkpoint, prepare  # noqa: E402
from slap2_f0.simulate import SimConfig, Template, draw, window  # noqa: E402
from slap2_f0.stats import load_stats  # noqa: E402
from slap2_f0.train import eval_sim  # noqa: E402

BINS = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128)]
CONDS = ("mid_end", "mid_start", "end", "start", "trial_end", "trial_start")  # mid_*: the whole window, by
# frames from its loss region's end or start (paired with end and start)
CLASSES = ("sup", "bg", "all")
QUANT = ("bias", "base", "corr", "wm", "mix")  # (mu-truth), (base-truth), log(mu/base), (wm-truth), mix


def sim_overrides(cfg: SimConfig, items: list[str]) -> SimConfig:
    """key=value overrides of a SimConfig (tuples as a,b)."""
    types = {f.name: f.type for f in dc_fields(SimConfig)}
    for it in items:
        k, v = it.split("=", 1)
        if k not in types:
            raise SystemExit(f"unknown SimConfig field {k}")
        cur = getattr(cfg, k)
        if isinstance(cur, tuple):
            val = tuple(float(x) for x in v.split(","))
        elif isinstance(cur, bool):
            val = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            val = int(v)
        else:
            val = float(v)
        setattr(cfg, k, val)
    return cfg


def frame_dist(cond: str, nf: int, T: int, t0: int) -> np.ndarray:
    """Frames from the relevant edge for each loss-region frame."""
    t = np.arange(nf)
    if cond in ("end", "mid_end"):
        return nf - 1 - t
    if cond in ("start", "mid_start"):
        return t
    if cond == "trial_end":
        return (T - 1) - (t0 + t)
    return t0 + t  # trial_start


def binned(per_frame: np.ndarray, dist: np.ndarray) -> np.ndarray:
    return np.array([np.nanmean(per_frame[(dist >= a) & (dist < b)]) if ((dist >= a) & (dist < b)).any()
                     else np.nan for a, b in BINS])


# trunk: the branch on the missing side; stats: the window statistics (and with edge_features the per-side
# statistics and gates); extra: the same-frame context (half-planes, pooled and motion features)
def wait_for_gpu(device, need_gb: float, poll_s: float = 20.0):
    """Wait until the GPU has need_gb free: on a shared 8 GB card, running short of memory makes
    Windows page GPU memory to the host and a forward pass takes minutes instead of 0.5 s."""
    if device.type != "cuda" or need_gb <= 0:
        return
    with torch.cuda.device(device):  # the requested GPU, not the current one
        torch.cuda.empty_cache()  # this process's cached blocks count as used
        while torch.cuda.mem_get_info(device)[0] < need_gb * 2**30:
            time.sleep(poll_s)


SWAPS = {"trunk": None, "stats": ("wm", "wv", "n", "edge", "gates"), "extra": ("extra",)}


def parts_to(p: dict, device) -> dict:
    """A ``model.parts`` dict (tensors, lists of tensors, None) on device: a crop's copy, so a cropped
    view no longer holds its whole GPU allocation."""
    f = lambda a: a.to(device) if torch.is_tensor(a) else a  # noqa: E731
    return {k: [f(e) for e in v] if isinstance(v, list) else f(v) for k, v in p.items()}


@torch.no_grad()
def run_models(models, wins, scale, gain, device, tR, nf, need_gb=0.0):
    """Each model on each window (a dict of name -> window); returns {name: {cond: dict of (nf, H', W')
    arrays}}. For "end" and "start" also the baseline with one path of the fusion's inputs taken from
    the whole window ("mid"): the trunk branch on the missing side (future for end, past for start),
    the window statistics, or the same-frame context; edge - swapped is that path's share."""
    out = {}
    one = lambda v: torch.full((1,), v, device=device)  # noqa: E731
    np_ = lambda a: a.float().cpu().numpy()[0]  # noqa: E731
    for name, model, amp, wcfg in models:
        sR = wcfg.s_reach
        op = wins["mid"]["movie"].shape[1] - 2 * sR
        region = (slice(tR, tR + nf), slice(sR, sR + op), slice(sR, sR + op))
        parts, res = {}, {}
        for c, w in wins.items():  # one window at a time: a batch of them does not fit an 8 GB card
            wait_for_gpu(device, need_gb)
            x = make_inputs(torch.from_numpy(w["movie"][None]).to(device),
                            torch.from_numpy(w["mean_image"][None]).to(device),
                            torch.from_numpy(w["varfac"][None]).to(device), one(scale),
                            torch.from_numpy(w["var_map"][None]).to(device), one(gain),
                            varfac_norm=model.cfg.varfac_norm)
            if amp is not None:
                x = x.to(amp)
            with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
                p = model.parts(x, one(scale), one(gain), region)
                mu, _, aux = model.head(p, aux=True)
            res[c] = {"mu": np_(mu) * scale, "base": np_(aux["base"]) * scale,
                      "wm": np_(aux["window_mean"]) * scale, "mix": np_(aux["mix"])}
            if c in ("mid", "end", "start"):  # the swaps' inputs wait on the CPU, so the GPU is free for the
                parts[c] = parts_to(p, "cpu")  # next forward (wait_for_gpu cannot free referenced tensors)
            del x, p, mu, aux
        for c in ("end", "start"):
            if c not in parts or "mid" not in parts:
                continue
            for sw, keys in SWAPS.items():
                keys = keys or (("hf",) if c == "end" else ("hp",))
                q = dict(parts[c])
                q.update({k: parts["mid"][k] for k in keys})
                q = parts_to(q, device)
                with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
                    mu, _ = model.head(q)
                res[f"{c}_sw_{sw}"] = {"mu": np_(mu) * scale}
                del q, mu
        out[name] = res
        del parts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--trials", nargs="+", default=[])
    ap.add_argument("--ckpt", nargs="+", default=[])
    ap.add_argument("--kind", default="fastpz", help="evaluation kind of the simulation (train.eval_sim)")
    ap.add_argument("--pz_legacy", action="store_true", help="low counts as up to 2026-11-21 (v7-v9, S0-S3)")
    ap.add_argument("--sim", nargs="*", default=[], help="SimConfig overrides, key=value (tuples a,b)")
    ap.add_argument("--n", type=int, default=96, help="draws (virtual trials)")
    ap.add_argument("--seed", type=int, default=90_000)
    ap.add_argument("--stats", default="")
    ap.add_argument("--support_frac", type=float, default=0.75, help="share of windows centred on support pixels")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--need_gb", type=float, default=2.5, help="GPU memory to wait for before each forward pass")
    ap.add_argument("--no_trial_edges", action="store_true", help="skip the windows at the real trial edges")
    ap.add_argument("--save_every", type=int, default=8, help="draws between partial saves of OUT")
    ap.add_argument("--out_frames", type=int, default=128, help="loss-region frames per window (edge bins reach 128)")
    ap.add_argument("--out_px", type=int, default=32, help="loss-region pixels per side")
    ap.add_argument("--summary", action="store_true", help="only print the summary of an existing OUT")
    args = ap.parse_args()
    if args.summary:
        z = np.load(args.out)
        arr = {k: z[k] for k in z.files if k != "bins"}
        names = sorted({k.split("|")[0] for k in arr})
        summarize(arr, names, [c for c in CONDS if any(f"|{c}|" in k for k in arr)])
        return
    device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = sim_overrides(eval_sim(args.kind, pz_legacy=args.pz_legacy), args.sim)
    models = []
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        model, wcfg, _ = load_checkpoint(path, device)
        amp = prepare(model, device, "auto")
        models.append((name, model, amp, wcfg))
    wc = models[0][3]
    if any(m[3].t_reach != wc.t_reach or m[3].s_reach != wc.s_reach for m in models):
        raise SystemExit("checkpoints with different reach: run them separately")
    tR, sR, op, nf = wc.t_reach, wc.s_reach, args.out_px, args.out_frames
    nt, ns = nf + 2 * tR, op + 2 * sR
    stats = load_stats(args.trials, args.stats or None)
    tfs = [TrialFile(f) for f in args.trials]
    tps = [Template(tf, stats[str(f)]) for tf, f in zip(tfs, args.trials)]
    conds = CONDS[:4] if args.no_trial_edges else CONDS
    swapped = [f"{c}_sw_{sw}" for c in ("end", "start") for sw in SWAPS]
    # per draw, model, condition, class, quantity: binned values
    res = {(m[0], c, k, q): [] for m in models for c in conds for k in CLASSES for q in QUANT}
    res.update({(m[0], c, k, "bias"): [] for m in models for c in swapped for k in CLASSES})
    t_start = time.time()
    for i in range(args.n):
        rng = np.random.default_rng((args.seed, i))
        j = i % len(tps)
        tp, tf = tps[j], tfs[j]
        r = draw(tp, cfg, rng)
        pool = tf.support & tf.valid if rng.random() < args.support_frac and (tf.support & tf.valid).any() else tf.valid
        vr, vc = np.nonzero(pool)
        k = rng.integers(len(vr))
        r0, c0 = int(vr[k]) - op // 2 - sR, int(vc[k]) - op // 2 - sR
        T = tp.T
        tm = int(rng.integers(tR, T - nf - tR + 1))  # loss region start, fully mid-trial
        w = window(tp, r, tm - tR, nt, r0, ns, c0, ns, rng)
        wins = {"mid": w}
        for cond, sl in (("end", slice(tR + nf, None)), ("start", slice(None, tR))):
            v = dict(w)
            v["movie"] = w["movie"].copy()
            v["movie"][sl] = np.nan
            wins[cond] = v
        t0s = {"mid": tm, "end": tm, "start": tm}
        if not args.no_trial_edges:
            wins["trial_end"] = window(tp, r, T - nf - tR, nt, r0, ns, c0, ns, rng)
            wins["trial_start"] = window(tp, r, -tR, nt, r0, ns, c0, ns, rng)
            t0s.update(trial_end=T - nf, trial_start=0)
        pred = run_models(models, wins, float(tf.scale), float(stats[str(args.trials[j])]["gain"]), device, tR, nf,
                          args.need_gb)
        sup = w["support"][sR:sR + op, sR:sR + op]
        val = np.isfinite(w["mean_image"][sR:sR + op, sR:sR + op])
        lab = w["labeled"][sR:sR + op, sR:sR + op]
        masks = {"sup": sup & val, "bg": val & ~lab, "all": val}
        for cond in list(conds) + swapped:
            src = "mid" if cond.startswith("mid") else cond.split("_sw_")[0]
            truth = wins[src]["truth"][tR:tR + nf, sR:sR + op, sR:sR + op]
            dist = frame_dist(src if src != "mid" else cond, nf, T, t0s[src])
            for name, *_ in models:
                p = pred[name][cond if "_sw_" in cond else src]
                if "_sw_" in cond:
                    q = {"bias": p["mu"] - truth}
                else:
                    q = {"bias": p["mu"] - truth, "base": p["base"] - truth, "corr": np.log(p["mu"] / p["base"]),
                         "wm": p["wm"] - truth, "mix": p["mix"]}
                for kc, m in masks.items():
                    if m.sum() < 5:
                        for qq in q:
                            res[(name, cond, kc, qq)].append(np.full(len(BINS), np.nan))
                        continue
                    lvl = np.nanmean(truth[:, m])
                    for qq, a in q.items():
                        pf = np.nanmean(a[:, m], 1)
                        if qq in ("bias", "base", "wm"):
                            pf = 100 * pf / lvl
                        res[(name, cond, kc, qq)].append(binned(pf, dist))
        if (i + 1) % 8 == 0 or i == 0:
            print(f"draw {i + 1}/{args.n}, {time.time() - t_start:.0f} s", flush=True)
        if (i + 1) % args.save_every == 0 and i + 1 < args.n:  # partial results
            np.savez(args.out, bins=np.array(BINS), **{"|".join(k): np.array(v) for k, v in res.items()})
    for tf in tfs:
        tf.close()
    arr = {"|".join(k): np.array(v) for k, v in res.items()}
    np.savez(args.out, bins=np.array(BINS), **arr)
    summarize(arr, [m[0] for m in models], list(conds))


def summarize(arr, names, conds):
    print(f"\nbins (frames from the edge): {BINS}; % of the true baseline, mean +- SE over draws")
    for name in names:
        for kc in CLASSES:
            for cond in conds:
                b = arr[f"{name}|{cond}|{kc}|bias"]
                line = f"{name:10s} {kc:3s} {cond:11s} bias      " + _fmt(b)
                print(line)
                if cond in ("end", "start"):
                    for qq, lab in (("bias", "edge-mid"), ("base", "base e-m"), ("wm", "wmean e-m")):
                        d = arr[f"{name}|{cond}|{kc}|{qq}"] - arr[f"{name}|mid_{cond}|{kc}|{qq}"]
                        print(f"{name:10s} {kc:3s} {cond:11s} {lab:9s} " + _fmt(d))
                    d = 100 * (arr[f"{name}|{cond}|{kc}|corr"] - arr[f"{name}|mid_{cond}|{kc}|corr"])
                    print(f"{name:10s} {kc:3s} {cond:11s} corr e-m  " + _fmt(d))
                    print(f"{name:10s} {kc:3s} {cond:11s} mix       " + _fmt(arr[f"{name}|{cond}|{kc}|mix"], pct=False)
                          + "  (mid " + _fmt(arr[f"{name}|mid_{cond}|{kc}|mix"], pct=False, se=False) + ")")
                    for sw in SWAPS:
                        key = f"{name}|{cond}_sw_{sw}|{kc}|bias"
                        if key in arr:  # the path's share of the edge effect: edge - (edge with that path from mid)
                            d = arr[f"{name}|{cond}|{kc}|bias"] - arr[key]
                            print(f"{name:10s} {kc:3s} {cond:11s} {'by ' + sw:9s} " + _fmt(d))


def _fmt(a, pct=True, se=True):
    m = np.nanmean(a, 0)
    s = np.nanstd(a, 0) / np.sqrt(np.maximum(np.isfinite(a).sum(0), 1))
    if pct:
        return " ".join(f"{x:+6.2f}" + (f"+-{y:4.2f}" if se else "") for x, y in zip(m, s))
    return " ".join(f"{x:+5.2f}" for x in m)


if __name__ == "__main__":
    main()
