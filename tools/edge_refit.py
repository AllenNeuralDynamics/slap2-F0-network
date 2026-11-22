"""Can the fusion layers learn the trial edges? Refit them on a frozen trunk.

usage:
  python edge_refit.py OUT_DIR --trials T1.h5 [T2.h5 ...] --ckpt PATH [--kind fastpz --pz_legacy]
      [--n_train 96] [--n_eval 24] [--steps 3000] [--arms gated,ctrl,edge] [--stats DIR]
      [--save_cache | --load_cache CACHE.pt [--extra_train N] [--extra_eval N]] [--device cpu --threads N]

The checkpoint's trunk (past and future branches, half-planes, pooled
features: everything before the pointwise fusion layers, ``model.parts``) is
run once on simulated windows and its outputs cached; only the fusion layers
and heads are then trained, from the checkpoint's weights, with training's
loss (``train._losses``), in up to three arms on the same cache:

* ctrl: the checkpoint's architecture (edge features only if it has them);
* edge: with ``edge_features`` (per-side window statistics into the fusion and
  edge-gated corrections), warm-started so that it starts as ctrl;
* gated: the same model, but only the edge-gated head is trained (132
  parameters): the baseline in a trial's middle cannot change (the gates are
  0 there), so this asks only whether a correction for the missing side, read
  from the features the checkpoint already fuses, removes the edge bias.

Training windows are placed as in training (``data.WindowDataset``): with
probability ``--edge_frac`` the loss region starts or ends at a trial edge
(+-64 frames), else anywhere. Evaluation windows are those of
``edge_windows.py`` (whole, end- and start-truncated, paired), cached the same
way. Reported: the edge bias (truncated - whole, % of the true baseline) at
support and background pixels by frames from the edge, before and after each
refit, and the mid-window error against the truth (so that a gain at the edges
is not paid for in the middle). Training windows draw seeds 95_000+ (96_000+
for ``--extra_train``, all at trial edges), evaluation windows 90_000+ (as
``edge_windows.py``), so the two never share a draw.

A cache (``--save_cache``) records what its trunk outputs depend on (hashes of
the trunk's weights and of the trials' statistics, the checkpoint's config and
window, the simulation and the window sizes); ``--load_cache`` reuses it only
when these match.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_windows import BINS, frame_dist, wait_for_gpu  # noqa: E402
from slap2_f0.io import TrialFile  # noqa: E402
from slap2_f0.losses import gaussian_nll  # noqa: E402
from slap2_f0.model import SlabBlindSpotNet, config_from_dict, make_inputs  # noqa: E402
from slap2_f0.simulate import Template, draw, window  # noqa: E402
from slap2_f0.stats import load_stats  # noqa: E402
from slap2_f0.train import eval_sim, load_compatible  # noqa: E402

HEAD = ("fuse", "head_mu", "head_mix", "head_var", "head_edge", "log_floor")  # trained; the rest is frozen


def _digest(h, v):
    """Feed a value (dicts, arrays, tensors, scalars) into the hash h."""
    if isinstance(v, dict):
        for k in sorted(v, key=str):
            h.update(str(k).encode())
            _digest(h, v[k])
        return
    a = v.detach().cpu().float().numpy() if torch.is_tensor(v) else np.asarray(v)
    if a.dtype == object:
        h.update(repr(v).encode())
    else:
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(np.ascontiguousarray(a).tobytes())


def cache_source(args, ck, stats) -> dict:
    """What cached trunk outputs depend on: the frozen weights (everything but HEAD), the checkpoint's
    config and window, the simulation, the window sizes, the template trials and their resolved
    statistics (the simulator's baseline, gain and noise; a replaced sidecar or --stats cache changes
    them), the weights and statistics hashed."""
    h = hashlib.sha256()
    _digest(h, {k: v for k, v in ck["model"].items() if not k.startswith(HEAD)})
    hs = hashlib.sha256()
    _digest(hs, {str(i): stats[str(t)] for i, t in enumerate(args.trials)})
    return {"trunk_sha256": h.hexdigest(), "stats_sha256": hs.hexdigest(), "config": dict(ck["config"]),
            "window": dict(ck["window"]), "kind": args.kind, "pz_legacy": bool(args.pz_legacy),
            "out_frames": args.out_frames, "out_px": args.out_px, "trials": [str(t) for t in args.trials]}


def build(ck, edge: bool, device):
    """The checkpoint's model, with edge features added if ``edge`` (a checkpoint that has them keeps them)."""
    cfg = config_from_dict(ck["config"])
    cfg.edge_features = cfg.edge_features or edge
    m = SlabBlindSpotNet(cfg).to(device)
    load_compatible(m, ck["model"], ck["config"])
    return m


@torch.no_grad()
def cache_window(model, w, scale, gain, device, tR, op, sR, nf, need_gb):
    """model.parts on the loss region of window w, on the CPU in float16 (plus the targets)."""
    wait_for_gpu(device, need_gb)
    one = lambda v: torch.full((1,), v, device=device)  # noqa: E731
    x = make_inputs(torch.from_numpy(w["movie"][None]).to(device), torch.from_numpy(w["mean_image"][None]).to(device),
                    torch.from_numpy(w["varfac"][None]).to(device), one(scale),
                    torch.from_numpy(w["var_map"][None]).to(device), one(gain), varfac_norm=model.cfg.varfac_norm)
    if next(model.parameters()).dtype == torch.float16:
        x = x.half()
    region = (slice(tR, tR + nf), slice(sR, sR + op), slice(sR, sR + op))
    p = model.parts(x, one(scale), one(gain), region)
    half = lambda a: a.detach().to("cpu", torch.float16) if a.is_floating_point() else a.cpu()  # noqa: E731
    out = {k: (half(v) if torch.is_tensor(v) else [half(e) for e in v] if isinstance(v, list) else v)
           for k, v in p.items()}
    for k in ("n",):  # counts stay exact
        out[k] = p[k].detach().cpu()
    sl = (slice(tR, tR + nf), slice(sR, sR + op), slice(sR, sR + op))
    out["truth"] = torch.from_numpy(np.nan_to_num(w["truth"][sl] / scale)).half()[None]
    out["noisy"] = torch.from_numpy(np.nan_to_num(w["noisy"][sl] / scale)).half()[None]
    wt = np.isfinite(w["movie"][sl]) & np.isfinite(w["truth"][sl]) & np.isfinite(w["noisy"][sl])
    out["weight"] = torch.from_numpy(wt.astype(np.float32)).half()[None]
    mi = np.nan_to_num(w["mean_image"][sR:sR + op, sR:sR + op] / scale)
    gs = gain / scale
    out["ref_var"] = torch.from_numpy((gs * np.maximum(mi, 1e-3) + 1.0 / scale ** 2).astype(np.float32))[None, None]
    out["scale"] = scale
    return out


def to_dev(p, device, model):
    """A cached window on the device in float32, as the model's head takes it (without the edge
    statistics for a model without edge_features)."""
    f = lambda a: a.to(device).float() if torch.is_tensor(a) else a  # noqa: E731
    out = {k: (f(v) if torch.is_tensor(v) else [f(e) for e in v] if isinstance(v, list) else v) for k, v in p.items()}
    if not model.cfg.edge_features:
        out["edge"] = out["gates"] = None
    return out


def loss_of(model, p, w_var=0.1):
    mu, var = model.head(p)
    w = p["weight"]
    mu_loss = (w * (p["truth"] - mu) ** 2 / p["ref_var"]).sum() / w.sum().clamp_min(1.0)
    return mu_loss + w_var * gaussian_nll(p["noisy"], mu.detach(), var, w), mu_loss


class AdamW:
    """Minimal AdamW (as torch.optim.AdamW, default betas), for refits on the CPU: torch.optim's
    step probes the CUDA stream, and on a shared GPU that is full, opening the context hangs
    for many minutes (hiding the GPU with CUDA_VISIBLE_DEVICES=-1 crashes torch 2.14 on Windows)."""

    def __init__(self, params, lr, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8):
        self.params, self.lr, self.wd, self.b, self.eps, self.t = list(params), lr, weight_decay, betas, eps, 0
        self.m = [torch.zeros_like(q) for q in self.params]
        self.v = [torch.zeros_like(q) for q in self.params]

    def zero_grad(self):
        for q in self.params:
            q.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = self.b
        for q, m, v in zip(self.params, self.m, self.v):
            if q.grad is None:
                continue
            q.mul_(1 - self.lr * self.wd)
            m.mul_(b1).add_(q.grad, alpha=1 - b1)
            v.mul_(b2).addcmul_(q.grad, q.grad, value=1 - b2)
            q.addcdiv_(m / (1 - b1 ** self.t), (v / (1 - b2 ** self.t)).sqrt_().add_(self.eps), value=-self.lr)


def refit(model, train, device, steps, lr, batch, seed, only=HEAD):
    for n_, prm in model.named_parameters():
        prm.requires_grad_(n_.split(".")[0] in only)
    opt = AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    ema = None
    for step in range(1, steps + 1):
        opt.zero_grad()
        tot = 0.0
        for i in rng.integers(len(train), size=batch):
            loss, mu_loss = loss_of(model, to_dev(train[i], device, model))
            (loss / batch).backward()
            tot += mu_loss.item() / batch
        torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
        opt.step()
        ema = tot if ema is None else 0.98 * ema + 0.02 * tot
        if step % 500 == 0 or step == 1:
            print(f"  step {step} mu_loss {ema:.4f}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, evals, device):
    """Edge bias (truncated - whole, % of the truth) per class and condition, by frames from the edge,
    and the whole windows' MSE against the truth (relative to the oracle constant: truth's mean)."""
    out = {}
    se_net = se_or = 0.0
    for d in evals:
        mus = {c: model.head(to_dev(d[c], device, model))[0][0].cpu().numpy() for c in ("mid", "end", "start")}
        truth = d["mid"]["truth"][0].float().numpy()
        wt = d["mid"]["weight"][0].float().numpy() > 0
        se_net += float(((mus["mid"] - truth) ** 2)[wt].sum())
        tm = np.where(wt, truth, np.nan)
        se_or += float(np.nansum((tm - np.nanmean(tm, 0, keepdims=True)) ** 2))
        nf = truth.shape[0]
        for kc, m in d["masks"].items():
            if m.sum() < 5:
                continue
            lvl = np.nanmean(truth[:, m])
            for c in ("end", "start"):
                diff = 100 * np.nanmean((mus[c] - mus["mid"])[:, m], 1) / lvl
                dist = frame_dist(c, nf, 0, 0)
                b = [np.nanmean(diff[(dist >= lo) & (dist < hi)]) for lo, hi in BINS]
                out.setdefault(f"{kc}|{c}", []).append(b)
            bias = 100 * np.nanmean((mus["mid"] - truth)[:, m], 1) / lvl  # once per draw (its SE is over draws)
            out.setdefault(f"{kc}|mid_bias", []).append([np.nanmean(bias)])
    res = {k: (np.nanmean(np.array(v), 0).tolist(), (np.nanstd(np.array(v), 0) / np.sqrt(len(v))).tolist())
           for k, v in out.items()}
    res["mse_vs_oracle"] = se_net / max(se_or, 1e-12)
    return res


def show(name, r):
    print(f"{name}: mid-window MSE / oracle {r['mse_vs_oracle']:.3f}")
    for k in ("sup|end", "sup|start", "bg|end", "bg|start"):
        if k in r:
            m, s = r[k]
            print(f"  {k:10s} " + " ".join(f"{a:+6.2f}+-{b:4.2f}" for a, b in zip(m, s)))
    for k in ("sup|mid_bias", "bg|mid_bias"):
        if k in r:
            print(f"  {k:12s} {r[k][0][0]:+.2f}")


def training_windows(args, tps, tfs, stats, cfg, trunk, device, n, seed, edge_frac):
    """Cached trunk outputs of n simulated training windows, placed as in training: with probability
    edge_frac the loss region starts or ends at a trial edge (+-64 frames), else anywhere."""
    tR, sR = args.t_reach, args.s_reach
    nf, op = args.out_frames, args.out_px
    nt, ns = nf + 2 * tR, op + 2 * sR
    out, t0 = [], time.time()
    for i in range(n):
        rng = np.random.default_rng((seed, i))
        j = i % len(tps)
        tp, tf = tps[j], tfs[j]
        st = stats[str(args.trials[j])]
        r = draw(tp, cfg, rng)
        if rng.random() < edge_frac:
            e = int(rng.integers(-64, 65))
            ts = e if rng.random() < 0.5 else tp.T - nf + e
        else:
            ts = int(rng.integers(-nf // 2, tp.T - nf // 2))
        vr, vc = np.nonzero(tf.valid)
        k = rng.integers(len(vr))
        w = window(tp, r, ts - tR, nt, int(vr[k]) - op // 2 - sR, ns, int(vc[k]) - op // 2 - sR, ns, rng)
        if np.isfinite(w["movie"][tR:tR + nf, sR:sR + op, sR:sR + op]).mean() < 0.05:
            continue
        out.append(cache_window(trunk, w, float(tf.scale), float(st["gain"]), device, tR, op, sR, nf, args.need_gb))
        if (i + 1) % 16 == 0:
            print(f"training window {i + 1}/{n}, {time.time() - t0:.0f} s", flush=True)
    return out


def evaluation_windows(args, tps, tfs, stats, cfg, trunk, device, first, n):
    """Cached trunk outputs of the evaluation draws first..first+n-1 (as edge_windows.py: a window in
    the trial's middle, whole and with the frames after or before its loss region removed)."""
    tR, sR = args.t_reach, args.s_reach
    nf, op = args.out_frames, args.out_px
    nt, ns = nf + 2 * tR, op + 2 * sR
    out, t0 = [], time.time()
    for i in range(first, first + n):
        rng = np.random.default_rng((90_000, i))
        j = i % len(tps)
        tp, tf = tps[j], tfs[j]
        st = stats[str(args.trials[j])]
        r = draw(tp, cfg, rng)
        pool = tf.support & tf.valid if rng.random() < 0.75 and (tf.support & tf.valid).any() else tf.valid
        vr, vc = np.nonzero(pool)
        k = rng.integers(len(vr))
        r0, c0 = int(vr[k]) - op // 2 - sR, int(vc[k]) - op // 2 - sR
        tm = int(rng.integers(tR, tp.T - nf - tR + 1))
        w = window(tp, r, tm - tR, nt, r0, ns, c0, ns, rng)
        d = {"mid": cache_window(trunk, w, float(tf.scale), float(st["gain"]), device, tR, op, sR, nf, args.need_gb)}
        for c, sl in (("end", slice(tR + nf, None)), ("start", slice(None, tR))):
            v = dict(w)
            v["movie"] = w["movie"].copy()
            v["movie"][sl] = np.nan
            d[c] = cache_window(trunk, v, float(tf.scale), float(st["gain"]), device, tR, op, sR, nf, args.need_gb)
        sup = w["support"][sR:sR + op, sR:sR + op]
        val = np.isfinite(w["mean_image"][sR:sR + op, sR:sR + op])
        lab = w["labeled"][sR:sR + op, sR:sR + op]
        d["masks"] = {"sup": sup & val, "bg": val & ~lab}
        out.append(d)
        if (i + 1 - first) % 8 == 0:
            print(f"evaluation window {i + 1 - first}/{n}, {time.time() - t0:.0f} s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--trials", nargs="+", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", default="fastpz")
    ap.add_argument("--pz_legacy", action="store_true")
    ap.add_argument("--stats", default="")
    ap.add_argument("--n_train", type=int, default=96)
    ap.add_argument("--n_eval", type=int, default=24)
    ap.add_argument("--edge_frac", type=float, default=0.5)
    ap.add_argument("--out_frames", type=int, default=128)
    ap.add_argument("--out_px", type=int, default=24)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--need_gb", type=float, default=2.5)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--arms", default="ctrl,edge")
    ap.add_argument("--save_cache", action="store_true", help="keep the trunk outputs in OUT_DIR/cache.pt (GBs)")
    ap.add_argument("--load_cache", default="",
                    help="start from this cache.pt (its training and evaluation windows; refused unless made with "
                         "the same checkpoint trunk, simulation and window sizes)")
    ap.add_argument("--extra_train", type=int, default=0,
                    help="more training windows on top of a loaded cache, all at trial edges (seeds 96_000+)")
    ap.add_argument("--extra_eval", type=int, default=0, help="more evaluation draws on top of a loaded cache")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tR, sR = ck["window"]["t_reach"], ck["window"]["s_reach"]
    nf, op = args.out_frames, args.out_px
    nt, ns = nf + 2 * tR, op + 2 * sR
    cfg = eval_sim(args.kind, pz_legacy=args.pz_legacy)
    stats = load_stats(args.trials, args.stats or None)
    tfs = [TrialFile(f) for f in args.trials]
    tps = [Template(tf, stats[str(f)]) for tf, f in zip(tfs, args.trials)]
    args.t_reach, args.s_reach = tR, sR
    if (args.extra_train or args.extra_eval) and not args.load_cache:
        raise SystemExit("--extra_train and --extra_eval extend a --load_cache")
    source = cache_source(args, ck, stats)
    trunk = None
    if args.extra_train or args.extra_eval or not args.load_cache:
        trunk = build(ck, True, device).eval()  # its parts hold the edge statistics too; ctrl ignores them
        if device.type == "cuda":
            trunk.half()
    if args.load_cache:  # only when asked, and only from the same trunk, simulation and window sizes
        saved = torch.load(args.load_cache, weights_only=False)
        had = saved.get("source", {}) if isinstance(saved, dict) else {}
        bad = [k for k in source if had.get(k) != source[k]]
        if bad:
            raise SystemExit(f"{args.load_cache} was not made with this checkpoint and these settings "
                             f"({', '.join(bad)} differ)")
        train, evals = saved["train"], saved["evals"]
        print(f"cache: {len(train)} training and {len(evals)} evaluation windows", flush=True)
        if args.extra_train:
            train += training_windows(args, tps, tfs, stats, cfg, trunk, device, args.extra_train, 96_000, 1.0)
        if args.extra_eval:
            evals += evaluation_windows(args, tps, tfs, stats, cfg, trunk, device, len(evals), args.extra_eval)
        print(f"{len(train)} training and {len(evals)} evaluation windows", flush=True)
    else:
        train, evals = [], []
        train += training_windows(args, tps, tfs, stats, cfg, trunk, device, args.n_train, 95_000, args.edge_frac)
        evals += evaluation_windows(args, tps, tfs, stats, cfg, trunk, device, 0, args.n_eval)
        if args.save_cache:  # several GB
            torch.save({"source": source, "train": train, "evals": evals}, out / "cache.pt")
    del trunk
    results = {}
    for arm in args.arms.split(","):
        model = build(ck, arm in ("edge", "gated"), device).float()
        if arm == args.arms.split(",")[0]:
            results["start"] = evaluate(model, evals, device)
            show("checkpoint", results["start"])
        t0 = time.time()
        print(f"refit {arm}", flush=True)
        refit(model, train, device, args.steps, args.lr * (10 if arm == "gated" else 1), args.batch, seed=1,
              only=("head_edge",) if arm == "gated" else HEAD)
        results[arm] = evaluate(model, evals, device)
        show(f"refit {arm} ({time.time() - t0:.0f} s)", results[arm])
        # as training saves them (predict.load_checkpoint, edge_windows.py and --ckpt here read "window")
        torch.save({"model": model.state_dict(), "config": copy.deepcopy(vars(model.cfg)), "window": ck["window"],
                    "step": ck.get("step"), "refit": {"arm": arm, "of": str(args.ckpt), "steps": args.steps}},
                   out / f"{arm}.pt")
    (out / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
