"""Baseline bias of checkpoints per pixel class on simulated (or real) windows, with
switches that isolate the low-photon-count bias (docs/plan.md, "Low photon counts:
the baseline sits low").

usage: python lowcount_probe.py --ckpt NAME=PATH ... --sessions SESSION_DIR ...
           [--variants free,freepz,freex4,freeflat1vf1] [--windows 32]
           [--out_frames 64 --out_px 24] [--out probe.csv]

Every checkpoint is scored on the same windows (seeded as `evaluate` seeds its
kinds). Classes: footprint support, unlabeled background, labeled outside every
footprint. Per class: 8-frame MSE relative to the mean image's (ratio), bias
(target - net, scaled units), the mean image's bias, the bias as a share of the
class's mean target (relbias) and that mean.

Variants (combine the parts in one name, e.g. freepzx4):
  real, sim, free (no activity)  the data kind
  pz / pznew        realistic low counts, legacy draw (as S0-S3 trained) / current draw
  dim               a third of the photons (SimConfig.photon_gain 3, as evaluate's dim kinds)
  x<k>              k times the photons: gain / k, measured skew / sqrt(k) (the simulator's
                    event-size factor stays), variance maps / k, and with pz the counts
                    per zero-count event / k (as SimConfig.photon_gain 1/k)
  flat<level>       the template's baseline replaced by a flat field at level x the trial
                    scale, with no excess (flatm<level>: the template's own baseline, no excess)
  vf1               the variance-factor input set to 1 (no information)
  noexc, noslow     no excess / no slow excess

A smaller loss region (--out_frames, --out_px) gives the same outputs per pixel
(the model's reach is fixed) at a fraction of the cost. cuDNN autotuning stays
off: on a shared GPU it stalled for minutes.
"""
import argparse
import csv
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from slap2_f0.data import WindowConfig, WindowDataset, loss_region  # noqa: E402
from slap2_f0.io import list_trials  # noqa: E402
from slap2_f0.model import SlabBlindSpotNet, config_from_dict  # noqa: E402
from slap2_f0.simulate import SimConfig, Template  # noqa: E402
from slap2_f0.stats import load_stats  # noqa: E402
from slap2_f0.train import EVAL_SEEDS, _forward, movmean_t  # noqa: E402

CLASSES = ("sup", "bg", "lab")


class ProbeDataset(WindowDataset):
    """WindowDataset with altered templates. level (None: unchanged): a flat baseline
    at level x the trial scale (level <= 0: the template's own baseline), and no
    excess. k: k times the photons, where the stats come from `more_photons`; the
    counts per zero-count event, which Template derives from the unchanged mean and
    zero-frame maps, are divided by k here. Activity as the SimConfig says."""

    def __init__(self, *args, level=None, k=1.0, **kw):
        super().__init__(*args, **kw)
        self.level, self.k = level, k

    def _template(self, fn):
        if fn not in self._tpl:
            tp = Template(self._trial(fn), self.stats[fn])
            if self.level is not None:
                if self.level > 0:
                    tp.B = np.where(np.isfinite(tp.B), np.float32(self.level * tp.tf.scale), np.nan).astype(np.float32)
                tp.fast = np.zeros_like(tp.fast)
                tp.slow = np.zeros_like(tp.slow)
            if self.k != 1 and tp.zc is not None:
                tp.zc = (tp.zc / np.float32(self.k)).astype(np.float32)
            self._tpl[fn] = tp
        return self._tpl[fn]


def sim_config(v: str) -> SimConfig | None:
    if v.startswith("real"):
        return None
    cfg = SimConfig(silent_trial=1.0) if v.startswith("free") else SimConfig()
    if "pz" in v:
        cfg = replace(cfg, photon_zeros=True, pz_legacy="pznew" not in v)
    if "noexc" in v:
        cfg = replace(cfg, excess=0.0)
    if "noslow" in v:
        cfg = replace(cfg, n_slow=0)
    if "dim" in v:
        cfg = replace(cfg, photon_gain=(3.0, 3.0))
    return cfg


def more_photons(stats: dict, k: float) -> dict:
    """k times the photons: gain / k; skew / sqrt(k) keeps Template.q; variance maps / k."""
    return {f: dict(s, gain=s["gain"] / k, skew=s["skew"] / np.sqrt(k),
                    **{n: s[n] / k for n in ("var_res", "var8_res", "var64_res", "var_movie") if n in s})
            for f, s in stats.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", nargs="+", required=True, help="NAME=PATH")
    p.add_argument("--sessions", nargs="+", required=True, help="session folders of exported trials")
    p.add_argument("--variants", default="free,freepz")
    p.add_argument("--windows", type=int, default=32)
    p.add_argument("--out", default="")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_frames", type=int, default=0, help="loss-region frames (default: the checkpoint's)")
    p.add_argument("--out_px", type=int, default=0, help="loss-region pixels per side (default: the checkpoint's)")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    models = []
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location=dev, weights_only=False)
        m = SlabBlindSpotNet(config_from_dict(ck["config"])).to(dev).eval()
        m.load_state_dict(ck["model"])
        models.append((name, m, WindowConfig(**ck["window"])))
    wcfg = models[0][2]
    if a.out_frames:
        wcfg = replace(wcfg, out_frames=a.out_frames)
    if a.out_px:
        wcfg = replace(wcfg, out_px=a.out_px)
    region = loss_region(wcfg)
    sel = (slice(None),) + region
    amp = torch.bfloat16 if dev.type == "cuda" else None
    rows = []
    for sdir in a.sessions:
        files = list_trials(sdir)
        stats = load_stats(files, None)
        sess = Path(sdir).name
        for v in a.variants.split(","):
            base = "real" if v.startswith("real") else ("free" if v.startswith("free") else "sim")
            mx = re.search(r"x(\d+)", v)
            k = float(mx.group(1)) if mx else 1.0
            st_v = more_photons(stats, k) if mx else stats
            mf = re.search(r"flat(m?)([0-9.]+)", v)
            level = None if not mf else (-1.0 if mf.group(1) else float(mf.group(2)))
            ds = ProbeDataset(files, wcfg, st_v, level=level, k=k, length=a.windows,
                              seed=EVAL_SEEDS[base] + a.seed, sim=sim_config(v), flip=False)
            tot = {n: {} for n, _, _ in models}
            t0 = time.time()
            for b in torch.utils.data.DataLoader(ds, batch_size=1, num_workers=a.workers):
                if "vf1" in v:
                    b["varfac"] = torch.ones_like(b["varfac"])
                for name, model, _ in models:
                    with torch.no_grad():
                        bb, mu, _ = _forward(model, b, dev, amp, region)
                    y, w = bb["target"][sel], bb["weight"][sel]
                    s = bb["scale"].view(-1, 1, 1)
                    mi = torch.nan_to_num((bb["mean_image"] / s)[(slice(None),) + region[1:]]).unsqueeze(1).expand_as(y)
                    sup = bb["support"][(slice(None),) + region[1:]].bool().unsqueeze(1)
                    lab = bb["labeled"][(slice(None),) + region[1:]].bool().unsqueeze(1)
                    y8, n8 = movmean_t(y, w)
                    w8 = w * (n8 >= 4)
                    T = tot[name]
                    for c, cw in {"sup": w * sup, "bg": w * ~lab, "lab": w * (lab & ~sup)}.items():
                        for pn, pr in (("net", mu), ("mi", mi)):
                            p8, _ = movmean_t(pr, w)
                            T[f"{c}_se8_{pn}"] = T.get(f"{c}_se8_{pn}", 0) + (cw * w8 * (y8 - p8) ** 2).sum().item()
                            T[f"{c}_d_{pn}"] = T.get(f"{c}_d_{pn}", 0) + (cw * (y - pr)).sum().item()
                        T[f"{c}_y"] = T.get(f"{c}_y", 0) + (cw * y).sum().item()
                        T[f"{c}_n"] = T.get(f"{c}_n", 0) + cw.sum().item()
            for name, T in tot.items():
                r = {"ckpt": name, "session": sess, "variant": v, "windows": a.windows}
                for c in CLASSES:
                    n = max(T.get(f"{c}_n", 0), 1)
                    r[f"{c}_ratio"] = T.get(f"{c}_se8_net", np.nan) / max(T.get(f"{c}_se8_mi", 0), 1e-12)
                    r[f"{c}_bias"] = T.get(f"{c}_d_net", 0) / n
                    r[f"{c}_bias_mi"] = T.get(f"{c}_d_mi", 0) / n
                    r[f"{c}_relbias"] = T.get(f"{c}_d_net", 0) / max(T.get(f"{c}_y", 0), 1e-9)
                    r[f"{c}_meany"] = T.get(f"{c}_y", 0) / n
                rows.append(r)
                print(f"{name:10s} {sess[:6]} {v:14s} " + "  ".join(
                    f"{c}: ratio {r[c + '_ratio']:.2f} bias {100 * r[c + '_relbias']:+.1f}%" for c in CLASSES)
                    + f"  ({time.time() - t0:.0f} s)", flush=True)
            if a.out:
                with open(a.out, "w", newline="") as fh:
                    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    wr.writeheader()
                    wr.writerows(rows)


if __name__ == "__main__":
    main()
