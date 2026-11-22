"""Training loop for the blind-spot baseline network.

Three kinds of data (``--data``):
    sim   simulated windows built on real trials as templates, with the true,
          noise-free baseline as the target (``simulate``): pretraining
    real  real windows, with movie - activity as the (noisy) target:
          fine-tuning, usually from a simulation-pretrained checkpoint
          (``--init``)
    mix   each window simulated with probability ``--sim_frac``, else real
``--silent_trial`` and ``--silent_frac`` make simulated windows partly or
wholly activity-free (``SimConfig``); ``--subtract_frac`` feeds real windows
with movie - activity as the input (``data``); ``--exclude`` keeps sessions (e.g. a
GFP control kept for testing) out of training and validation.

Loss (``_losses``): squared error of the baseline weighted by 1 / photon-
noise variance, which is minimized by the mean (unbiased at any brightness),
plus ``w_var`` times a Gaussian NLL for the noise head around the detached
baseline.

Files are split into training and validation by session when there are at
least three sessions (so validation measures generalization to unseen
sessions), otherwise by trial. Every validation evaluates fixed windows from
the held-out files, both real (``real_*``) and simulated (``sim_*``, and
with ``--val_free`` ``free_*``: simulated without any activity, where every
bias is spurious), and from the training files (``tr_*``); see ``evaluate``
for the metrics.

best.pt is chosen on held-out windows of the training data kind (``--select``,
default ``balanced``): the network's 8-frame MSE relative to the mean image's,
averaged over footprint-support and background pixels. The per-frame loss
can't pick checkpoints on real data: photon noise buries the 1-3% that can be
learned (v2 run). The untrained network (step 0, which predicts the mean
image, score 1) is a candidate, so best.pt is never worse than the start.

Validation sessions can be named (``--val_sessions``, and ``--train_sessions``
for the training ones), and the training data capped per session and path
(``--max_train_s``), for data-scaling experiments.

Outputs in --out: config.json, metrics.csv (one row per validation),
last.pt, best.pt, steps/step_<N>.pt with ``--keep_every`` (model only), and
stats/ (cached whole-trial statistics).

Fine-tuning regularizers (docs/plan.md, "Fine-tuning trade-off", 2026-11-21):
``--ema`` validates, selects and saves an exponential moving average of the
weights ("model"; the raw weights go to "model_raw"), which averages out the
checkpoint-to-checkpoint noise; ``--l2sp`` pulls the weights back toward the
``--init`` checkpoint after every step (decoupled, like AdamW's weight decay).
"""

from __future__ import annotations

import copy
import csv
import itertools
import json
import math
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import WindowConfig, WindowDataset, loss_region
from .io import TrialFile, filter_trials, list_trials
from .losses import gaussian_nll, student_t_nll
from .model import ModelConfig, SlabBlindSpotNet, blind_window_stats, config_from_dict, make_inputs
from .simulate import SimConfig, excess_summary
from .stats import load_stats


class LoaderWait:
    """Time the training loop spends waiting for the data loaders, per log
    interval. ``idle()`` when the loop goes back for the next batch, ``got()``
    when it has it; ``report()`` gives (wait, interval) since the last report
    and starts a new interval. Work between ``got()`` and ``idle()``
    (forward, backward, validation) counts towards the interval only."""

    def __init__(self, clock=time.time):
        self.clock = clock
        self.wait = 0.0
        self.t_log = self.tick = clock()

    def idle(self) -> None:
        self.tick = self.clock()

    def got(self) -> None:
        self.wait += self.clock() - self.tick

    def report(self) -> tuple[float, float]:
        now = self.clock()
        out = (self.wait, now - self.t_log)
        self.wait, self.t_log = 0.0, now
        return out


class StepProfiler:
    """--profile_steps N: profile N optimizer steps (torch.profiler, CPU and
    CUDA) and print the operations with the most self time on the GPU and on
    the CPU, the GPU's busy share of the wall time (the union of its kernels'
    intervals), and torch.compile's counters (frames compiled, graph breaks,
    recompilations). For finding what limits a training step.

    The window opens after step ``start`` has finished (its validation and
    checkpoints included) and closes after step ``start + n``'s optimizer step,
    before that step's validation; ``quiet_start`` picks a start whose window
    holds no validation or checkpoint. ``finish`` closes a window that training
    left open (it ended early or raised) and says so."""

    def __init__(self, n: int, start: int = 20, device: torch.device | str = "cuda"):
        if n < 0:
            raise ValueError(f"--profile_steps must be 0 (off) or positive, not {n}")
        self.n, self.start, self.prof, self.t = n, start, None, 0.0
        self.device = torch.device(device)  # the training device: the one to wait for
        self.acts = [torch.profiler.ProfilerActivity.CPU] + (
            [torch.profiler.ProfilerActivity.CUDA] if self.device.type == "cuda" else [])

    @staticmethod
    def quiet_start(n: int, everies, first: int = 20) -> int:
        """The first start >= first whose window (start, start + n] holds no
        multiple of any of ``everies`` (validation, saving, keeping; 0 = never)."""
        ev = [e for e in everies if e and e > 0]
        if n and ev and n >= min(ev):  # every n consecutive steps would hold a validation or checkpoint
            raise ValueError(f"--profile_steps {n} must be smaller than every enabled interval "
                             f"(--val_every, --save_every, --keep_every: {sorted(ev)})")
        s = first
        while any(k % e == 0 for e in ev for k in range(s + 1, s + n + 1)):
            s += 1
        return s

    def after_step(self, step: int) -> None:
        """At the end of an optimizer step's iteration (after validation and checkpoints)."""
        if self.n and self.prof is None and step == self.start:
            self.prof = torch.profiler.profile(activities=self.acts)
            self.prof.__enter__()
            self.t = time.time()

    def before_val(self, step: int) -> None:
        """Right after an optimizer step, before its validation and checkpoints."""
        if self.prof is not None and step == self.start + self.n:
            self._close(self.n)

    def finish(self, step: int) -> None:
        if self.prof is not None:
            self._close(max(0, step - self.start), incomplete=True)

    def _close(self, steps: int, incomplete: bool = False) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        wall = time.time() - self.t
        prof, self.prof = self.prof, None
        prof.__exit__(None, None, None)
        spans = sorted((e.time_range.start, e.time_range.end) for e in prof.events()
                       if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA)
        busy, end = 0.0, float("-inf")  # union of kernel intervals (us)
        for a0, a1 in spans:
            if a1 > end:
                busy += a1 - max(a0, end)
                end = a1
        note = f" (incomplete: {steps} of {self.n} steps)" if incomplete else ""
        print(f"profile of {steps} steps{note}: {wall:.1f} s wall, GPU busy {busy / 1e6:.1f} s "
              f"({100 * busy / 1e6 / max(wall, 1e-9):.0f}%)", flush=True)
        ev = prof.key_averages()
        for key in ("self_cuda_time_total", "self_cpu_time_total"):
            try:
                print(ev.table(sort_by=key, row_limit=20, max_name_column_width=70), flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"profile table by {key}: {e}", flush=True)
        try:
            from torch._dynamo.utils import counters
            print("torch.compile counters: " + json.dumps({k: dict(v) for k, v in counters.items()},
                                                           default=str)[:4000], flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"torch.compile counters: {e}", flush=True)


def quality_files(files: list[Path], args) -> list[Path]:
    """``io.filter_trials`` with the run's thresholds, reporting what it left out."""
    kept, report = filter_trials(files, getattr(args, "min_peak_r", 0.3), getattr(args, "min_path_trials", 4))
    if report:
        print(f"trial quality: left out {len(files) - len(kept)} of {len(files)} trials:\n  " + "\n  ".join(report),
              flush=True)
    return kept


def session_groups(files: list[Path]) -> dict[str, list[Path]]:
    """Trial files by session label, in the order given."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        with TrialFile(f) as tr:
            groups[str(tr.attrs.get("label", f.parent.parent.name))].append(f)
    return dict(groups)


def split_files(files: list[Path], val_fraction: float, seed: int = 0) -> tuple[list[Path], list[Path], str]:
    """Hold out whole sessions if there are at least 3, else whole trials."""
    groups = session_groups(files)
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


def explicit_split(groups: dict[str, list[Path]], val_sessions: str, train_sessions: str = ""
                   ) -> tuple[list[Path], list[Path], str]:
    """Training and validation files from named sessions (comma-separated):
    validation = val_sessions; training = train_sessions, or all other sessions."""
    val_names = [s for s in val_sessions.split(",") if s]
    train_names = [s for s in train_sessions.split(",") if s] or [s for s in groups if s not in val_names]
    missing = [s for s in val_names + train_names if s not in groups]
    if missing:
        raise SystemExit(f"sessions not found: {', '.join(missing)} (have: {', '.join(sorted(groups))})")
    both = set(val_names) & set(train_names)
    if both:
        raise SystemExit(f"sessions in both training and validation: {', '.join(sorted(both))}")
    val = [f for s in val_names for f in groups[s]]
    train = [f for s in train_names for f in groups[s]]
    return train, val, f"by session (validation: {', '.join(val_names) or 'none'}; training: {', '.join(train_names)})"


def cap_seconds(files: list[Path], max_s: float, fps: float = 80.0) -> list[Path]:
    """The first trials of each session and path (in file order) up to max_s
    seconds of movie: a trial is kept if the ones before it make less than max_s
    (so the total exceeds max_s by less than one trial). 0 keeps all."""
    if not max_s:
        return list(files)
    used: dict[tuple[str, str], float] = defaultdict(float)
    out = []
    for f in files:
        with TrialFile(f) as tr:
            key = (str(tr.attrs.get("label", f.parent.parent.name)), f.parent.name)
            n = tr.n_frames
        if used[key] < max_s:
            out.append(f)
            used[key] += n / fps
    return out


def seconds_by_session(files: list[Path], fps: float = 80.0) -> dict[str, float]:
    """Movie seconds per session, summed over paths."""
    tot: dict[str, float] = defaultdict(float)
    for f in files:
        with TrialFile(f) as tr:
            tot[str(tr.attrs.get("label", f.parent.parent.name))] += tr.n_frames / fps
    return dict(tot)


def without_noise_stats(files: list[Path], stats: dict) -> list[Path]:
    """Files whose noise statistics are not finite. They come from the
    unlabeled valid pixels; a trial whose labeled mask covers every valid pixel
    (seen once in phase 1, a dim trial) has none, and simulating from it fails."""
    keys = ("gain", "rho_t1", "rho_t2", "rho_r", "rho_c")
    bad = lambda s: not all(np.isfinite(float(s[k])) for k in keys) or float(s["gain"]) <= 0  # noqa: E731
    return [f for f in files if bad(stats[str(f)])]


def _drop_files(files: list[Path], stats: dict) -> set[str]:
    """Print and return the files without finite noise statistics."""
    bad = without_noise_stats(files, stats)
    if bad:
        print(f"left out {len(bad)} trial file(s) without finite noise statistics (no unlabeled valid pixels): "
              + ", ".join(f"{f.parent.parent.name}/{f.parent.name}/{f.name}" for f in bad), flush=True)
    return {str(f) for f in bad}


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
    x = make_inputs(b["movie"], b["mean_image"], b["varfac"], b["scale"], b["var_map"], b["gain"],
                    varfac_norm=getattr(model, "_orig_mod", model).cfg.varfac_norm)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        out = model(x, b["scale"], b["gain"], region, aux=aux)
    if aux:
        return b, out[0].float(), out[1].float(), out[2]
    return b, out[0].float(), out[1].float()


def fast_model(model: torch.nn.Module, device: torch.device, args) -> torch.nn.Module:
    """The model to call for forward passes on the GPU: cuDNN autotuning (the
    window shape is fixed, so algorithms are timed once per shape; on the L40S
    its default choice made a training step 1.6x slower) and torch.compile
    (1.7x faster training steps and forward passes on the A10G and the L40S;
    docs/plan.md, "GPU machine"). Saving, the optimizer and the config use the
    plain model. Without Triton (e.g. on Windows), or with --no_compile, the
    plain model; compile errors fall back to it."""
    if device.type != "cuda":
        return model
    torch.backends.cudnn.benchmark = not getattr(args, "no_cudnn_benchmark", False)
    if getattr(args, "no_compile", False):
        return model
    import importlib.util
    if importlib.util.find_spec("triton") is None:
        print("torch.compile skipped: Triton is not installed", flush=True)
        return model
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True  # fall back to eager on a compile error
    return torch.compile(model)


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
def evaluate(model, loader, region, device, amp_dtype, args, per_window: list | None = None,
             tot: dict | None = None, finish: bool = True) -> dict[str, float]:
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
    split into footprint support, unlabeled background and labeled pixels
    outside every footprint (lab_; these have loss weight only with
    w_unfitted > 0 or in a trial without sources, such as a GFP control). For
    real targets
    the photon noise adds the same amount to every MSE; for simulated ones the
    MSE is the baseline error itself. bias_* = mean(target - baseline);
    mean_target per class, so bias / mean_target is the bias as a share of the
    baseline (activity-free simulations of dim sessions: 5-10% at every class).
    sup_leak_*: the slope of (baseline - target) on the activity at support
    pixels, 8-frame means: how much of the activity the baseline follows (0 =
    none; for real windows, whose target still holds the activity GIAnT did
    not fit, this is the leak minus that share).
    Calibration: z = (movie - activity - mu) / sqrt(var) should have mean 0 and
    variance 1; r8 is the variance of 8-frame averages of the residual over
    the mean predicted per-frame variance (the factor that converts the
    network's sigma to SiLo's).

    per_window, if a list, gets one dict per window with its 8-frame squared-
    error sums and weights at support and background pixels (sup_n8,
    sup_se8_<baseline>, bg_n8, bg_se8_<baseline> for net, mean_image, silo
    and oracle), so scores can be bootstrapped over windows.

    loader may be any iterable of batches. With tot (a defaultdict(float)) and
    finish=False the sums accumulate into tot and tot is returned, so windows
    can be scored in chunks; evaluate_totals(tot) turns them into the scores."""
    model.eval()
    sel = (slice(None),) + region
    tot = defaultdict(float) if tot is None else tot
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
        cfg = getattr(model, "_orig_mod", model).cfg
        ob, _, ob_n = blind_window_stats(b["noisy"] * wv, wv, cfg.gap, cfg.window_reach)
        ob = torch.where(ob_n[sel] > 0, ob[sel], oracle)
        preds = {"net": m, "mean_image": torch.nan_to_num(mi), "window": aux["window_mean"],
                 "silo": torch.nan_to_num(silo), "oracle": oracle, "oracle_blind": ob}
        sup = b["support"][(slice(None),) + region[1:]].bool().unsqueeze(1)
        bg = ~b["labeled"][(slice(None),) + region[1:]].bool().unsqueeze(1)
        classes = {"": w, "sup_": w * sup, "bg_": w * bg, "lab_": w * (~bg & ~sup)}
        y8, n8 = movmean_t(y, w)
        w8 = w * (n8 >= 4)
        act = torch.nan_to_num(b["movie"][sel] / s.unsqueeze(1) - yn)  # movie - (movie - activity)
        a8, _ = movmean_t(act, w)
        sw8 = (w * sup * w8).float()
        if per_window is not None:
            pw = [dict() for _ in range(y.shape[0])]
            for cname in ("sup_", "bg_"):
                cw8 = (classes[cname] * w8).float()
                for i, val in enumerate(cw8.sum((1, 2, 3)).tolist()):
                    pw[i][f"{cname}n8"] = val
                for pname in ("net", "mean_image", "silo", "oracle"):
                    p8, _ = movmean_t(preds[pname], w)
                    for i, val in enumerate((cw8 * (y8 - p8) ** 2).sum((1, 2, 3)).tolist()):
                        pw[i][f"{cname}se8_{pname}"] = val
            per_window.extend(pw)
        for cname, cw in classes.items():
            n = cw.sum().item()
            if n == 0:
                continue
            tot[cname + "n"] += n
            tot[cname + "target"] += (cw * y).sum().item()  # for bias relative to the class's baseline
            cw8 = cw * w8
            tot[cname + "n8"] += cw8.sum().item()
            for pname, p in preds.items():
                d = y - p
                tot[f"{cname}mse_{pname}"] += (cw * d ** 2).sum().item()
                p8, _ = movmean_t(p, w)
                tot[f"{cname}mse8_{pname}"] += (cw8 * (y8 - p8) ** 2).sum().item()
                tot[f"{cname}bias_{pname}"] += (cw * d).sum().item()
                if cname == "sup_":  # leak: regression of (baseline - target) on the activity
                    e8 = p8 - y8
                    for k, val in (("a", a8), ("aa", a8 * a8), ("e", e8), ("ae", a8 * e8)):
                        tot[f"leak_{k}_{pname}"] += (sw8 * val).sum().item()
                    tot[f"leak_n_{pname}"] += sw8.sum().item()
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
    return evaluate_totals(tot) if finish else tot


def evaluate_totals(tot: dict) -> dict[str, float]:
    """Scores from evaluate's accumulated sums."""
    n = max(tot["n"], 1.0)
    out = {"loss": tot["loss"] / n, "var_loss": tot["var_loss"] / n}
    zm = tot["z"] / n
    out.update(z_mean=zm, z_var=tot["z2"] / n - zm ** 2, within_2=tot["in2"] / n,
               r8=tot["r8_num"] / max(tot["r8_den"], 1e-9), mix=tot["mix"] / n)
    for pname in [k[len("leak_n_"):] for k in tot if k.startswith("leak_n_")]:
        nn_ = tot[f"leak_n_{pname}"]
        if nn_ > 0:
            ma, me = tot[f"leak_a_{pname}"] / nn_, tot[f"leak_e_{pname}"] / nn_
            va = tot[f"leak_aa_{pname}"] / nn_ - ma ** 2
            out[f"sup_leak_{pname}"] = (tot[f"leak_ae_{pname}"] / nn_ - ma * me) / va if va > 1e-12 else math.nan
    for k, v in tot.items():
        if "_mse" in k or k.startswith("mse") or "bias_" in k:
            cname = k.split("mse")[0] if "mse" in k else k.split("bias_")[0]
            denom = tot.get(cname + ("n8" if "mse8_" in k else "n"), 0.0)
            out[k] = v / max(denom, 1.0)
    for cname in ("", "sup_", "bg_", "lab_"):  # mean target: bias_* / mean_target is the bias as a share of it
        if tot.get(cname + "n", 0.0) > 0:
            out[f"{cname}mean_target"] = tot[cname + "target"] / tot[cname + "n"]
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
    if how == "balanced_free":  # real support and background, and labeled pixels without activity
        a, b = row.get("free_lab_mse8_net", math.nan), row.get("free_lab_mse8_mean_image", math.nan)
        r = [x for x in (ratio("sup_"), ratio("bg_"), a / b if b > 0 else math.nan) if math.isfinite(x)]
        if r:
            return float(np.mean(r))
    r = ratio("")
    return r if math.isfinite(r) else math.inf


def parse_select_on(spec: str) -> dict[str, float]:
    """--select_on "real" or "real,fast" or "real:1,fast:0.5": validation sets
    and their weights in the checkpoint score."""
    out = {}
    for part in spec.split(","):
        name, _, w = part.strip().partition(":")
        if name:
            out[name] = float(w) if w else 1.0
    return out


def combined_selection(row: dict, per_group: dict, kinds: dict, how: str) -> float:
    """Checkpoint score (lower is better): the weighted mean over the selection
    sets of each set's score. For balanced, a set's score is the mean over
    validation mice of each mouse's balanced ratio (one pooled group: the
    pooled ratio, as before); for the other scores, selection_score on the
    group-averaged row."""
    tot = wsum = 0.0
    for name, w in kinds.items():
        if how == "balanced" and per_group.get(name):
            vals = [v for v in per_group[name].values() if math.isfinite(v)]
            v = float(np.mean(vals)) if vals else math.inf
        else:
            v = selection_score(row, name, how)
        tot, wsum = tot + w * v, wsum + w
    return tot / wsum if wsum > 0 else math.inf


def _model_config(args) -> ModelConfig:
    mcfg = ModelConfig(gap=args.gap, width=args.width, spatial_kernel=args.spatial_kernel,
                       downsample=args.downsample, full_res_blocks=args.full_res_blocks,
                       checkpoint=not args.no_checkpoint, window_reach=args.window_reach,
                       max_log_correction=args.max_log_correction)
    if args.temporal_dilations:
        mcfg.temporal_dilations = [int(d) for d in args.temporal_dilations.split(",")]
    if args.spatial_dilations:
        mcfg.spatial_dilations = [int(d) for d in args.spatial_dilations.split(",")]
    mcfg.box_radius, mcfg.hp_width, mcfg.hp_full_res_blocks = args.box_radius, args.hp_width, args.hp_full_res_blocks
    mcfg.varfac_norm = args.varfac_norm
    mcfg.hp_shared = args.hp_shared
    mcfg.motion_features = getattr(args, "motion_features", False)
    mcfg.edge_features = getattr(args, "edge_features", False)
    mcfg.branch_mlp = getattr(args, "branch_mlp", 0)
    mcfg.fuse_blocks = getattr(args, "fuse_blocks", 0)
    if args.hp_temporal_dilations:
        mcfg.hp_temporal_dilations = [int(d) for d in args.hp_temporal_dilations.split(",")]
    if args.hp_spatial_dilations:
        mcfg.hp_spatial_dilations = [int(d) for d in args.hp_spatial_dilations.split(",")]
    if args.pool_scales:
        mcfg.pool_scales = [int(d) for d in args.pool_scales.split(",")]
    return mcfg


def adapt_varfac_input(state: dict, width: int) -> dict:
    """Fold a "window" checkpoint's variance-factor weights (``ModelConfig.
    varfac_norm``) so that a "trial" model starts out computing what it did.

    In most training windows the old input was saturated: +3 at imaged pixels
    and -3 at unimaged ones. The layers that take it are pointwise (the
    branches' input convolutions and the first fusion layer), so its weight w
    moves to the validity flag (6 w, with -3 w in the bias) in the branches,
    and to the bias (3 w; outputs are at imaged pixels) in the fusion layer,
    and the variance-factor weight starts at 0. Exact except on discarded
    frames and in windows that were not saturated."""
    state = dict(state)
    for k in [k for k in state if k.endswith(".inp.weight")]:
        wt, b = state[k].clone(), state[k[:-6] + "bias"].clone()
        w3 = wt[:, 3].clone()
        wt[:, 1] += 6 * w3
        b -= 3 * w3.flatten(1).sum(1)
        wt[:, 3] = 0
        state[k], state[k[:-6] + "bias"] = wt, b
    col = 2 * width + 5  # [past, future, features]: the variance factor is the last feature
    wt, b = state["fuse.0.weight"].clone(), state["fuse.0.bias"].clone()
    b += 3 * wt[:, col].flatten(1).sum(1)
    wt[:, col] = 0
    state["fuse.0.weight"], state["fuse.0.bias"] = wt, b
    return state


def fusion_columns(src: ModelConfig, dst: ModelConfig) -> list[tuple[slice, slice]]:
    """(source columns, target columns) of the first fusion layer for each input
    part (``ModelConfig.fusion_inputs``) the source has; the target must have
    each with the same size."""
    def offsets(cfg):  # name -> (first column, count)
        out, at = {}, 0
        for name, n in cfg.fusion_inputs():
            out[name], at = (at, n), at + n
        return out

    s, d = offsets(src), offsets(dst)
    pairs = []
    for name, (a, n) in s.items():
        if not n:
            continue
        b, m = d[name]
        if m != n:
            raise SystemExit(f"checkpoint fusion input {name} ({n} channels) does not fit the model's ({m})")
        pairs.append((slice(a, a + n), slice(b, b + n)))
    return pairs


def load_compatible(model: torch.nn.Module, state: dict, src_config: dict | ModelConfig) -> str:
    """Load a checkpoint (its weights ``state`` and config ``src_config``) into
    a model that may have more parts (e.g. a slab model's weights into a box
    model): matching tensors are copied; a tensor whose input-channel axis grew
    gets the old columns and zeros for the new ones, so the model starts out
    computing what the old one did. The first fusion layer's columns are
    matched by input part (``ModelConfig.fusion_inputs``), whether it grew or
    not: a new part may sit before an old one (motion or box features added to
    a model with edge features), and a checkpoint part the model lacks is
    refused even when the widths agree. Returns a summary."""
    own = model.state_dict()
    copied, grown, new = 0, [], []
    state = dict(state)
    if not isinstance(src_config, ModelConfig):
        src_config = config_from_dict(src_config)
    if any(k.startswith("hp_views.") for k in own) and any(k.startswith("hp.") for k in state):
        for k in own:  # a shared half-plane trunk (hp.*) starts every per-direction trunk (hp_views.<k>.*)
            src = "hp." + k.split(".", 2)[2] if k.startswith("hp_views.") else None
            if src in state and k not in state:
                state[k] = state[src]
        state = {k: v for k, v in state.items() if not k.startswith("hp.")}
    for k, v in own.items():
        if k not in state:
            new.append(k)
            continue
        w = state[k]
        fits = w.dim() == v.dim() >= 2 and w.shape[0] == v.shape[0] and w.shape[2:] == v.shape[2:]
        if k == "fuse.0.weight" and fits:  # by input part, also at equal width
            pairs = fusion_columns(src_config, model.cfg)
            if sum(s.stop - s.start for s, _ in pairs) != w.shape[1]:
                raise SystemExit(f"checkpoint tensor {k} {tuple(w.shape)} does not match its config")
            nv = torch.zeros_like(v)
            for s, d in pairs:
                nv[:, d] = w[:, s]
            own[k] = nv
            if w.shape == v.shape:
                copied += 1
            else:
                grown.append(k)
        elif w.shape == v.shape:
            own[k] = w
            copied += 1
        elif fits and w.shape[1] < v.shape[1]:
            nv = torch.zeros_like(v)
            nv[:, : w.shape[1]] = w
            own[k] = nv
            grown.append(k)
        else:
            raise SystemExit(f"checkpoint tensor {k} {tuple(w.shape)} does not fit the model's {tuple(v.shape)}")
    unused = [k for k in state if k not in own]
    if unused:
        raise SystemExit(f"checkpoint tensors the model does not have: {unused[:5]}")
    model.load_state_dict(own)
    return f"{copied} tensors copied" + (f", {len(grown)} widened ({', '.join(grown)})" if grown else "") +         (f", {len(new)} new" if new else "")


def stats_cache(args, out: Path) -> Path:
    """The stats cache folder (--stats_cache, default <out>/stats), first
    seeded with the cached stats in --stats_from (e.g. an earlier run's stats/
    on a read-only mount)."""
    cache = Path(args.stats_cache) if args.stats_cache else out / "stats"
    if args.stats_from:
        cache.mkdir(parents=True, exist_ok=True)
        for p in Path(args.stats_from).glob("*.npz"):
            if not (cache / p.name).exists():
                shutil.copy2(p, cache / p.name)
    return cache


# simulated windows with fast shared brightness changes, sized like the real
# trials' (evaluate --kinds fast): field-wide and regional, 1.5-20 frames
FAST_GLOBAL_SD, FAST_FIELD_SD = 0.06, 0.04  # real: field-wide 2-5%, per 48-px region 2-4% per frame
FAST_SIM = dict(fast_global_sd=FAST_GLOBAL_SD, fast_field_sd=FAST_FIELD_SD)
# residual motion, focus and local motion at the sizes measured on the viewer trials
# (plan.md, "Residual motion and blur"): rigid 0.05-0.17 px per axis, mostly fast;
# blur 0.10-0.32 px^2 per axis; local 0.07-0.22 px beyond the simulator's level
MOTION_SIM = dict(motion_sd_px=(0.0, 0.2), motion_tau_frames=(0.5, 40.0), focus_sd_px2=(0.0, 0.25),
                  local_sd_px=(0.0, 0.2))


def eval_sim(kind: str, photon_zeros: bool = False, pz_legacy: bool = False) -> SimConfig | None:
    """The simulation of an evaluation kind (real: None). A kind may end in
    "pz" (realistic low counts), "shuf" (shuffled activity timing, no
    stimulus rhythm) and/or "dim" (a third of the photons: the template's gain
    times 3, SimConfig.photon_gain), e.g. "fastpz", "fastpzshuf", "freedim";
    photon_zeros adds "pz"; pz_legacy draws the low counts as up to 2026-11-21
    (SimConfig.pz_legacy)."""
    base = _base_kind(kind)
    cfg = {"sim": SimConfig(), "free": SimConfig(silent_trial=1.0), "fast": SimConfig(**FAST_SIM),
           "dark": SimConfig(**FAST_SIM, dark_frac=(0.3, 0.3)),
           "motion": SimConfig(**FAST_SIM, **MOTION_SIM)}.get(base)
    if cfg is not None:
        cfg.photon_zeros = photon_zeros or "pz" in kind[len(base):]
        cfg.pz_legacy = pz_legacy
        cfg.shuffle_prob = 1.0 if "shuf" in kind[len(base):] else 0.0
        if "dim" in kind[len(base):]:
            cfg.photon_gain = (DIM_GAIN, DIM_GAIN)
    return cfg


# the "dim" evaluation kinds: the template's photons divided by this, which takes
# the phase-1 mice with 1-2 photons per frame at footprint support to the 0.3-0.7
# of 776270, 876767 and 878030 (docs/plan.md, "Low photon counts")
DIM_GAIN = 3.0


def _base_kind(kind: str) -> str:
    for b in ("real", "sim", "free", "fast", "dark", "motion"):
        if kind.startswith(b):
            return b
    return kind


# dark: fast simulations where 30% of the footprint pixels sit on a nearly dark baseline;
# the "pz" and "shuf" variants use their base kind's seed, so they score the same windows
EVAL_SEEDS = {"real": 10_000, "sim": 20_000, "free": 40_000, "fast": 50_000, "dark": 60_000, "motion": 70_000}


def sim_config(args, free: bool = False) -> SimConfig:
    """Simulation settings from the arguments; free: no activity anywhere."""
    if free:
        return SimConfig(silent_trial=1.0, photon_zeros=args.photon_zeros, pz_legacy=args.pz_legacy,
                         photon_gain=tuple(float(v) for v in args.photon_gain.split(",")))
    lo, hi = (float(v) for v in args.silent_frac.split(","))
    dlo, dhi = (float(v) for v in args.dark_frac.split(","))
    pair = lambda s: tuple(float(v) for v in s.split(","))  # noqa: E731
    return SimConfig(silent_trial=args.silent_trial, silent_frac=(lo, hi), fast_global_sd=args.fast_global_sd,
                     fast_field_sd=args.fast_field_sd, n_fast_fields=args.n_fast_fields, dark_frac=(dlo, dhi),
                     shuffle_prob=args.shuffle_prob, photon_zeros=args.photon_zeros, pz_legacy=args.pz_legacy,
                     photon_gain=pair(args.photon_gain),
                     motion_sd_px=pair(args.motion_sd), motion_tau_frames=pair(args.motion_tau),
                     focus_sd_px2=pair(args.focus_sd), local_sd_px=pair(args.local_sd))


def train(args) -> None:
    if not (math.isfinite(args.ema) and 0 <= args.ema < 1):
        raise SystemExit(f"--ema must be a decay in [0, 1) (0: off), not {args.ema}")
    if not (math.isfinite(args.l2sp) and args.l2sp >= 0):
        raise SystemExit(f"--l2sp must be a strength >= 0 (0: off), not {args.l2sp}")
    if args.l2sp and not args.init:
        raise SystemExit("--l2sp pulls toward the --init checkpoint: give --init")
    torch.manual_seed(args.seed)
    files = quality_files(list_trials(args.root), args)
    if args.exclude:
        skip = set(args.exclude.split(","))
        groups = session_groups(files)
        print("excluded from training: " + ", ".join(f"{s} ({len(groups[s])} trials)" for s in sorted(skip & set(groups))))
        files = [f for s, fl in groups.items() if s not in skip for f in fl]
    if not files:
        raise SystemExit(f"no trial files under {args.root}")
    if args.val_sessions or args.train_sessions:
        if not args.val_sessions:
            raise SystemExit("--train_sessions needs --val_sessions")
        train_files, val_files, how = explicit_split(session_groups(files), args.val_sessions, args.train_sessions)
    else:
        train_files, val_files, how = split_files(files, args.val_fraction, args.seed)
    if args.max_train_s:
        train_files = cap_seconds(train_files, args.max_train_s)
        how += f"; training capped at {args.max_train_s:g} s per session and path"
    files = train_files + val_files  # only these are read
    secs = seconds_by_session(train_files)
    print(f"training data: {sum(secs.values()):.0f} s of path-time from {len(secs)} sessions ("
          + ", ".join(f"{s} {v:.0f} s" for s, v in sorted(secs.items())) + ")", flush=True)
    mcfg = _model_config(args)
    if mcfg.gap + mcfg.window_reach > mcfg.gap + 1 + mcfg.temporal_reach:
        raise SystemExit("window_reach must not exceed the temporal reach + 1 (the window would run out)")
    wcfg = WindowConfig(out_frames=args.out_frames, out_px=args.out_px,
                        t_reach=mcfg.gap + 1 + mcfg.temporal_reach, s_reach=mcfg.spatial_reach,
                        w_unfitted=args.w_unfitted, real_edge_s=args.real_edge_s)
    # event masking only in training windows, so validation metrics stay comparable across runs
    wcfg_train = replace(wcfg, event_mask=args.event_mask, event_margin=args.event_margin,
                         sim_event_weight=args.sim_event_weight, edge_frac=args.edge_frac)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = None
    if device.type == "cuda" and not args.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stats = load_stats(files, stats_cache(args, out), workers=args.workers, sidecar_dirs=args.sidecars)
    print(f"trial stats for {len(files)} files in {time.time() - t0:.0f} s; gain "
          + " ".join(f"{stats[str(f)]['gain']:.1f}" for f in files[:8]) + (" ..." if len(files) > 8 else ""), flush=True)
    bad = _drop_files(files, stats)
    if bad:
        train_files = [f for f in train_files if str(f) not in bad]
        val_files = [f for f in val_files if str(f) not in bad]
        files = train_files + val_files
        secs = seconds_by_session(train_files)
        if not train_files:
            raise SystemExit("no training files left")

    model = SlabBlindSpotNet(mcfg).to(device)
    if args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        state = ck["model"]
        if config_from_dict(ck["config"]).varfac_norm == "window" and mcfg.varfac_norm == "trial":
            state = adapt_varfac_input(state, mcfg.width)
            print("variance-factor weights folded from the window-median input to the trial-median input")
        print(f"initialized from {args.init} (step {ck.get('step')}): {load_compatible(model, state, ck['config'])}")
    n_params = sum(p.numel() for p in model.parameters())
    net = fast_model(model, device, args)  # forward passes; model for saving and the optimizer
    anchor =[p.detach().clone() for p in model.parameters()] if args.l2sp else None
    # --ema: the moving average (from the initial weights) is what validation scores and the checkpoints hold
    avg_model = copy.deepcopy(model).requires_grad_(False) if args.ema else None
    net_val = net if avg_model is None else fast_model(avg_model, device, args)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    warm = min(1000, max(1, args.steps // 5))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))

    sim = sim_config(args) if args.data in ("sim", "mix") else None
    frac = args.sim_frac if args.data == "mix" else 1.0
    # validation loaders start their workers at each validation and stop them
    # after it: kept alive, the 5 sets' workers (12 each) held memory for the
    # whole run on top of the training workers'
    kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    mk = lambda fl, n, seed, s, fr=1.0: DataLoader(  # noqa: E731
        WindowDataset(fl, wcfg_train if seed == args.seed else wcfg, stats, length=n, seed=seed, sim=s,
                      flip=("none" if args.no_flip else args.flip) if seed == args.seed else "none",
                      sim_frac=fr, subtract_frac=args.subtract_frac if seed == args.seed else 0.0,
                      max_open=getattr(args, "max_open", 32)),
        batch_size=args.batch, persistent_workers=False, **kw)
    dl = mk(train_files, args.steps * args.batch * args.accum, args.seed, sim, frac)
    # validation sets: one loader per validation mouse (session), --val_windows
    # windows each, the first windows held-out evaluation draws for that
    # session with the same seed (plan.md, "Experiments 1 and 2": the number of
    # mice, not of windows, limits checkpoint selection); --val_pooled draws
    # --val_windows from all validation files together (S0-S3 and earlier)
    vgroups = {"all": val_files} if args.val_pooled or not val_files else session_groups(val_files)
    sets = {}
    if val_files:
        sets["real"] = (10_000, None)
        sets["sim"] = (20_000, eval_sim("sim", args.photon_zeros, args.pz_legacy))
        if args.val_free:
            sets["free"] = (40_000, sim_config(args, free=True))
        if args.val_fast:
            sets["fast"] = (50_000, eval_sim("fast", args.photon_zeros, args.pz_legacy))
        if args.val_motion:
            sets["motion"] = (70_000, eval_sim("motion", args.photon_zeros, args.pz_legacy))
    evals = {name: {g: mk(fl, args.val_windows, seed0 + args.seed, s) for g, fl in vgroups.items()}
             for name, (seed0, s) in sets.items()}
    evals["tr"] = {"all": mk(train_files, args.val_windows, 30_000 + args.seed, sim, frac)}  # did it learn anything at all?
    region = loss_region(wcfg)
    sel_kinds = parse_select_on(args.select_on or ("real" if args.data == "mix" else args.data)) if val_files         else {"tr": 1.0}
    missing = [k for k in sel_kinds if k not in evals]
    if missing:
        raise SystemExit(f"--select_on {','.join(missing)}: no such validation set ({', '.join(evals)})")
    sel_key = "+".join(sel_kinds) + f"_{args.select}"

    (out / "config.json").write_text(json.dumps({
        "model": asdict(mcfg), "window": asdict(wcfg), "args": vars(args), "n_params": n_params,
        "device": str(device), "amp": str(amp_dtype), "split": how, "sim": asdict(sim or SimConfig()),
        "train_files": [str(f) for f in train_files], "val_files": [str(f) for f in val_files],
        "train_seconds": secs,
    }, indent=2, default=str))
    print(f"{len(files)} trial files; split {how}: {len(train_files)} train, {len(val_files)} validation; "
          f"training on {args.data} data, selecting on {sel_key}")
    print(f"model {n_params:,} parameters on {device} (amp {amp_dtype}); window {wcfg.out_frames + 2 * wcfg.t_reach} frames x "
          f"{wcfg.out_px + 2 * wcfg.s_reach}^2 px, loss on {wcfg.out_frames} x {wcfg.out_px}^2", flush=True)

    fields = None
    best = math.inf
    ema = None

    def weights():  # with --ema the average is the model; the raw weights are kept beside it
        if avg_model is None:
            return {"model": model.state_dict()}
        return {"model": avg_model.state_dict(), "model_raw": model.state_dict()}

    def save(name, step):
        torch.save({**weights(), "optimizer": opt.state_dict(), "config": asdict(mcfg),
                    "window": asdict(wcfg), "step": step}, out / name)

    def keep(step):  # --keep_every: the model at this step, for later selection studies (no optimizer state)
        if args.keep_every:
            (out / "steps").mkdir(exist_ok=True)
            torch.save({**weights(), "config": asdict(mcfg), "window": asdict(wcfg), "step": step},
                       out / "steps" / f"step_{step:06d}.pt")

    def validate(step):
        nonlocal fields, best
        row = {"step": step, "train_loss": ema if ema is not None else float("nan")}
        per_group = {}  # set -> group -> that group's selection value
        for name, loaders in evals.items():
            res = {g: evaluate(net_val, ld, region, device, amp_dtype, args) for g, ld in loaders.items()}
            for k in dict.fromkeys(k for r in res.values() for k in r):  # mean over groups, the old column names
                vals = [r[k] for r in res.values() if k in r and math.isfinite(r[k])]
                row[f"{name}_{k}"] = float(np.mean(vals)) if vals else math.nan
            per_group[name] = {g: selection_score({f"x_{k}": v for k, v in r.items()}, "x", args.select)
                               for g, r in res.items()}
            if len(res) > 1:
                for g, v in per_group[name].items():
                    row[f"{name}@{g.split('_')[0]}_{args.select}"] = v
        row[sel_key] = combined_selection(row, per_group, sel_kinds, args.select)
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
    keep(0)
    step = 0
    opt.zero_grad(set_to_none=True)
    # time spent waiting for the data loaders, per log interval: the rest is the
    # model (and validation); near the interval's length means data-bound
    waits = LoaderWait()
    n_prof = getattr(args, "profile_steps", 0)
    prof = StepProfiler(n_prof, StepProfiler.quiet_start(n_prof, (args.val_every, args.save_every, args.keep_every)),
                        device=device)
    try:
        for micro, b in enumerate(dl, 1):  # accum micro-batches per optimizer step
            waits.got()
            b, mu, var = _forward(net, b, device, amp_dtype, region)
            loss, mu_loss, var_loss = _losses(args, b, mu, var, region)
            scaler.scale(loss / args.accum).backward()
            if micro % args.accum:
                waits.idle()
                continue
            step = micro // args.accum
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            lr_step = sched.get_last_lr()[0]  # the rate this step used
            prof.before_val(step)
            sched.step()
            with torch.no_grad():
                if anchor is not None:  # --l2sp: back toward the initial weights by lr x l2sp of the distance
                    torch._foreach_lerp_(list(model.parameters()), anchor, min(1.0, lr_step * args.l2sp))
                if avg_model is not None:
                    torch._foreach_lerp_(list(avg_model.parameters()), list(model.parameters()), 1.0 - args.ema)
            ema = mu_loss.item() if ema is None else 0.98 * ema + 0.02 * mu_loss.item()
            if step % args.log_every == 0 or step == 1:
                mem = f" gpu peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if device.type == "cuda" else ""
                wait, interval = waits.report()
                print(f"step {step} mu_loss {mu_loss.item():.4f} (avg {ema:.4f}) var_loss {var_loss.item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f} s{mem}; data wait {wait:.1f} of "
                      f"{interval:.1f} s", flush=True)
            last_step = step == args.steps or (args.max_hours and time.time() - t0 > 3600 * args.max_hours)
            if last_step:  # close a window that training ends inside, before the final validation and saves
                prof.finish(step)
            if step % args.val_every == 0 or last_step:
                validate(step)
            if step % args.save_every == 0 or last_step:
                save("last.pt", step)
            if args.keep_every and (step % args.keep_every == 0 or last_step):
                keep(step)
            prof.after_step(step)
            if last_step:
                break
            waits.idle()
    finally:
        prof.finish(step)  # a window left open (training ended early or raised)
    print(f"done after {step} steps, {time.time() - t0:.0f} s; best {sel_key} {best:.5g}")


def evaluate_checkpoints(args) -> None:
    """Scores of saved checkpoints on each session's windows, real and
    simulated (see ``evaluate``): one row per checkpoint, session and data
    kind in <out>/eval.csv. The windows depend only on the seed and the
    session's files, so every checkpoint is scored on the same ones; with the
    training seed, the first ``val_windows`` of a held-out session are the
    ones its training validation used. With --per_window, windows.csv has
    each window's error sums (see ``evaluate``)."""
    files = quality_files(list_trials(args.root), args)
    if not files:
        raise SystemExit(f"no trial files under {args.root}")
    groups = session_groups(files)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = None
    if device.type == "cuda" and not args.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = stats_cache(args, out)
    t0 = time.time()
    stats = load_stats(files, cache, workers=args.workers, sidecar_dirs=args.sidecars)
    print(f"trial stats for {len(files)} files in {time.time() - t0:.0f} s; sessions: "
          + ", ".join(f"{s} ({len(f)} trials)" for s, f in sorted(groups.items())), flush=True)
    bad = _drop_files(files, stats)
    if bad:
        files = [f for f in files if str(f) not in bad]
        groups = session_groups(files)
    excess_report(groups, stats, out / "excess.csv")
    rows = []
    specs = []
    for spec in args.ckpt:  # name=path, or name=folder/step_*.pt for all kept steps (named name@step_<N>)
        name, path = spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec)
        if any(c in path for c in "*?["):
            matches = sorted(Path(path).parent.glob(Path(path).name))
            if not matches:
                raise SystemExit(f"no checkpoints match {path}")
            specs += [(f"{name}@{m.stem}", str(m)) for m in matches]
        else:
            specs.append((name, path))
    # every checkpoint with the same window configuration is scored on the same
    # windows, drawn once (the loaders were the limit once the forward pass is
    # compiled: about 5 simulated windows/s on 16 vCPUs), in chunks of
    # --share_chunk windows; checkpoints of one architecture share one
    # (compiled) model and only their weights are loaded in turn
    cks = []
    for name, path in specs:
        ck = torch.load(path, map_location=device, weights_only=False)
        wcfg = WindowConfig(**ck["window"])  # checkpoints from before real_edge_s get its default (2 s)
        if args.real_edge_s is not None:
            wcfg = replace(wcfg, real_edge_s=args.real_edge_s)
        cks.append({"name": name, "path": path, "step": ck.get("step"), "config": ck["config"], "state": ck["model"],
                    "wcfg": wcfg})
        print(f"{name}: {path} (step {ck.get('step')})", flush=True)
    models = {}
    for c in cks:
        key = json.dumps(c["config"], sort_keys=True, default=str)
        if key not in models:
            models[key] = fast_model(SlabBlindSpotNet(config_from_dict(c["config"])).to(device), device, args)
        c["model"] = models[key]
    by_window = defaultdict(list)
    for c in cks:
        by_window[json.dumps(asdict(c["wcfg"]), sort_keys=True, default=str)].append(c)

    def write_rows(c, sess, kind, r, pw, secs):
        rows.append({"checkpoint": c["name"], "step": c["step"], "session": sess, "kind": kind,
                     "trials": len(groups[sess]), "windows": args.windows, **r})
        if pw:  # one row per window, for bootstrap intervals over windows and sessions
            new = not (out / "windows.csv").exists()
            with open(out / "windows.csv", "a", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=["checkpoint", "step", "session", "kind", "window"] + list(pw[0]))
                if new:
                    wr.writeheader()
                for i, q in enumerate(pw):
                    wr.writerow({"checkpoint": c["name"], "step": c["step"], "session": sess, "kind": kind,
                                 "window": i, **q})
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with open(out / "eval.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields, restval=float("nan"))
            wr.writeheader()
            wr.writerows(rows)

        def rel(cl, pn):
            a, b = r.get(f"{cl}mse8_{pn}", math.nan), r.get(f"{cl}mse8_mean_image", math.nan)
            return f"{100 * (a / b - 1):+.1f}%" if b > 0 else "nan"

        print(f"  {c['name']} {sess} {kind} ({secs:.0f} s for {len(by_window[wkey])} checkpoints): "
              + "; ".join(f"{cl or 'all_'}[net {rel(cl, 'net')} silo {rel(cl, 'silo')} oracle {rel(cl, 'oracle')} "
                          f"bias net {r.get(f'{cl}bias_net', math.nan):+.3f} "
                          f"mean image {r.get(f'{cl}bias_mean_image', math.nan):+.3f}]"
                          for cl in ("sup_", "bg_", ""))
              + f"; z_var {r['z_var']:.3f} mix {r['mix']:.3f}", flush=True)

    for wkey, group in by_window.items():
        wcfg = group[0]["wcfg"]
        region = loss_region(wcfg)
        for sess in sorted(groups):
            for kind in args.kinds.split(","):
                sim = eval_sim(kind, pz_legacy=args.pz_legacy)
                seed = EVAL_SEEDS[_base_kind(kind)] + args.seed
                ds = WindowDataset(groups[sess], wcfg, stats, length=args.windows, seed=seed, sim=sim, flip=False)
                loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                                    pin_memory=device.type == "cuda")
                t1 = time.time()
                tots = {c["name"]: defaultdict(float) for c in group}
                pws = {c["name"]: ([] if args.per_window else None) for c in group}
                it = iter(loader)
                while True:
                    chunk = list(itertools.islice(it, max(1, args.share_chunk // args.batch)))
                    if not chunk:
                        break
                    for c in group:
                        getattr(c["model"], "_orig_mod", c["model"]).load_state_dict(c["state"])
                        evaluate(c["model"], chunk, region, device, amp_dtype, args, per_window=pws[c["name"]],
                                 tot=tots[c["name"]], finish=False)
                for c in group:
                    write_rows(c, sess, kind, evaluate_totals(tots[c["name"]]), pws[c["name"]], time.time() - t1)
    print(f"done in {time.time() - t0:.0f} s")


def excess_report(groups: dict, stats: dict, path: Path) -> None:
    """Per-trial excess of the residual variance over photon noise
    (``simulate.excess_summary``) to a CSV, and session medians to the log."""
    rows = []
    for sess, fl in sorted(groups.items()):
        for f in fl:
            with TrialFile(f) as tf:
                rows.append({"session": sess, "file": str(f), **excess_summary(tf, stats[str(f)])})
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, restval=float("nan"))
        wr.writeheader()
        wr.writerows(rows)
    print("residual variance over photon noise (session medians of trials), valid pixels / brightest 10% / "
          "labeled / support; 1, 8, 64-frame means:")
    for sess in sorted(groups):
        rs = [r for r in rows if r["session"] == sess and "all_r1" in r]
        if not rs:
            print(f"  {sess}: no residual maps in the stats")
            continue
        med = lambda k: float(np.nanmedian([r.get(k, math.nan) for r in rs]))  # noqa: E731
        print(f"  {sess}: all " + "/".join(f"{med(f'all_r{n}'):.2f}" for n in (1, 8, 64))
              + "  top10% " + "/".join(f"{med(f'all_top10_r{n}'):.2f}" for n in (1, 8, 64))
              + "  labeled " + "/".join(f"{med(f'lab_r{n}'):.2f}" for n in (1, 8, 64))
              + "  support " + "/".join(f"{med(f'sup_r{n}'):.2f}" for n in (1, 8, 64))
              + f"  | top10% fast {med('top10_fast'):+.2f}, slow SD/mean {med('top10_slow_sd_rel'):.3f}, "
              + f"brightness {med('top10_mean_over_scale'):.1f}x scale; support px {med('support_px'):.0f}", flush=True)


def add_quality_arguments(p) -> None:
    p.add_argument("--min_peak_r", type=float, default=0.3,
                   help="leave out trials whose activity calibration failed: the exporter's peak correlation "
                        "(attribute peakR) below this (0: keep all; io.filter_trials)")
    p.add_argument("--min_path_trials", type=int, default=4,
                   help="leave out an imaging path whole if fewer than this many of its trials are usable (calibrated "
                        "exports only; 0: keep all)")


def add_eval_arguments(p) -> None:
    p.add_argument("root", help="folder with exported trial files (searched recursively)")
    p.add_argument("--ckpt", nargs="+", required=True,
                   help="checkpoints to score, each path or name=path; a path may be a glob (e.g. "
                        "run=/data/run/train/steps/step_*.pt), each match named name@<file stem>")
    p.add_argument("--out", default="results/eval")
    p.add_argument("--kinds", default="real,sim",
                   help="comma-separated: real, sim, free (simulated without activity), fast (simulated with fast "
                        "shared brightness changes), motion (fast plus residual motion, focus and local motion); "
                        "suffixes pz (realistic low counts), shuf (shuffled timing), dim (a third of the photons), "
                        "e.g. freedim")
    p.add_argument("--windows", type=int, default=192, help="windows per session and data kind")
    p.add_argument("--real_edge_s", type=float, default=None,
                   help="real windows: no loss within this many seconds of a trial edge (default: the checkpoint's, "
                        "2 s for checkpoints from before the option; 0 = keep the edges)")
    p.add_argument("--pz_legacy", action="store_true",
                   help="pz kinds (fastpz, motionpz, ...): draw the low counts as up to 2026-11-21, to compare with "
                        "evaluations made before (SimConfig.pz_legacy)")
    p.add_argument("--share_chunk", type=int, default=64,
                   help="windows held in memory at once while every checkpoint is scored on them (about 100 MB each)")
    p.add_argument("--per_window", action="store_true",
                   help="also write windows.csv: each window's 8-frame squared errors at support and background "
                        "pixels (for bootstrap intervals over windows and sessions)")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--stats_cache", default="", help="folder for cached trial stats (default: <out>/stats)")
    p.add_argument("--stats_from", default="", help="copy cached trial stats from here first (e.g. a training run's)")
    p.add_argument("--sidecars", default="",
                   help="comma-separated stats overlay folders (<session>/<path>/<trial>_stats_v5.npz, e.g. a data "
                        "asset made by `slap2-f0 stats ROOT --out DIR`), searched after each trial's own sidecar")
    add_quality_arguments(p)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_compile", action="store_true", help="no torch.compile (default: compiled when Triton is installed)")
    p.add_argument("--no_cudnn_benchmark", action="store_true", help="cuDNN's default algorithms instead of autotuning")
    p.add_argument("--seed", type=int, default=0, help="the training seed reproduces its validation windows")


def probe(args) -> None:
    """Peak GPU memory and time per training step over a grid of settings, on
    random inputs of the real window size; writes probe.csv to --out."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    grid = [(b, w, fr, md, br) for br in [int(v) for v in args.box_radii.split(",")]
            for md in [int(v) for v in args.max_dilations.split(",")]
            for fr in [int(v) for v in args.full_res_list.split(",")]
            for w in [int(v) for v in args.widths.split(",")]
            for b in [int(v) for v in args.batches.split(",")]]
    for b, w, fr, md, br in grid:
        dil = [2 ** k for k in range(int(math.log2(md)) + 1)]
        mcfg = ModelConfig(width=w, full_res_blocks=fr, temporal_dilations=dil, box_radius=br, hp_width=args.hp_width)
        T = args.out_frames + 2 * (mcfg.gap + 1 + mcfg.temporal_reach)
        S = args.out_px + 2 * mcfg.spatial_reach
        region = (slice(T // 2 - args.out_frames // 2, T // 2 + args.out_frames // 2),
                  slice(mcfg.spatial_reach, mcfg.spatial_reach + args.out_px),
                  slice(mcfg.spatial_reach, mcfg.spatial_reach + args.out_px))
        row = {"batch": b, "width": w, "full_res_blocks": fr, "max_dilation": md, "box_radius": br, "window_frames": T,
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
    p.add_argument("--box_radii", default="-1", help="comma-separated; -1 = slab model")
    p.add_argument("--hp_width", type=int, default=24)
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
    p.add_argument("--box_radius", type=int, default=-1,
                   help="box blind spot: pixels further than this (rows or columns) are visible at every frame; "
                        "-1 = slab blind spot (no same-frame context)")
    p.add_argument("--hp_width", type=int, default=24, help="channels of the half-plane branches (box model)")
    p.add_argument("--hp_temporal_dilations", default="", help="comma-separated, default 1,2,4,8,16")
    p.add_argument("--hp_spatial_dilations", default="", help="comma-separated, default 1,1,1,2,4")
    p.add_argument("--hp_full_res_blocks", type=int, default=2)
    p.add_argument("--pool_scales", default="", help="half-widths of the pooled same-frame features, 0 = whole window; "
                                                     "default 8,16,32,0")
    p.add_argument("--varfac_norm", default="trial", choices=["trial", "window"],
                   help="variance-factor input relative to the trial's median (default) or the window's (v1-v6)")
    p.add_argument("--out_frames", type=int, default=64)
    p.add_argument("--out_px", type=int, default=32)
    p.add_argument("--w_unfitted", type=float, default=0.0, help="loss weight of labeled pixels outside every footprint")
    p.add_argument("--event_mask", type=float, default=0.0,
                   help="real training windows: no loss where GIAnT activity exceeds this times its RMS over the trial, "
                        "and --event_margin frames around (the target there holds unfitted activity; 0 = off)")
    p.add_argument("--event_margin", type=int, default=8)
    p.add_argument("--real_edge_s", type=float, default=2.0,
                   help="real windows: no loss within this many seconds of a trial's first or last frame, where "
                        "GIAnT's baseline (so the target) is biased; also in validation (0 = keep the edges)")
    p.add_argument("--edge_frac", type=float, default=0.0,
                   help="share of training windows whose loss region starts or ends at a trial edge (the baseline "
                        "is biased in the first and last ~0.5 s otherwise)")
    p.add_argument("--sim_event_weight", type=float, default=1.0,
                   help="simulated training windows: loss weight where the activity exceeds half the photon-noise SD "
                        "(and --event_margin frames around); > 1 makes a baseline that follows the activity cost more")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=1, help="micro-batches of --batch windows per optimizer step")
    p.add_argument("--profile_steps", type=int, default=0,
                   help="profile this many optimizer steps after 20 warm-up steps and print the top GPU and CPU "
                        "operations and torch.compile's counters (0: off)")
    p.add_argument("--max_open", type=int, default=32,
                   help="trials (with their simulation templates, about 8 MB each) each data-loader worker keeps "
                        "open, least recently used first out; 0 = no limit")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--max_hours", type=float, default=0.0, help="stop (and validate, save) after this long; 0 = no limit")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--data", choices=["sim", "real", "mix"], default="sim",
                   help="train on simulated windows (true baseline), real residuals or a mix (--sim_frac)")
    p.add_argument("--sim_frac", type=float, default=0.5, help="with --data mix: fraction of simulated windows")
    p.add_argument("--silent_trial", type=float, default=0.0,
                   help="probability that a simulated trial has no activity at all")
    p.add_argument("--silent_frac", default="0,0",
                   help="lo,hi: otherwise this fraction (uniform) of a simulated trial has no activity")
    p.add_argument("--subtract_frac", type=float, default=0.0,
                   help="fraction of real training windows whose input is movie - activity (no fitted events)")
    p.add_argument("--fast_global_sd", type=float, default=0.0,
                   help="simulated fast field-wide brightness changes: largest SD (fraction), 0 = none")
    p.add_argument("--fast_field_sd", type=float, default=0.0,
                   help="simulated fast regional brightness changes: largest SD per field (fraction), 0 = none")
    p.add_argument("--n_fast_fields", type=int, default=3)
    p.add_argument("--dark_frac", default="0,0",
                   help="lo,hi: fraction (uniform) of footprint pixels of a simulated trial whose baseline is scaled "
                        "down by 0.02-0.3 (sources on nearly dark pixels)")
    p.add_argument("--val_fast", action="store_true",
                   help="also validate on held-out windows simulated with fast shared brightness changes (fast_*)")
    p.add_argument("--motion_sd", default="0,0",
                   help="lo,hi: simulated rigid residual motion, SD per axis uniform in this range (px; 0,0 = none)")
    p.add_argument("--motion_tau", default="0.5,40", help="lo,hi: its time constant, log-uniform (frames)")
    p.add_argument("--focus_sd", default="0,0",
                   help="lo,hi: simulated focus fluctuations, SD of the isotropic blur change (px^2; 0,0 = none)")
    p.add_argument("--local_sd", default="0,0",
                   help="lo,hi: simulated local (non-rigid) motion, SD of its fast and of its slow part (px; 0,0 = none)")
    p.add_argument("--val_motion", action="store_true",
                   help="also validate on held-out windows simulated with fast brightness changes, residual motion, "
                        "focus fluctuations and local motion at the measured sizes (motion_*)")
    p.add_argument("--exclude", default="", help="comma-separated sessions to leave out of training and validation")
    p.add_argument("--val_sessions", default="",
                   help="comma-separated validation sessions (best.pt is chosen on these); default: drawn at random "
                        "(--val_fraction)")
    p.add_argument("--train_sessions", default="",
                   help="comma-separated training sessions (needs --val_sessions); default: all other sessions")
    p.add_argument("--max_train_s", type=float, default=0.0,
                   help="use only the first trials of each training session and path, up to this many seconds "
                        "(0: all)")
    p.add_argument("--keep_every", type=int, default=0,
                   help="also keep the model every this many steps (and at step 0 and the last) as "
                        "steps/step_<N>.pt (0: off)")
    p.add_argument("--val_free", action="store_true",
                   help="also validate on held-out windows simulated without activity (free_*)")
    p.add_argument("--init", default="", help="start from this checkpoint (e.g. a simulation-pretrained best.pt)")
    p.add_argument("--ema", type=float, default=0.0,
                   help="validate, select and save an exponential moving average of the weights, this decay per "
                        "optimizer step (0.995: a 200-step horizon), starting at the initial weights; the raw "
                        "weights are saved as model_raw (0: off)")
    p.add_argument("--l2sp", type=float, default=0.0,
                   help="with --init: after each step, move the weights back toward the initial ones by "
                        "lr x l2sp of the distance (decoupled L2-SP; at lr 1e-4, 10 is a 1000-step time constant; "
                        "0: off)")
    p.add_argument("--stats_cache", default="", help="folder for cached trial stats (default: <out>/stats)")
    p.add_argument("--stats_from", default="", help="copy cached trial stats from here first (e.g. an earlier run's)")
    p.add_argument("--sidecars", default="",
                   help="comma-separated stats overlay folders (<session>/<path>/<trial>_stats_v5.npz, e.g. a data "
                        "asset made by `slap2-f0 stats ROOT --out DIR`), searched after each trial's own sidecar")
    add_quality_arguments(p)
    p.add_argument("--w_mu", type=float, default=1.0, help="weight of the mean loss")
    p.add_argument("--w_var", type=float, default=0.1, help="weight of the noise (variance) loss")
    p.add_argument("--window_reach", type=int, default=254, help="frames beyond the gap for the blind window stats")
    p.add_argument("--max_log_correction", type=float, default=0.7, help="soft cap of |log| of the baseline correction")
    p.add_argument("--select", choices=["balanced", "balanced_free", "mse8", "loss"], default="balanced",
                   help="best.pt criterion on held-out windows: 8-frame MSE vs the mean image, averaged over "
                        "support and background (balanced; balanced_free adds labeled pixels of the activity-free "
                        "simulation, needs --val_free) or over all pixels (mse8), or the mean loss")
    p.add_argument("--select_on", default="",
                   help="validation set(s) for best.pt, with optional weights: real, sim, free, fast, motion, "
                        "e.g. real,fast or real:1,fast:0.5 (the weighted mean of each set's score); default: the "
                        "training data kind (real for --data mix)")
    p.add_argument("--val_pooled", action="store_true",
                   help="validation windows drawn from all validation sessions together (S0-S3 and earlier); "
                        "default: --val_windows windows per validation session, scores averaged over sessions")
    p.add_argument("--no_flip", action="store_true", help="no random flips of training windows (= --flip none)")
    p.add_argument("--flip", default="rows", choices=["rows", "both", "none"],
                   help="random flips of training windows: up/down only (rows, the DMD's short axis; default), "
                        "also left/right (both, v1-v6), or none. Never rotations: the microscope scans")
    p.add_argument("--motion_features", action="store_true",
                   help="box model: also the local displacement and blur seen in the surround, as the change they "
                        "make at the pixel (model.motion_features); warm-starts from checkpoints without them")
    p.add_argument("--fuse_blocks", type=int, default=0,
                   help="pointwise residual blocks after the two fusion layers (ModelConfig.fuse_blocks): more "
                        "processing after the branches are combined, same reach; starts out exactly as --init")
    p.add_argument("--branch_mlp", type=int, default=0,
                   help="pointwise residual blocks after every dilated block of every branch "
                        "(ModelConfig.branch_mlp): more nonlinear processing of the same window, same reach")
    p.add_argument("--edge_features", action="store_true",
                   help="per-side window statistics into the fusion and edge-gated corrections of the baseline and "
                        "noise (ModelConfig.edge_features); starts out exactly as the --init checkpoint")
    p.add_argument("--hp_shared", action="store_true",
                   help="one half-plane trunk for all four directions (v6) instead of one per direction")
    p.add_argument("--photon_zeros", action="store_true",
                   help="simulate realistic low counts (zero-inflated, never negative; SimConfig.photon_zeros); "
                        "validation simulations too")
    p.add_argument("--pz_legacy", action="store_true",
                   help="with --photon_zeros: draw the low counts as up to 2026-11-21 (v7-v9, S0-S3), whose noise "
                        "had about half the measured correlations and too much skew (SimConfig.pz_legacy)")
    p.add_argument("--shuffle_prob", type=float, default=0.0,
                   help="share of simulated trials with shuffled activity timing (no stimulus rhythm)")
    p.add_argument("--photon_gain", default="1,1",
                   help="lo,hi: simulated windows multiply the template's gain (counts per photon) by a factor "
                        "log-uniform in this range, the noise of a dimmer session (1,1 = off; SimConfig.photon_gain)")
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--val_windows", type=int, default=256)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--no_amp", action="store_true", help="disable mixed precision on the GPU")
    p.add_argument("--no_compile", action="store_true", help="no torch.compile (default: compiled when Triton is installed)")
    p.add_argument("--no_cudnn_benchmark", action="store_true", help="cuDNN's default algorithms instead of autotuning")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1000)
