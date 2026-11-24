"""Random training windows from exported trials, real or simulated.

A window covers ``out_frames`` frames and ``out_px`` x ``out_px`` pixels where
the loss is computed, plus the model's temporal and spatial reach on every
side, so every output sees real data (or a real trial edge) in its context,
never an artificial window border. Windows may run past the trial's ends,
which teaches the model the trial edges.

Real windows (``sim=None``): the target is the residual movie - activity.
Loss weights (see docs/plan.md): valid pixels and frames only; footprint-
support pixels and unlabeled (off-dendrite) pixels weight 1; labeled pixels
outside every footprint weight ``w_unfitted`` (default 0), because their
residual still holds activity GIAnT did not fit. In a trial without sources
(``nSources`` 0, e.g. a GFP control without an activity sensor) the activity
is known to be zero, so every valid pixel weighs 1. Frames within ``real_edge_s``
(2 s) of a trial's first or last frame weigh 0: GIAnT's baseline is biased there, and
with it the activity and the target. With ``event_mask`` > 0,
voxels where GIAnT's activity exceeds event_mask times the pixel's activity
RMS over the trial, and ``event_margin`` frames around them, weigh 0: there
the target still holds the part of the events GIAnT did not fit (up to ~20%),
which the context makes predictable (the recordings' stimuli recur every 0.5
and 3 s, well outside the blind spot), so a network rewarded for predicting
it puts activity into the baseline. Simulations teach the baseline during
events, with the true baseline as target.

Simulated windows (``sim=SimConfig()``): each item is a new virtual trial
built on a random template (see ``simulate``); the target is the noise-free
baseline, and every valid pixel and frame has weight 1, or ``sim_event_weight``
where the activity exceeds half the pixel's photon-noise SD (and
``event_margin`` frames around): a baseline that follows the activity (the
v6 box models do, when many synapses respond to a stimulus together, which
looks like a frame-wide brightness change) then costs more. The item's gain
is the virtual trial's (``SimConfig.photon_gain`` can raise it), so the
network's gain input and the loss weights see the noise that was simulated.

Activity-subtracted real windows (``subtract_frac``): with this
probability a real window's input is movie - activity too, with the
whole-trial inputs of that residual (its mean and variance maps), so bright
dendrites show their real noise and excess but no fitted events; the target
is unchanged. In sensor sessions every bright pixel in the loss carries
activity, and a network fine-tuned on them alone learned to set the baseline
of bright dendrites low even without events (v4 run, GFP control).

Trial edges (``edge_frac``): with this probability a window's loss region
starts at the trial's first frame or ends at its last (within
``edge_frames``). Uniformly placed windows put only ~7% of their outputs in the
first or last 64 frames, and the v6 baselines are biased there by 1-4% of the
baseline (simulations against the truth, and real trials).

Mixed windows (``sim`` and ``sim_frac`` < 1): each item is simulated with
probability ``sim_frac`` and real otherwise, so one batch can hold both
(e.g. fine-tuning on real residuals with simulated activity-free regions
as counterexamples to "bright labeled pixels always carry activity").

Every item also carries ``noisy`` (movie - activity, for the noise head), the
whole-trial variance map and the trial's gain (``stats``), and the support and
labeled masks for evaluation. ``flip`` flips windows at random: "rows"
(default) up/down only, along the DMD's short axis, the one flip that leaves
the microscope's scan alike; "both" also left/right (v1-v6); "none". Never
rotations: the image is not the same rotated (``model.ModelConfig.hp_shared``).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import TrialFile
from .simulate import SimConfig, Template, draw, window


@dataclass
class WindowConfig:
    out_frames: int = 64
    out_px: int = 32
    t_reach: int = 0  # set from the model: gap + 1 + temporal_reach
    s_reach: int = 0  # set from the model: spatial_reach
    w_unfitted: float = 0.0
    min_valid_fraction: float = 0.05  # of loss pixels
    event_mask: float = 0.0  # real windows: no loss where activity > event_mask x its RMS (0 = off)
    event_margin: int = 8  # frames around those voxels that weigh 0 too
    sim_event_weight: float = 1.0  # simulated windows: loss weight during events (1 = as elsewhere)
    edge_frac: float = 0.0  # share of windows whose loss region starts or ends at a trial edge (+-edge_frames)
    edge_frames: int = 64
    # real windows: no loss within real_edge_s of a trial's first or last frame, where GIAnT's
    # baseline, and so the target movie - activity, is biased (0 = keep the edges)
    real_edge_s: float = 2.0


def _crop(a: np.ndarray, r0: int, nr: int, c0: int, nc: int, fill=np.nan) -> np.ndarray:
    out = np.full((nr, nc), fill, np.float32)
    R, C = a.shape
    rs, re, cs, ce = max(r0, 0), min(r0 + nr, R), max(c0, 0), min(c0 + nc, C)
    if rs < re and cs < ce:
        out[rs - r0: re - r0, cs - c0: ce - c0] = a[rs:re, cs:ce]
    return out


class WindowDataset(Dataset):
    """Each item is a random window from a random trial (real or simulated).

    Each worker keeps at most ``max_open`` trials open, with their simulation
    templates (about 8 MB per trial of a 15 min export), least recently used
    first out; 0 = no limit. Unbounded, 12 workers drawing from S1's 253
    trials ran the 64 GB machine out of memory (exit code 137); reopening a
    trial and rebuilding its template takes about 0.1 s.
    """

    def __init__(self, files, cfg: WindowConfig, stats: dict, length: int = 10000, seed: int = 0,
                 sim: SimConfig | None = None, flip: str | bool = "rows", sim_frac: float = 1.0,
                 subtract_frac: float = 0.0, max_open: int = 32):
        self.files = [str(f) for f in files]
        self.cfg = cfg
        self.stats = {str(k): v for k, v in stats.items()}
        self.length = length
        self.seed = seed
        self.sim = sim
        self.flip = {True: "both", False: "none"}.get(flip, flip) if isinstance(flip, bool) else flip
        if self.flip not in ("rows", "both", "none"):
            raise ValueError(f"flip must be rows, both or none, not {flip!r}")
        self.sim_frac = sim_frac
        self.subtract_frac = subtract_frac
        self.max_open = max_open
        self._open: OrderedDict[str, TrialFile] = OrderedDict()
        self._tpl: dict[str, Template] = {}

    def __len__(self) -> int:
        return self.length

    def _trial(self, fn: str) -> TrialFile:
        tf = self._open.get(fn)
        if tf is not None:
            self._open.move_to_end(fn)
            return tf
        tf = self._open[fn] = TrialFile(fn)
        while self.max_open > 0 and len(self._open) > self.max_open:
            old, otf = self._open.popitem(last=False)
            self._tpl.pop(old, None)  # a template refers to its trial
            otf.close()
        return tf

    def _template(self, fn: str) -> Template:
        if fn not in self._tpl:
            self._tpl[fn] = Template(self._trial(fn), self.stats[fn])
        return self._tpl[fn]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng((self.seed, idx))
        c = self.cfg
        nt, ns = c.out_frames + 2 * c.t_reach, c.out_px + 2 * c.s_reach
        sl = (slice(c.t_reach, c.t_reach + c.out_frames), slice(c.s_reach, c.s_reach + c.out_px),
              slice(c.s_reach, c.s_reach + c.out_px))
        use_sim = self.sim is not None and (self.sim_frac >= 1 or rng.random() < self.sim_frac)
        use_sub = not use_sim and self.subtract_frac > 0 and rng.random() < self.subtract_frac
        for _ in range(50):
            fn = self.files[rng.integers(len(self.files))]
            tr, st = self._trial(fn), self.stats[fn]
            # windows at the first or last frames of the trial: simulated ones (and real ones only
            # when real windows keep their edges)
            if c.edge_frac > 0 and (use_sim or c.real_edge_s <= 0) and rng.random() < c.edge_frac:
                e = int(rng.integers(-c.edge_frames, c.edge_frames + 1))
                t0 = e if rng.random() < 0.5 else tr.n_frames - c.out_frames + e
            else:
                t0 = int(rng.integers(-c.out_frames // 2, tr.n_frames - c.out_frames // 2))
            vr, vc = np.nonzero(tr.valid)  # centre windows on valid pixels, where the data are
            k = rng.integers(len(vr))
            r0, c0 = int(vr[k]) - c.out_px // 2 - c.s_reach, int(vc[k]) - c.out_px // 2 - c.s_reach
            t0 -= c.t_reach
            gain = float(st["gain"])
            if use_sim:
                tp = self._template(fn)
                w = window(tp, draw(tp, self.sim, rng), t0, nt, r0, ns, c0, ns, rng)
                movie, target, noisy = w["movie"], w["truth"], w["noisy"]
                mean_image, var_map, varfac, mean_act = w["mean_image"], w["var_map"], w["varfac"], w["mean_act"]
                labeled, support = w["labeled"], w["support"]
                gain = float(w.get("gain", gain))  # the virtual trial's (SimConfig.photon_gain)
                weight = np.isfinite(movie).astype(np.float32)
                if c.sim_event_weight != 1:
                    sd = np.sqrt(np.maximum(gain * np.nan_to_num(mean_image), 1e-6))
                    ev = _dilate_time(w["activity"] > 0.5 * sd[None], c.event_margin)
                    weight *= np.where(ev, np.float32(c.sim_event_weight), np.float32(1.0))
            else:
                p = tr.read(t0, nt, r0, ns, c0, ns)
                movie, mean_image, varfac = p.movie, p.mean_image, p.varfac
                labeled, support = p.labeled, p.support
                target = noisy = p.movie - p.activity
                var_map = np.where(np.isfinite(mean_image), _crop(st["var_movie"], r0, ns, c0, ns), np.nan)
                mean_act = np.where(np.isfinite(mean_image), _crop(st["mean_act"], r0, ns, c0, ns), np.nan)
                if use_sub:  # the residual as input, with its own whole-trial maps; no events left to see
                    unf = st.get("unfit")  # GIAnT fit artifacts: the movie there (their residual is hugely negative)
                    movie = target if unf is None else np.where(_crop(unf.astype(np.float32), r0, ns, c0, ns, 0.0)[None] > 0.5,
                                                                p.movie, target)
                    inside = np.isfinite(mean_image)
                    mean_image = np.where(inside, _crop(st["mean_res"], r0, ns, c0, ns), np.nan)
                    var_map = np.where(inside, _crop(st["var_res"], r0, ns, c0, ns), np.nan)
                    mean_act = np.where(inside, 0.0, np.nan).astype(np.float32)
                weight = np.isfinite(target).astype(np.float32)
                w_unf = c.w_unfitted if tr.attrs.get("nSources", 1) else 1.0  # no sources: no activity
                weight *= np.where(labeled & ~support, w_unf, 1.0).astype(np.float32)[None]
                unfit = st.get("unfit")
                if unfit is not None and unfit.any():  # GIAnT fit artifacts (stats.unfit_pixels): no loss
                    weight *= (_crop(unfit.astype(np.float32), r0, ns, c0, ns, 0.0) < 0.5).astype(np.float32)[None]
                if c.real_edge_s > 0:  # trial edges: GIAnT's baseline is biased there
                    e = int(round(c.real_edge_s * float(tr.attrs.get("alignHz", 80.0))))
                    tt = t0 + np.arange(nt)  # the trial frame of each window frame
                    weight[(tt < e) | (tt >= tr.n_frames - e)] = 0
                if c.event_mask > 0:
                    rms = np.sqrt(np.maximum(_crop(st["mean_act2"], r0, ns, c0, ns, 0.0), 0))
                    ev = np.nan_to_num(p.activity) > c.event_mask * np.maximum(rms, 1e-6)[None]
                    weight *= ~_dilate_time(ev & (rms > 0)[None], c.event_margin)
            if weight[sl].mean() >= c.min_valid_fraction:
                break
        if self.flip != "none":
            axes = [ax for ax in ((1, 2) if self.flip == "both" else (1,)) if rng.random() < 0.5]
            if axes:
                f3 = lambda a: np.ascontiguousarray(np.flip(a, axes))  # noqa: E731
                f2 = lambda a: np.ascontiguousarray(np.flip(a, [ax - 1 for ax in axes]))  # noqa: E731
                movie, target, noisy, weight = f3(movie), f3(target), f3(noisy), f3(weight)
                mean_image, var_map, varfac, mean_act, labeled, support = (
                    f2(a) for a in (mean_image, var_map, varfac, mean_act, labeled, support))
        s = np.float32(tr.scale)
        return {
            "movie": torch.from_numpy(movie.astype(np.float32)),
            "mean_image": torch.from_numpy(mean_image.astype(np.float32)),
            "varfac": torch.from_numpy(varfac.astype(np.float32)),
            "var_map": torch.from_numpy(var_map.astype(np.float32)),
            "mean_act": torch.from_numpy(mean_act.astype(np.float32)),
            "scale": torch.tensor(s),
            "gain": torch.tensor(np.float32(gain)),
            "target": torch.from_numpy(np.nan_to_num(target / s).astype(np.float32)),
            "noisy": torch.from_numpy(np.nan_to_num(noisy / s).astype(np.float32)),
            "weight": torch.from_numpy(weight * np.isfinite(target) * np.isfinite(noisy)),
            "support": torch.from_numpy(support.astype(np.uint8)),
            "labeled": torch.from_numpy(labeled.astype(np.uint8)),
            "sim": torch.tensor(np.float32(use_sim)),  # 1: simulated (target = the true baseline)
        }


def _dilate_time(m: np.ndarray, k: int) -> np.ndarray:
    """m (T, ...) bool, true within k frames of a true voxel."""
    if k <= 0 or not m.any():
        return m
    out = m.copy()
    for d in range(1, k + 1):
        out[d:] |= m[:-d]
        out[:-d] |= m[d:]
    return out


def loss_region(cfg: WindowConfig) -> tuple[slice, slice, slice]:
    """Index of the loss region within a window (frames, rows, cols)."""
    return (slice(cfg.t_reach, cfg.t_reach + cfg.out_frames), slice(cfg.s_reach, cfg.s_reach + cfg.out_px),
            slice(cfg.s_reach, cfg.s_reach + cfg.out_px))
