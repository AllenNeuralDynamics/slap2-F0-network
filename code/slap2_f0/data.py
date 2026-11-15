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
residual still holds activity GIAnT did not fit.

Simulated windows (``sim=SimConfig()``): each item is a new virtual trial
built on a random template (see ``simulate``); the target is the noise-free
baseline, and every valid pixel and frame has weight 1.

Every item also carries ``noisy`` (movie - activity, for the noise head), the
whole-trial variance map and the trial's gain (``stats``), and the support and
labeled masks for evaluation. With ``flip`` the window is flipped at random
along rows and/or columns (the noise is symmetric under flips).
"""

from __future__ import annotations

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


def _crop(a: np.ndarray, r0: int, nr: int, c0: int, nc: int, fill=np.nan) -> np.ndarray:
    out = np.full((nr, nc), fill, np.float32)
    R, C = a.shape
    rs, re, cs, ce = max(r0, 0), min(r0 + nr, R), max(c0, 0), min(c0 + nc, C)
    if rs < re and cs < ce:
        out[rs - r0: re - r0, cs - c0: ce - c0] = a[rs:re, cs:ce]
    return out


class WindowDataset(Dataset):
    """Each item is a random window from a random trial (real or simulated)."""

    def __init__(self, files, cfg: WindowConfig, stats: dict, length: int = 10000, seed: int = 0,
                 sim: SimConfig | None = None, flip: bool = True):
        self.files = [str(f) for f in files]
        self.cfg = cfg
        self.stats = {str(k): v for k, v in stats.items()}
        self.length = length
        self.seed = seed
        self.sim = sim
        self.flip = flip
        self._open: dict[str, TrialFile] = {}
        self._tpl: dict[str, Template] = {}

    def __len__(self) -> int:
        return self.length

    def _trial(self, fn: str) -> TrialFile:
        if fn not in self._open:
            self._open[fn] = TrialFile(fn)
        return self._open[fn]

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
        for _ in range(50):
            fn = self.files[rng.integers(len(self.files))]
            tr, st = self._trial(fn), self.stats[fn]
            t0 = int(rng.integers(-c.out_frames // 2, tr.n_frames - c.out_frames // 2))
            vr, vc = np.nonzero(tr.valid)  # centre windows on valid pixels, where the data are
            k = rng.integers(len(vr))
            r0, c0 = int(vr[k]) - c.out_px // 2 - c.s_reach, int(vc[k]) - c.out_px // 2 - c.s_reach
            t0 -= c.t_reach
            if self.sim is not None:
                tp = self._template(fn)
                w = window(tp, draw(tp, self.sim, rng), t0, nt, r0, ns, c0, ns, rng)
                movie, target, noisy = w["movie"], w["truth"], w["noisy"]
                mean_image, var_map, varfac, mean_act = w["mean_image"], w["var_map"], w["varfac"], w["mean_act"]
                labeled, support = w["labeled"], w["support"]
                weight = np.isfinite(movie).astype(np.float32)
            else:
                p = tr.read(t0, nt, r0, ns, c0, ns)
                movie, mean_image, varfac = p.movie, p.mean_image, p.varfac
                labeled, support = p.labeled, p.support
                target = noisy = p.movie - p.activity
                var_map = np.where(np.isfinite(mean_image), _crop(st["var_movie"], r0, ns, c0, ns), np.nan)
                mean_act = np.where(np.isfinite(mean_image), _crop(st["mean_act"], r0, ns, c0, ns), np.nan)
                weight = np.isfinite(target).astype(np.float32)
                weight *= np.where(labeled & ~support, c.w_unfitted, 1.0).astype(np.float32)[None]
            if weight[sl].mean() >= c.min_valid_fraction:
                break
        if self.flip:
            axes = [ax for ax in (1, 2) if rng.random() < 0.5]
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
            "gain": torch.tensor(np.float32(st["gain"])),
            "target": torch.from_numpy(np.nan_to_num(target / s).astype(np.float32)),
            "noisy": torch.from_numpy(np.nan_to_num(noisy / s).astype(np.float32)),
            "weight": torch.from_numpy(weight * np.isfinite(target) * np.isfinite(noisy)),
            "support": torch.from_numpy(support.astype(np.uint8)),
            "labeled": torch.from_numpy(labeled.astype(np.uint8)),
        }


def loss_region(cfg: WindowConfig) -> tuple[slice, slice, slice]:
    """Index of the loss region within a window (frames, rows, cols)."""
    return (slice(cfg.t_reach, cfg.t_reach + cfg.out_frames), slice(cfg.s_reach, cfg.s_reach + cfg.out_px),
            slice(cfg.s_reach, cfg.s_reach + cfg.out_px))
