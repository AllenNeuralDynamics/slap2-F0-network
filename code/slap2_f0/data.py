"""Random training windows from exported trials.

A window covers ``out_frames`` frames and ``out_px`` x ``out_px`` pixels where
the loss is computed, plus the model's temporal and spatial reach on every
side, so every output sees real data (or a real trial edge) in its context,
never an artificial window border. Windows may run past the trial's ends,
which teaches the model the trial edges.

Loss weights (see docs/plan.md):
    valid pixels and frames only;
    footprint-support pixels and unlabeled (off-dendrite) pixels weight 1;
    labeled pixels outside every footprint weight ``w_unfitted`` (default 0),
    because their residual still holds activity GIAnT did not fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import TrialFile


@dataclass
class WindowConfig:
    out_frames: int = 64
    out_px: int = 32
    t_reach: int = 0  # set from the model: gap + 1 + temporal_reach
    s_reach: int = 0  # set from the model: spatial_reach
    w_unfitted: float = 0.0
    min_valid_fraction: float = 0.05  # of loss pixels


class WindowDataset(Dataset):
    """Infinite-style dataset: each item is a random window from a random trial."""

    def __init__(self, files, cfg: WindowConfig, length: int = 10000, seed: int = 0):
        self.files = [str(f) for f in files]
        self.cfg = cfg
        self.length = length
        self.seed = seed
        self._open: dict[str, TrialFile] = {}

    def __len__(self) -> int:
        return self.length

    def _trial(self, fn: str) -> TrialFile:
        if fn not in self._open:
            self._open[fn] = TrialFile(fn)
        return self._open[fn]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng((self.seed, idx))
        c = self.cfg
        for _ in range(50):
            tr = self._trial(self.files[rng.integers(len(self.files))])
            t0 = int(rng.integers(-c.out_frames // 2, tr.n_frames - c.out_frames // 2))
            # centre windows on valid pixels, where the data are
            vr, vc = np.nonzero(tr.valid)
            k = rng.integers(len(vr))
            r0 = int(vr[k]) - c.out_px // 2
            c0 = int(vc[k]) - c.out_px // 2
            p = tr.read(t0 - c.t_reach, c.out_frames + 2 * c.t_reach, r0 - c.s_reach, c.out_px + 2 * c.s_reach,
                        c0 - c.s_reach, c.out_px + 2 * c.s_reach)
            target = p.movie - p.activity
            w = np.isfinite(target).astype(np.float32)
            pix_w = np.where(p.labeled & ~p.support, c.w_unfitted, 1.0).astype(np.float32)
            w *= pix_w[None]
            sl = (slice(c.t_reach, c.t_reach + c.out_frames), slice(c.s_reach, c.s_reach + c.out_px),
                  slice(c.s_reach, c.s_reach + c.out_px))
            if w[sl].mean() >= c.min_valid_fraction:
                break
        return {
            "movie": torch.from_numpy(p.movie),
            "mean_image": torch.from_numpy(p.mean_image),
            "varfac": torch.from_numpy(p.varfac),
            "scale": torch.tensor(tr.scale, dtype=torch.float32),
            "target": torch.from_numpy(np.nan_to_num(target) / tr.scale),
            "weight": torch.from_numpy(w),
        }


def loss_region(cfg: WindowConfig) -> tuple[slice, slice, slice]:
    """Index of the loss region within a window (frames, rows, cols)."""
    return (slice(cfg.t_reach, cfg.t_reach + cfg.out_frames), slice(cfg.s_reach, cfg.s_reach + cfg.out_px),
            slice(cfg.s_reach, cfg.s_reach + cfg.out_px))
