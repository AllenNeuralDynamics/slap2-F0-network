"""Blind-spot networks for the baseline (F0) and noise of SLAP2 movies.

The output at frame t never depends on any input frame within ``gap`` frames
of t (|t' - t| <= gap), at any pixel: a "slab" blind spot. It is built in,
not trained in, so one forward pass predicts every pixel and frame:

* a **past branch** sees frames <= t - gap - 1 through causal temporal
  convolutions followed by a shift of gap + 1 frames;
* a **future branch** does the same on the time-reversed input;
* both branches mix space freely within each frame (per-frame convolutions),
  which keeps the blind spot, since it only constrains time;
* the branches meet in 1x1x1 layers only.

With ``spatial_kernel=1`` the network sees only each pixel's own trace (v0 in
docs/plan.md). The full box blind spot (v2: pixels further than a radius
visible at the same time) adds four spatial half-plane branches and is not
implemented yet.

Outputs, in units of the trial scale (see ``io.TrialFile.scale``):
    mu      baseline, as mean_image * exp(correction), positive
    var     noise variance; in counts (gain * mu * scale + floor) * exp(correction),
            so a single gain fits trials of different brightness
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

N_INPUTS = 4  # movie, valid flag, mean image, variance factor


@dataclass
class ModelConfig:
    gap: int = 8  # frames hidden on each side of the output frame
    width: int = 32  # feature channels per branch
    temporal_dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    # per block; blocks after full_res_blocks run at 1/downsample resolution, so
    # their dilations count double in pixels (default reach: 23 px)
    spatial_dilations: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 2, 2, 4])
    spatial_kernel: int = 3  # 1 = each pixel's own trace only
    n_inputs: int = N_INPUTS
    # memory: the first full_res_blocks blocks of each branch run at full
    # resolution, the rest at 1/downsample in space (per-frame down/upsampling,
    # so the blind spot in time is unaffected); checkpointing recomputes blocks
    # in the backward pass instead of storing their activations
    full_res_blocks: int = 2
    downsample: int = 2
    checkpoint: bool = True

    @property
    def spatial_factor(self) -> int:
        """Spatial downsampling actually used (none for the own-trace model)."""
        return 1 if self.spatial_kernel == 1 else max(1, self.downsample)

    @property
    def temporal_reach(self) -> int:
        """Frames of context on each side beyond the blind spot."""
        return 2 * sum(self.temporal_dilations)

    @property
    def spatial_reach(self) -> int:
        """Pixels of context on each side (an upper bound)."""
        if self.spatial_kernel == 1:
            return 0
        k = self.spatial_kernel // 2
        s = self.spatial_dilations
        reach = 0
        for i in range(len(self.temporal_dilations)):
            reach += k * s[min(i, len(s) - 1)] * (1 if i < self.full_res_blocks else self.spatial_factor)
        return reach + (self.spatial_factor - 1)


class ChannelNorm(nn.Module):
    """LayerNorm over the channel axis at each (frame, row, col)."""

    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, width, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, width, 1, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.mean(1, keepdim=True)
        v = x.var(1, keepdim=True, unbiased=False)
        return (x - m) / torch.sqrt(v + self.eps) * self.weight + self.bias


class CausalBlock(nn.Module):
    """Per-frame spatial conv, then a causal dilated temporal conv, residual."""

    def __init__(self, width: int, t_dil: int, s_dil: int, s_kernel: int):
        super().__init__()
        self.t_pad = 2 * t_dil  # kernel 3 in time
        if s_kernel > 1:
            pad = (s_kernel // 2) * s_dil
            self.spatial = nn.Conv3d(width, width, (1, s_kernel, s_kernel), padding=(0, pad, pad), dilation=(1, s_dil, s_dil))
        else:
            self.spatial = nn.Conv3d(width, width, 1)
        self.temporal = nn.Conv3d(width, width, (3, 1, 1), dilation=(t_dil, 1, 1))
        # normalize over channels at each position only: statistics pooled over
        # frames (GroupNorm, BatchNorm) would leak hidden frames into the output
        self.norm = ChannelNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T, H, W)
        h = F.gelu(self.spatial(x))
        h = F.pad(h, (0, 0, 0, 0, self.t_pad, 0))  # left (past) padding only
        h = self.temporal(h)
        return x + F.gelu(self.norm(h))


class CausalBranch(nn.Module):
    """Output at t depends on inputs at frames <= t - shift only.

    The first ``full_res_blocks`` blocks run at full resolution; the rest run
    on a per-frame strided downsampling and are upsampled (nearest, per frame)
    and added back. Every spatial operation acts within a frame, so causality
    in time is kept exactly."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        w, s_dils, f = cfg.width, cfg.spatial_dilations, cfg.spatial_factor
        self.inp = nn.Conv3d(cfg.n_inputs, w, 1)
        blocks = [CausalBlock(w, d, s_dils[min(i, len(s_dils) - 1)], cfg.spatial_kernel)
                  for i, d in enumerate(cfg.temporal_dilations)]
        n_full = len(blocks) if f == 1 else min(cfg.full_res_blocks, len(blocks))
        self.full = nn.ModuleList(blocks[:n_full])
        self.low = nn.ModuleList(blocks[n_full:])
        self.down = nn.Conv3d(w, w, (1, f, f), stride=(1, f, f)) if len(self.low) else None
        self.shift = cfg.gap + 1

    def _run(self, block: nn.Module, h: torch.Tensor) -> torch.Tensor:
        if self.cfg.checkpoint and self.training and h.requires_grad:
            return torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
        return block(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for b in self.full:
            h = self._run(b, h)
        if self.down is not None:
            T, H, W = h.shape[2:]
            f = self.cfg.spatial_factor
            hp = F.pad(h, (0, (-W) % f, 0, (-H) % f))
            low = self.down(hp)
            for b in self.low:
                low = self._run(b, low)
            up = F.interpolate(low, size=(T, hp.shape[3], hp.shape[4]), mode="nearest")
            h = h + up[..., :H, :W]
        # shift forward in time: out[t] = h[t - shift]
        h = F.pad(h, (0, 0, 0, 0, self.shift, 0))[:, :, : x.shape[2]]
        return h


class SlabBlindSpotNet(nn.Module):
    """Past and future causal branches fused pointwise; see the module docstring."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        w = self.cfg.width
        self.past = CausalBranch(self.cfg)
        self.future = CausalBranch(self.cfg)
        self.fuse = nn.Sequential(nn.Conv3d(2 * w, 2 * w, 1), nn.GELU(), nn.Conv3d(2 * w, w, 1), nn.GELU())
        self.head_mu = nn.Conv3d(w, 1, 1)
        self.head_var = nn.Conv3d(w, 1, 1)
        nn.init.zeros_(self.head_mu.weight)
        nn.init.zeros_(self.head_mu.bias)
        nn.init.zeros_(self.head_var.weight)
        nn.init.zeros_(self.head_var.bias)
        # noise model in movie counts: var = (gain * mu + floor) * exp(s). The
        # movie's variance is about 6.6 x its mean on 780125 (photons times
        # detector gain, with the 80 Hz kernel); start there.
        self.log_gain = nn.Parameter(torch.tensor(math.log(6.6)))
        self.log_floor = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, scale: torch.Tensor | None = None,
                region: tuple[slice, slice, slice] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, n_inputs, T, H, W) from ``make_inputs``; scale (B,) the trial
        scale the inputs were divided by; region (frames, rows, cols) limits the
        output to that part of the window (training needs only the loss region,
        and the pointwise layers then run only there). Returns mu and var, each
        (B, T', H', W'), in units of the trial scale. The noise model is applied
        in counts (mu * scale), so one gain fits trials of any brightness."""
        hp = self.past(x)
        hf = self.future(x.flip(2)).flip(2)
        if region is not None:
            sel = (slice(None), slice(None)) + tuple(region)
            hp, hf, x = hp[sel], hf[sel], x[sel]
        h = self.fuse(torch.cat([hp, hf], 1))
        mean_image = x[:, 2:3].clamp_min(1e-3)
        mu = mean_image * torch.exp(self.head_mu(h).clamp(-5, 5))
        s = torch.ones(x.shape[0], device=x.device, dtype=mu.dtype) if scale is None else scale.to(mu.dtype)
        s = s.view(-1, 1, 1, 1, 1)
        var_counts = self.log_gain.exp() * mu * s + self.log_floor.exp()
        var = var_counts / s ** 2 * torch.exp(self.head_var(h).clamp(-8, 8))
        return mu[:, 0], var[:, 0]


def make_inputs(movie: torch.Tensor, mean_image: torch.Tensor, varfac: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Network inputs from a batch of windows.

    movie (B, T, H, W) with NaN where there is no data; mean_image and varfac
    (B, H, W); scale (B,). Returns (B, 4, T, H, W): movie / scale with missing
    values as 0, a validity flag, mean image / scale, and the variance factor
    relative to its median. The mean image is a whole-trial average, so it
    contains each frame with weight 1/T; that leak through the blind spot is
    negligible for trials of thousands of frames.
    """
    s = scale.view(-1, 1, 1, 1)
    valid = torch.isfinite(movie)
    m = torch.where(valid, movie, torch.zeros_like(movie)) / s
    mi = torch.nan_to_num(mean_image / s.squeeze(1), nan=0.0)
    vf = varfac / varfac.flatten(1).median(1).values.clamp_min(1e-6).view(-1, 1, 1)
    T = movie.shape[1]
    return torch.stack(
        [m, valid.float(), mi.unsqueeze(1).expand(-1, T, -1, -1), vf.unsqueeze(1).expand(-1, T, -1, -1)], 1
    )
