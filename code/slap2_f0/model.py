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

Inputs to the branches (``make_inputs``): the movie, its validity flag and
three whole-trial maps (mean image, SiLo's variance factor, and the trial's
variance relative to photon noise, which marks pixels with events). The
whole-trial maps contain every frame with weight 1/T, a negligible leak;
being constant in time they cannot say when an event happened.

At the output frame only, the fusion layers also get blind long-window
statistics of the movie: its mean and variance over gap < |dt| <= gap +
window_reach. They are computed inside the model (so the blind-spot test
covers them) and are not fed to the branches, because a context frame's
window would reach into the hidden frames.

Outputs, in units of the trial scale (see ``io.TrialFile.scale``):
    mu      baseline = base * exp(correction), where base = mean image +
            mix * (window mean - mean image); correction and mix start at 0,
            so the untrained network predicts the mean image exactly
    var     noise variance; in counts (gain * mu * scale + floor) *
            exp(correction), with the trial's measured gain (``stats``)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

N_INPUTS = 5  # movie, valid flag, mean image, variance factor, log variance ratio
N_FEATS = 6  # output-frame features for the fusion layers (SlabBlindSpotNet._features)


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
    window_reach: int = 254  # frames beyond the gap for the blind window statistics
    # memory: the first full_res_blocks blocks of each branch run at full
    # resolution, the rest at 1/downsample in space (per-frame down/upsampling,
    # so the blind spot in time is unaffected); checkpointing recomputes blocks
    # in the backward pass instead of storing their activations
    full_res_blocks: int = 1
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


class CausalBlock(nn.Module):
    """Per-frame spatial conv, then a causal dilated temporal conv, residual.

    The residual branch is scaled by a learnable factor that starts at 0
    (ReZero), which keeps deep stacks stable without a normalization layer.
    Normalizations that pool over frames (GroupNorm, BatchNorm) would leak
    hidden frames into the output, and a per-position channel norm costs
    several full-size temporaries per block."""

    def __init__(self, width: int, t_dil: int, s_dil: int, s_kernel: int):
        super().__init__()
        self.t_pad = 2 * t_dil  # kernel 3 in time
        if s_kernel > 1:
            pad = (s_kernel // 2) * s_dil
            self.spatial = nn.Conv3d(width, width, (1, s_kernel, s_kernel), padding=(0, pad, pad), dilation=(1, s_dil, s_dil))
        else:
            self.spatial = nn.Conv3d(width, width, 1)
        self.temporal = nn.Conv3d(width, width, (3, 1, 1), dilation=(t_dil, 1, 1))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T, H, W)
        h = F.gelu(self.spatial(x))
        h = F.pad(h, (0, 0, 0, 0, self.t_pad, 0))  # left (past) padding only
        return x + self.alpha * F.gelu(self.temporal(h))


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


def blind_window_stats(m: torch.Tensor, v: torch.Tensor, gap: int, reach: int):
    """Mean, variance and count of m over frames gap < |dt| <= gap + reach.

    m, v: (B, T, H, W), v the validity (0/1), m zero where invalid. Returns
    mean and variance (zero where there are no frames) and the count."""
    T = m.shape[1]
    m, v = m.float(), v.float()
    cs = [F.pad(x.cumsum(1), (0, 0, 0, 0, 1, 0)) for x in (m * v, m * m * v, v)]
    t = torch.arange(T, device=m.device)
    a1, b1 = (t - gap - reach).clamp(0, T), (t - gap).clamp(0, T)
    a2, b2 = (t + gap + 1).clamp(0, T), (t + gap + reach + 1).clamp(0, T)
    s1, s2, n = [c[:, b1] - c[:, a1] + c[:, b2] - c[:, a2] for c in cs]
    nn_ = n.clamp_min(1)
    mean = s1 / nn_
    var = (s2 / nn_ - mean ** 2).clamp_min(0)
    return mean, var, n


class SlabBlindSpotNet(nn.Module):
    """Past and future causal branches fused pointwise; see the module docstring."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        w = self.cfg.width
        self.past = CausalBranch(self.cfg)
        self.future = CausalBranch(self.cfg)
        self.fuse = nn.Sequential(nn.Conv3d(2 * w + N_FEATS, 2 * w, 1), nn.GELU(), nn.Conv3d(2 * w, w, 1), nn.GELU())
        self.head_mu = nn.Conv3d(w, 1, 1)
        self.head_mix = nn.Conv3d(w, 1, 1)
        self.head_var = nn.Conv3d(w, 1, 1)
        for head in (self.head_mu, self.head_mix, self.head_var):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.log_floor = nn.Parameter(torch.tensor(0.0))  # variance floor, counts^2

    def _features(self, x, wm, wv, n, gs):
        """Output-frame features, all O(1): window mean and variance relative to
        the mean image and to photon noise, the fraction of the window with
        data, log mean image, log variance ratio and variance factor."""
        mi = x[:, 2].clamp_min(1e-3)
        has = n > 0
        f1 = torch.where(has, torch.log(wm.clamp_min(1e-3) / mi), torch.zeros_like(mi))
        f2 = torch.where(has, torch.log((wv + 1e-6) / (gs * wm.clamp_min(1e-3))), torch.zeros_like(mi))
        f3 = n / (2 * self.cfg.window_reach)
        return torch.stack([f1.clamp(-3, 3), f2.clamp(-3, 5), f3, torch.log(mi), x[:, 4], x[:, 3]], 1)

    def forward(self, x: torch.Tensor, scale: torch.Tensor | None = None, gain: torch.Tensor | None = None,
                region: tuple[slice, slice, slice] | None = None, aux: bool = False):
        """x: (B, n_inputs, T, H, W) from ``make_inputs``; scale (B,) the trial
        scale the inputs were divided by; gain (B,) the trial's variance/mean
        ratio in counts; region (frames, rows, cols) limits the output to that
        part of the window (training needs only the loss region, and the
        pointwise layers then run only there). Returns mu and var, each (B, T',
        H', W'), in units of the trial scale, and with aux=True a dict with the
        window mean, the base and the mix."""
        B = x.shape[0]
        s = torch.ones(B, device=x.device) if scale is None else scale.to(x.device).float()
        g = torch.full((B,), 10.0, device=x.device) if gain is None else gain.to(x.device).float()
        hp = self.past(x)
        hf = self.future(x.flip(2)).flip(2)
        wm, wv, n = blind_window_stats(x[:, 0], x[:, 1], self.cfg.gap, self.cfg.window_reach)
        if region is not None:
            sel = (slice(None), slice(None)) + tuple(region)
            hp, hf, x = hp[sel], hf[sel], x[sel]
            wm, wv, n = (a[(slice(None),) + tuple(region)] for a in (wm, wv, n))
        gs = (g / s).view(-1, 1, 1, 1)  # gain in scale units: var = gs * mu
        feats = self._features(x.float(), wm, wv, n, gs).to(hp.dtype)
        h = self.fuse(torch.cat([hp, hf, feats], 1))
        mi = x[:, 2].float()
        wm = torch.where(n > 0, wm, mi)
        mix = self.head_mix(h)[:, 0].float()
        base = (mi + mix * (wm - mi)).clamp_min(1e-4)
        mu = base * torch.exp(self.head_mu(h)[:, 0].float().clamp(-5, 5))
        var = (gs * mu + self.log_floor.exp() / s.view(-1, 1, 1, 1) ** 2) * \
            torch.exp(self.head_var(h)[:, 0].float().clamp(-4, 4))
        if aux:
            return mu, var, {"window_mean": wm, "base": base, "mix": mix}
        return mu, var


def make_inputs(movie: torch.Tensor, mean_image: torch.Tensor, varfac: torch.Tensor, scale: torch.Tensor,
                var_map: torch.Tensor | None = None, gain: torch.Tensor | None = None) -> torch.Tensor:
    """Network inputs from a batch of windows.

    movie (B, T, H, W) in counts with NaN where there is no data; mean_image,
    varfac and var_map (B, H, W), the last a whole-trial map in counts^2;
    scale and gain (B,). Returns (B, 5, T, H, W): movie / scale with missing
    values as 0, a validity flag, mean image / scale, the variance factor
    relative to its median, and log(var_map / (gain * mean image)), about 0
    where photon noise explains the variance and positive where events add to
    it. The mean image is a whole-trial average, so it contains each frame
    with weight 1/T; that leak through the blind spot is negligible for trials
    of thousands of frames.
    """
    B, T = movie.shape[:2]
    s = scale.view(-1, 1, 1, 1)
    valid = torch.isfinite(movie)
    m = torch.where(valid, movie, torch.zeros_like(movie)) / s
    mi = torch.nan_to_num(mean_image / s.squeeze(1), nan=0.0)
    vf = varfac / varfac.flatten(1).median(1).values.clamp_min(1e-6).view(-1, 1, 1)
    if var_map is None:
        vr = torch.zeros_like(mi)
    else:
        g = torch.full((B,), 10.0, device=movie.device) if gain is None else gain
        ratio = var_map / (g.view(-1, 1, 1) * mean_image.clamp_min(1e-3))
        vr = torch.nan_to_num(torch.log(ratio.clamp_min(1e-3)), nan=0.0).clamp(-3, 5)
    return torch.stack([m, valid.float()] + [a.unsqueeze(1).expand(-1, T, -1, -1) for a in (mi, vf, vr)], 1)
