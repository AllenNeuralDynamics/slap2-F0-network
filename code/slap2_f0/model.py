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
docs/plan.md).

**Box blind spot** (``box_radius`` >= 0, v6): the hidden set shrinks from the
slab to the box |dt| <= gap, |dr| <= box_radius, |dc| <= box_radius. Pixels
further away in row or column are visible at every frame, the output frame
included, so fast changes shared across the field (brightness, residual
motion, defocus) can be followed. Two additions see them:

* four **half-plane branches** (Laine et al. 2019): convolutions whose
  receptive field stays in rows <= r (kernel 2 x 3 with top padding, and an
  aligned half-resolution path), shifted down by box_radius + 1 rows; the
  four 90-degree rotations of the input share one trunk and together see
  everything outside the box. In time they are symmetric and short (fast
  changes; the slow ones are the slab branches' job), and they run only on
  the output frames plus their temporal reach;
* **pooled same-frame features**: for squares of half-width ``pool_scales``
  around each pixel minus the box, the fractional deviation of the movie from
  the mean image, sum(valid * (movie - mean image)) / sum(valid * mean image),
  in the output frame and averaged over the 8 frames around it (integral
  images, exact and cheap; 0 = the whole window).

Inputs to the branches (``make_inputs``): the movie, its validity flag and
three whole-trial maps (mean image, SiLo's variance factor relative to its
median over the trial's imaged pixels, and the trial's variance relative to
photon noise, which marks pixels with events). The
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

The mix is tanh-bounded to (-1, 1) and both corrections are soft-capped
(``max_log_correction``, ``max_log_var``): with unbounded heads, single
windows occasionally produced baselines tens of times too large (v2 run).
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
    max_log_correction: float = 0.7  # soft cap of |log| of the baseline correction (factor 0.5-2)
    max_log_var: float = 3.0  # soft cap of |log| of the noise correction
    # box blind spot (see the module docstring); -1 keeps the slab (v1-v5)
    box_radius: int = -1
    hp_width: int = 24
    hp_temporal_dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    hp_spatial_dilations: list[int] = field(default_factory=lambda: [1, 1, 1, 2, 4])
    hp_full_res_blocks: int = 2  # the rest at half resolution
    pool_scales: list[int] = field(default_factory=lambda: [8, 16, 32, 0])
    # variance-factor input: "trial" = log of the factor relative to its median
    # over the trial's imaged pixels (``io.TrialFile``), the same in every
    # window; "window" (checkpoints up to v6) = relative to the median over the
    # window with unimaged pixels as 0, which is 0 whenever more than half of
    # the window is unimaged, so the input saturated at +3 in most windows
    varfac_norm: str = "trial"
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
    def box(self) -> bool:
        return self.box_radius >= 0

    @property
    def hp_temporal_reach(self) -> int:
        """Frames of context on each side of the half-plane branches."""
        return sum(self.hp_temporal_dilations)

    @property
    def hp_reach(self) -> int:
        """Pixels from an output to the furthest input of a half-plane branch
        (towards its half-plane; the other axis reaches less)."""
        d = self.hp_spatial_dilations
        nf = min(self.hp_full_res_blocks, len(d))
        low = d[nf:]
        return self.box_radius + 1 + sum(d[:nf]) + 2 * sum(low) + (1 if low else 0)

    @property
    def spatial_reach(self) -> int:
        """Pixels of context on each side (an upper bound) that a window must
        carry around its output region; the pooled features use whatever the
        window holds."""
        if self.spatial_kernel == 1:
            reach = 0
        else:
            k = self.spatial_kernel // 2
            s = self.spatial_dilations
            reach = 0
            for i in range(len(self.temporal_dilations)):
                reach += k * s[min(i, len(s) - 1)] * (1 if i < self.full_res_blocks else self.spatial_factor)
            reach += self.spatial_factor - 1
        return max(reach, self.hp_reach) if self.box else reach

    @property
    def n_pool(self) -> int:
        return 2 * len(self.pool_scales) if self.box else 0


def _residual(x: torch.Tensor, alpha: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """x + alpha * h; in place in inference when the dtypes agree (a model
    cast to half precision), which saves a full-size temporary per block."""
    if not torch.is_grad_enabled() and x.dtype == h.dtype == alpha.dtype:
        return x.add_(h.mul_(alpha))
    return x + alpha * h


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
        # past-only padding: padding both sides and keeping the first T outputs
        # is the same and saves a padded copy of h
        h = F.conv3d(h, self.temporal.weight, self.temporal.bias, padding=(self.t_pad, 0, 0),
                     dilation=self.temporal.dilation)[:, :, : x.shape[2]]
        return _residual(x, self.alpha, F.gelu(h))


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


class HalfPlaneBlock(nn.Module):
    """Spatial conv that sees rows r - dil and r only (kernel 2 x 3, top
    padding; columns symmetric), then a symmetric dilated temporal conv,
    residual with a ReZero scale."""

    def __init__(self, width: int, t_dil: int, s_dil: int):
        super().__init__()
        self.t_dil, self.s_dil = t_dil, s_dil
        self.spatial = nn.Conv3d(width, width, (1, 2, 3), dilation=(1, s_dil, s_dil))
        self.temporal = nn.Conv3d(width, width, (3, 1, 1), dilation=(t_dil, 1, 1))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T, H, W)
        d = self.s_dil
        # cols d | d, rows d on top only: padding rows on both sides and keeping
        # the first H outputs is the same and saves a padded copy of x
        h = F.conv3d(x, self.spatial.weight, self.spatial.bias, padding=(0, d, d),
                     dilation=self.spatial.dilation)[:, :, :, : x.shape[3]]
        h = F.conv3d(F.gelu(h), self.temporal.weight, self.temporal.bias, padding=(self.t_dil, 0, 0),
                     dilation=self.temporal.dilation)
        return _residual(x, self.alpha, F.gelu(h))


class HalfPlaneBranch(nn.Module):
    """Output at row r depends on inputs at rows <= r - (box_radius + 1) only,
    any column, frames within the temporal reach.

    Half resolution keeps the half-plane exact: a strided 2 x 2 conv makes low
    row i from rows 2i and 2i + 1; after the low blocks (which see low rows
    <= i), nearest upsampling and a one-row shift give full row r the low row
    (r - 1) // 2, which saw rows <= r only. The input height must be even."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.hp_width
        self.inp = nn.Conv3d(cfg.n_inputs, w, 1)
        blocks = [HalfPlaneBlock(w, td, sd) for td, sd in zip(cfg.hp_temporal_dilations, cfg.hp_spatial_dilations)]
        nf = min(cfg.hp_full_res_blocks, len(blocks))
        self.full = nn.ModuleList(blocks[:nf])
        self.low = nn.ModuleList(blocks[nf:])
        self.down = nn.Conv3d(w, w, (1, 2, 2), stride=(1, 2, 2)) if len(self.low) else None
        self.shift = cfg.box_radius + 1

    def _run(self, block: nn.Module, h: torch.Tensor) -> torch.Tensor:
        if self.cfg.checkpoint and self.training and h.requires_grad:
            return torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
        return block(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[3:]
        h = self.inp(x)
        for b in self.full:
            h = self._run(b, h)
        if self.down is not None:
            low = self.down(h)
            for b in self.low:
                low = self._run(b, low)
            up = F.interpolate(low, size=h.shape[2:], mode="nearest")
            h = h + F.pad(up, (0, 0, 1, 0))[..., :H, :]  # row r <- low row (r - 1) // 2
        return F.pad(h, (0, 0, self.shift, 0))[..., :H, :]  # out[r] = h[r - shift]


def _box_sums(a: torch.Tensor, half: int) -> torch.Tensor:
    """Sums of a (N, H, W) over squares of half-width `half` around every
    pixel (clipped at the edges), from an integral image."""
    N, H, W = a.shape
    I = F.pad(a.cumsum(1).cumsum(2), (1, 0, 1, 0))
    r = torch.arange(H, device=a.device)
    c = torch.arange(W, device=a.device)
    r0, r1 = (r - half).clamp(0, H), (r + half + 1).clamp(0, H)
    c0, c1 = (c - half).clamp(0, W), (c + half + 1).clamp(0, W)
    return I[:, r1][:, :, c1] - I[:, r0][:, :, c1] - I[:, r1][:, :, c0] + I[:, r0][:, :, c0]


def pooled_features(x: torch.Tensor, t0: int, t1: int, radius: int, scales: list[int]) -> torch.Tensor:
    """Same-frame fractional deviation of the movie from the mean image over
    squares around each pixel minus its box (|dr|, |dc| <= radius), for frames
    t0..t1-1: (B, 2 * len(scales), t1 - t0, H, W), the single frame and the
    mean of frames t-3..t+4, each clamped to +-1. x as from ``make_inputs``."""
    B, _, T, H, W = x.shape
    a, b = max(0, t0 - 3), min(T, t1 + 4)
    m, v, mi = x[:, 0, a:b].float(), x[:, 1, a:b].float(), x[:, 2, a:b].float()
    num = (v * (m - mi)).reshape(-1, H, W)
    den = (v * mi.clamp_min(0)).reshape(-1, H, W)
    bn, bd = _box_sums(num, radius), _box_sums(den, radius)
    out = []
    for s in scales:
        half = s if s > 0 else max(H, W)
        sn = (_box_sums(num, half) - bn).view(B, b - a, H, W)
        sd = (_box_sums(den, half) - bd).view(B, b - a, H, W)
        r1 = torch.where(sd > 1e-6, sn / sd.clamp_min(1e-6), torch.zeros_like(sn))
        cn = F.pad(sn.cumsum(1), (0, 0, 0, 0, 1, 0))
        cd = F.pad(sd.cumsum(1), (0, 0, 0, 0, 1, 0))
        t = torch.arange(t0, t1, device=x.device)
        lo, hi = (t - 3).clamp(a, b) - a, (t + 5).clamp(a, b) - a
        n8, d8 = cn[:, hi] - cn[:, lo], cd[:, hi] - cd[:, lo]
        r8 = torch.where(d8 > 1e-6, n8 / d8.clamp_min(1e-6), torch.zeros_like(n8))
        out += [r1[:, t0 - a: t1 - a].clamp(-1, 1), r8.clamp(-1, 1)]
    return torch.stack(out, 1)


def soft_cap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """cap * tanh(x / cap): the identity near 0, smoothly bounded by +-cap."""
    return cap * torch.tanh(x / cap)


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
        self.hp = HalfPlaneBranch(self.cfg) if self.cfg.box else None
        n_in = 2 * w + N_FEATS + (4 * self.cfg.hp_width + self.cfg.n_pool if self.cfg.box else 0)
        self.fuse = nn.Sequential(nn.Conv3d(n_in, 2 * w, 1), nn.GELU(), nn.Conv3d(2 * w, w, 1), nn.GELU())
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
        data, log mean image, log variance ratio and log variance factor."""
        mi = x[:, 2].clamp_min(1e-3)
        has = n > 0
        f1 = torch.where(has, torch.log(wm.clamp_min(1e-3) / mi), torch.zeros_like(mi))
        f2 = torch.where(has, torch.log((wv + 1e-6) / (gs * wm.clamp_min(1e-3))), torch.zeros_like(mi))
        f3 = n / (2 * self.cfg.window_reach)
        return torch.stack([f1.clamp(-3, 3), f2.clamp(-3, 5), f3, torch.log(mi), x[:, 4], x[:, 3]], 1)

    def _half_planes(self, x: torch.Tensor) -> torch.Tensor:
        """The four half-plane views: (B, 4 * hp_width, T, H, W). The input is
        padded to even size, so the half-resolution grids of all rotations
        line up (row pairs 2i, 2i + 1 in the window's own coordinates)."""
        H, W = x.shape[3:]
        xp = F.pad(x, (0, W % 2, 0, H % 2))
        if xp.shape[3] == xp.shape[4] and torch.is_grad_enabled():  # square: one batched pass over the rotations
            # (in inference one rotation at a time, below: a quarter of the memory)
            h = self.hp(torch.cat([torch.rot90(xp, k, (3, 4)) for k in range(4)], 0))
            outs = [torch.rot90(hk, -k, (3, 4)) for k, hk in enumerate(h.chunk(4, 0))]
        else:
            outs = [torch.rot90(self.hp(torch.rot90(xp, k, (3, 4))), -k, (3, 4)) for k in range(4)]
        return torch.cat(outs, 1)[..., :H, :W]

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
        T = x.shape[2]
        sel = None if region is None else (slice(None), slice(None)) + tuple(region)
        hp = self.past(x)
        if sel is not None:  # crop each branch as soon as it is done (memory)
            hp = hp[sel]
        hf = self.future(x.flip(2))
        if sel is not None:  # crop in flipped time, then flip only the cropped part
            ta, tb, _ = region[0].indices(T)
            hf = hf[(slice(None), slice(None), slice(T - tb, T - ta)) + tuple(region[1:])]
        hf = hf.flip(2)
        wm, wv, n = blind_window_stats(x[:, 0], x[:, 1], self.cfg.gap, self.cfg.window_reach)
        extra = []
        if self.hp is not None:  # box blind spot: same-frame context outside the box
            t0 = 0 if region is None or region[0].start is None else region[0].start
            t1 = T if region is None or region[0].stop is None else region[0].stop
            R = self.cfg.hp_temporal_reach
            a, b = max(0, t0 - R), min(T, t1 + R)
            hh = self._half_planes(x[:, :, a:b])[:, :, t0 - a: t1 - a]
            pf = pooled_features(x, t0, t1, self.cfg.box_radius, self.cfg.pool_scales)
            if region is not None:
                sp = (slice(None), slice(None), slice(None)) + tuple(region[1:])
                hh, pf = hh[sp], pf[sp]
            extra = [hh, pf.to(hh.dtype)]
        if region is not None:
            x = x[sel]
            wm, wv, n = (a[(slice(None),) + tuple(region)] for a in (wm, wv, n))
        gs = (g / s).view(-1, 1, 1, 1)  # gain in scale units: var = gs * mu
        feats = self._features(x.float(), wm, wv, n, gs).to(hp.dtype)
        h = self.fuse(torch.cat([hp, hf, feats] + [e.to(hp.dtype) for e in extra], 1))
        mi = x[:, 2].float()
        wm = torch.where(n > 0, wm, mi)
        mix = torch.tanh(self.head_mix(h)[:, 0].float())
        base = (mi + mix * (wm - mi)).clamp_min(1e-4)
        mu = base * torch.exp(soft_cap(self.head_mu(h)[:, 0].float(), self.cfg.max_log_correction))
        var = (gs * mu + self.log_floor.exp() / s.view(-1, 1, 1, 1) ** 2) * \
            torch.exp(soft_cap(self.head_var(h)[:, 0].float(), self.cfg.max_log_var))
        if aux:
            return mu, var, {"window_mean": wm, "base": base, "mix": mix}
        return mu, var


def config_from_dict(d: dict) -> ModelConfig:
    """A checkpoint's ModelConfig; checkpoints saved before ``varfac_norm``
    existed used the window median."""
    d = dict(d)
    d.setdefault("varfac_norm", "window")
    return ModelConfig(**d)


def make_inputs(movie: torch.Tensor, mean_image: torch.Tensor, varfac: torch.Tensor, scale: torch.Tensor,
                var_map: torch.Tensor | None = None, gain: torch.Tensor | None = None,
                varfac_norm: str = "trial") -> torch.Tensor:
    """Network inputs from a batch of windows.

    movie (B, T, H, W) in counts with NaN where there is no data; mean_image,
    varfac and var_map (B, H, W), the last a whole-trial map in counts^2;
    scale and gain (B,). Returns (B, 5, T, H, W): movie / scale with missing
    values as 0, a validity flag, mean image / scale, the log variance factor
    (clamped to +-3; see ``ModelConfig.varfac_norm``: with "trial", varfac must
    already be relative to the trial's median, as ``io.TrialFile`` gives it,
    and pixels without a factor get 0), and log(var_map / (gain * mean image)), about 0
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
    if varfac_norm == "window":
        vf = varfac / varfac.flatten(1).median(1).values.clamp_min(1e-6).view(-1, 1, 1)
        vf = torch.nan_to_num(torch.log(vf.clamp_min(1e-3)), nan=0.0).clamp(-3, 3)
    elif varfac_norm == "trial":
        ok = torch.isfinite(varfac) & (varfac > 0)
        vf = torch.where(ok, torch.log(torch.where(ok, varfac, torch.ones_like(varfac))).clamp(-3, 3),
                         torch.zeros_like(varfac))
    else:
        raise ValueError(f"varfac_norm {varfac_norm!r}")
    if var_map is None:
        vr = torch.zeros_like(mi)
    else:
        g = torch.full((B,), 10.0, device=movie.device) if gain is None else gain
        ratio = var_map / (g.view(-1, 1, 1) * mean_image.clamp_min(1e-3))
        vr = torch.nan_to_num(torch.log(ratio.clamp_min(1e-3)), nan=0.0).clamp(-3, 5)
    return torch.stack([m, valid.float()] + [a.unsqueeze(1).expand(-1, T, -1, -1) for a in (mi, vf, vr)], 1)
