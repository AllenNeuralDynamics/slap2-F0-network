"""Streaming inference: the same baselines as ``predict.predict_trial`` with
far less work (slap2-F0-network#33).

``predict_trial`` runs the whole network on each tile's window: the training
tile (48 x 48 output pixels) plus the model's reach on every side (23 px and
263 frames). Neighbouring tiles' windows overlap, so most input voxels are
processed two to four times, and each causal branch also runs over the 263
frames of the window that it never uses (the past branch over the future
halo, the future branch over the past one). This engine computes each
quantity once:

* **Rectangles.** The tiles are the reference's (``predict.tile_origins``);
  the union of their windows is split into rectangles (connected parts,
  each its bounding box, split by row bands when it would hold too many
  pixels). Corners stay at the windows' parity and sizes even, so the
  half-resolution paths pair the same rows and columns as in every tile.
  Convolutions are translation-equivariant and every output pixel lies at
  least the reach inside its rectangle, so the branches give the tile's
  values there.
* **Time in chunks with state.** The causal branches run once over the
  trial in chunks, the past forwards from frame -t_reach and the future
  backwards from T - 1 + t_reach, each block keeping the last 2 x dilation
  frames of its temporal convolution's input (zeros at the start, as the
  window's padding). This is the same computation as one window over the
  whole trial, without a temporal halo and in bounded memory. The future
  branch runs first; its output at the output pixels is kept (GPU, or host
  memory when it does not fit).
* **Same-frame parts per chunk.** The half-plane branches (symmetric in
  time, reach 31 frames) run per chunk with that halo. The pooled (and
  motion) features depend on the window's geometry beyond the reach (half-
  widths 32 and the whole window), so they are computed per tile on the
  tile's own window, as in the reference. Window statistics and edge
  features are per pixel in time and computed at the output pixels only.
* **Fusion at the output pixels only**, as a flat list of pixels (the
  fusion layers are pointwise), in blocks of pixels.

Numerics: the same operations in the same precision (the model cast to
float16 and float16 autocast, as ``predict.prepare``), but other shapes, so
cuDNN may choose other algorithms and float sums run in another order: the
outputs agree with the reference to float16 rounding, not bit for bit.
``tests/test_stream.py`` checks the equality in float32 on the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .model import (SlabBlindSpotNet, _residual, blind_window_stats, edge_features, make_inputs, motion_features,
                    soft_cap,
                    pooled_features)
from .predict import _is_oom, tile_origins


def _box_sums_at(a: torch.Tensor, half: int, r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``model._box_sums`` at rows r and columns c only: (N, len(r), len(c)),
    the same arithmetic for each element."""
    N, H, W = a.shape
    ii = F.pad(a.cumsum(1).cumsum(2), (1, 0, 1, 0))  # integral image
    r0, r1 = (r - half).clamp(0, H), (r + half + 1).clamp(0, H)
    c0, c1 = (c - half).clamp(0, W), (c + half + 1).clamp(0, W)
    return ii[:, r1][:, :, c1] - ii[:, r0][:, :, c1] - ii[:, r1][:, :, c0] + ii[:, r0][:, :, c0]


def pooled_features_at(x: torch.Tensor, t0: int, t1: int, radius: int, scales: list[int], r: torch.Tensor,
                       c: torch.Tensor) -> torch.Tensor:
    """``model.pooled_features`` evaluated at rows r and columns c of the window
    only (B, 2 * len(scales), t1 - t0, len(r), len(c)): the box sums are
    taken from the same integral images, so the values equal those of the
    full map at these pixels."""
    B, _, T, H, W = x.shape
    a, b = max(0, t0 - 3), min(T, t1 + 4)
    m, v, mi = x[:, 0, a:b].float(), x[:, 1, a:b].float(), x[:, 2, a:b].float()
    num = (v * (m - mi)).reshape(-1, H, W)
    den = (v * mi.clamp_min(0)).reshape(-1, H, W)
    bn, bd = _box_sums_at(num, radius, r, c), _box_sums_at(den, radius, r, c)
    nr, nc = len(r), len(c)
    out = []
    for s in scales:
        half = s if s > 0 else max(H, W)
        sn = (_box_sums_at(num, half, r, c) - bn).view(B, b - a, nr, nc)
        sd = (_box_sums_at(den, half, r, c) - bd).view(B, b - a, nr, nc)
        r1 = torch.where(sd > 1e-6, sn / sd.clamp_min(1e-6), torch.zeros_like(sn))
        cn = F.pad(sn.cumsum(1), (0, 0, 0, 0, 1, 0))
        cd = F.pad(sd.cumsum(1), (0, 0, 0, 0, 1, 0))
        t = torch.arange(t0, t1, device=x.device)
        lo, hi = (t - 3).clamp(a, b) - a, (t + 5).clamp(a, b) - a
        n8, d8 = cn[:, hi] - cn[:, lo], cd[:, hi] - cd[:, lo]
        r8 = torch.where(d8 > 1e-6, n8 / d8.clamp_min(1e-6), torch.zeros_like(n8))
        out += [r1[:, t0 - a: t1 - a].clamp(-1, 1), r8.clamp(-1, 1)]
    return torch.stack(out, 1)


def check_supported(cfg, wcfg) -> None:
    """Raise ValueError for checkpoints whose tiles the engine cannot reproduce:
    - the variance-factor input relative to the window's median (v6 and older);
    - spatial downsampling by more than 2: rectangles keep the windows' parity,
      not their phase modulo a larger factor;
    - half-plane branches reaching further in time than the window (their
      inputs would then be clipped at the window's edge, as the streams do not)."""
    if cfg.varfac_norm != "trial":
        raise ValueError("the stream engine needs a checkpoint with varfac_norm 'trial' (after v6)")
    if cfg.spatial_factor > 2:
        raise ValueError(f"the stream engine supports spatial downsampling by 1 or 2, not {cfg.spatial_factor}")
    if cfg.box and cfg.hp_temporal_reach > wcfg.t_reach:
        raise ValueError(f"the stream engine needs the half-planes' temporal reach ({cfg.hp_temporal_reach}) within "
                         f"the window's ({wcfg.t_reach})")


@dataclass
class Rect:
    """A rectangle of the trial grid (rows r0:r1, cols c0:c1, may extend past
    the grid) and the tiles (output origins) whose windows it contains."""
    r0: int
    r1: int
    c0: int
    c1: int
    tiles: list

    @property
    def area(self) -> int:
        """Pixels in the rectangle."""
        return (self.r1 - self.r0) * (self.c1 - self.c0)


def _bbox(tiles, op: int, sR: int) -> tuple[int, int, int, int]:
    """The rectangle holding the windows of these tiles."""
    rs = [r for r, _ in tiles]
    cs = [c for _, c in tiles]
    return min(rs) - sR, max(rs) + op + sR, min(cs) - sR, max(cs) + op + sR


def plan_rects(want: np.ndarray, op: int, sR: int, max_px: int = 0) -> list[Rect]:
    """Rectangles covering the windows of ``tile_origins(want, op)``: the
    connected parts of the windows' union (8-connected), each its bounding
    box; a part whose box holds more than ``max_px`` pixels (0: no limit) is
    split into groups of whole row bands. Every tile is in exactly one."""
    bands = tile_origins(want, op)
    tiles = [(r0, c0) for r0, cs in bands for c0 in cs]
    if not tiles:
        return []
    # union-find over tiles whose windows touch or overlap
    parent = list(range(len(tiles)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    w = op + 2 * sR
    for i, (ra, ca) in enumerate(tiles):
        for j in range(i):
            rb, cb = tiles[j]
            if abs(ra - rb) <= w and abs(ca - cb) <= w:
                parent[find(i)] = find(j)
    groups: dict[int, list] = {}
    for i, t in enumerate(tiles):
        groups.setdefault(find(i), []).append(t)
    out = []
    for g in groups.values():
        g.sort()
        parts = [g]
        if max_px > 0:
            parts, cur = [], []
            for r0 in sorted({r for r, _ in g}):
                band = [t for t in g if t[0] == r0]
                trial = cur + band
                a, b, c, d = _bbox(trial, op, sR)
                if cur and (b - a) * (d - c) > max_px:
                    parts.append(cur)
                    cur = band
                else:
                    cur = trial
            parts.append(cur)
        for p in parts:
            out.append(Rect(*_bbox(p, op, sR), tiles=p))
    out.sort(key=lambda r: (r.r0, r.c0))
    return out


class _CausalState:
    """A causal branch run forwards over chunks: each temporal convolution
    keeps the last 2 x dilation frames of its input (``CausalBlock``)."""

    def __init__(self, branch):
        self.b = branch
        self.full = [None] * len(branch.full)
        self.low = [None] * len(branch.low)

    @staticmethod
    def _block(blk, x, cache):
        h = F.gelu(blk.spatial(x))
        d2 = blk.t_pad  # 2 x dilation
        if cache is None:
            cache = h.new_zeros(h.shape[:2] + (d2,) + h.shape[3:])
        hh = torch.cat([cache, h], 2)
        del h
        new = hh[:, :, hh.shape[2] - d2:].clone()
        t = F.conv3d(hh, blk.temporal.weight, blk.temporal.bias, dilation=blk.temporal.dilation)
        del hh
        out = _residual(x, blk.alpha, F.gelu(t))
        for m in blk.mlp or ():
            out = m(out)
        return out, new

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """The branch before its shift on the next chunk (B, C, L, H, W)."""
        b = self.b
        h = b.inp(x)
        for i, blk in enumerate(b.full):
            h, self.full[i] = self._block(blk, h, self.full[i])
        if b.down is not None:
            L, H, W = h.shape[2:]
            f = b.cfg.spatial_factor
            if H % f or W % f:
                raise ValueError("rectangle sizes must be multiples of the spatial factor")
            low = b.down(h)
            for i, blk in enumerate(b.low):
                low, self.low[i] = self._block(blk, low, self.low[i])
            up = F.interpolate(low, size=(L, H, W), mode="nearest")
            h = h + up
        return h


# ------------------------------------------------- channels-last fast path
# The same layers on frames stored (T, H, W, C), channels last: the per-frame
# convolutions run as 2-D convolutions on (T, C, H, W) views with channels-last
# strides, the temporal ones as 2-D convolutions over (1, C, T, H * W) views,
# the 1 x 1 x 1 layers as matrix products over the channels, and the
# half-resolution paths are added at the output pixels only (no upsampled
# copy). cuDNN then needs no layout conversions.


def _conv_sp(x, conv, padding=0, stride=1):
    """A (1, k, k) or (1, kh, kw) Conv3d applied per frame to x (T, H, W, C)."""
    w = conv.weight[:, :, 0]
    y = F.conv2d(x.permute(0, 3, 1, 2), w, conv.bias, stride=stride, padding=padding,
                 dilation=conv.dilation[1:])
    return y.permute(0, 2, 3, 1)


def _conv_t(x, conv, padding=0):
    """A (3, 1, 1) Conv3d in time on x (T, H, W, C): 2-D over (1, C, T, H * W)."""
    T, H, W, C = x.shape
    v = x.reshape(1, T, H * W, C).permute(0, 3, 1, 2)
    y = F.conv2d(v, conv.weight[:, :, :, :, 0], conv.bias, padding=(padding, 0), dilation=(conv.dilation[0], 1))
    return y.permute(0, 2, 3, 1).reshape(y.shape[2], H, W, -1)


def _lin(x, conv):
    """A 1 x 1 x 1 Conv3d on the last (channel) axis."""
    return F.linear(x, conv.weight.flatten(1), conv.bias)


def _pointwise_cl(blocks, x):
    """PointwiseBlocks on (..., C)."""
    for m in blocks or ():
        x = _residual(x, m.alpha, _lin(F.gelu(_lin(x, m.w1)), m.w2))
    return x


class _CausalStateCL(_CausalState):
    """``_CausalState`` on channels-last frames (T, H, W, C); ``step`` returns
    the full-resolution part and the half-resolution part, which the caller
    adds at its pixels (the reference upsamples by nearest neighbour)."""

    @staticmethod
    def _block(blk, x, cache):
        if blk.spatial.kernel_size[1] > 1:
            pad = (blk.spatial.padding[1], blk.spatial.padding[2])
            h = F.gelu(_conv_sp(x, blk.spatial, padding=pad))
        else:
            h = F.gelu(_lin(x, blk.spatial))
        d2 = blk.t_pad
        if cache is None:
            cache = h.new_zeros((d2,) + h.shape[1:])
        hh = torch.cat([cache, h], 0)
        del h
        new = hh[hh.shape[0] - d2:].clone()
        t = _conv_t(hh, blk.temporal)
        del hh
        out = _residual(x, blk.alpha, F.gelu(t))
        return _pointwise_cl(blk.mlp, out), new

    def step(self, x):
        """x (L, H, W, n_inputs) -> (full (L, H, W, C), low (L, H/f, W/f, C) or None)."""
        b = self.b
        h = _lin(x, b.inp)
        for i, blk in enumerate(b.full):
            h, self.full[i] = self._block(blk, h, self.full[i])
        low = None
        if b.down is not None:
            f = b.cfg.spatial_factor
            if h.shape[1] % f or h.shape[2] % f:
                raise ValueError("rectangle sizes must be multiples of the spatial factor")
            low = _conv_sp(h, b.down, stride=f)
            for i, blk in enumerate(b.low):
                low, self.low[i] = self._block(blk, low, self.low[i])
        return h, low


class _SymBlockCL:
    """A HalfPlaneBlock streamed in time on channels-last frames: its temporal
    convolution is symmetric (frames f - d, f, f + d), so the output for a
    frame comes d frames after its input; ``step`` returns the outputs that
    became computable, oldest first, for consecutive frames from the stream's
    start. Before the start the convolution sees zeros (the window's padding);
    only outputs whose reach lies inside the stream are used."""

    def __init__(self, blk):
        self.blk, self.d = blk, blk.t_dil
        self.hbuf = self.xbuf = None  # h frames [p - d, q), x frames [p, q): p = next output frame

    def step(self, x):
        """x (L, H, W, C) -> outputs (n, H, W, C) for the next n frames (n may be 0)."""
        blk, d = self.blk, self.d
        if x.shape[0] == 0:
            return x
        H = x.shape[1]
        h = F.gelu(_conv_sp(x, blk.spatial, padding=(blk.s_dil, blk.s_dil))[:, :H])  # rows r - d and r
        if self.hbuf is None:
            self.hbuf, self.xbuf = h.new_zeros((d,) + h.shape[1:]), x[:0]
        hh = torch.cat([self.hbuf, h], 0)
        del h
        xx = torch.cat([self.xbuf, x], 0)
        n = hh.shape[0] - 2 * d
        if n <= 0:
            self.hbuf, self.xbuf = hh, xx
            return x[:0]
        t = _conv_t(hh, blk.temporal)  # valid: frames p .. p + n - 1
        self.hbuf, self.xbuf = hh[n:].clone(), xx[n:].clone()
        del hh
        out = _residual(xx[:n], blk.alpha, F.gelu(t))
        return _pointwise_cl(blk.mlp, out)


class _HalfPlaneStreamCL:
    """A HalfPlaneBranch streamed in time (``_SymBlockCL`` blocks), read at the
    output pixels: rows ``i`` (in the branch's rotated frame, before its row
    shift) and columns ``j``. ``step`` returns the branch's output at those
    pixels, (n, N, C), for the next n frames from the stream's start (delay:
    the sum of the temporal dilations)."""

    def __init__(self, br, i, j):
        self.br = br
        self.full = [_SymBlockCL(b) for b in br.full]
        self.low = [_SymBlockCL(b) for b in br.low]
        self.i, self.j = i, j
        self.fb = self.lb = None  # gathered full and low outputs not yet combined

    def step(self, x):
        """x (L, H, W, n_inputs) rotated frames -> (n, N, C)."""
        br = self.br
        h = _lin(x, br.inp)
        for b in self.full:
            h = b.step(h)
        g = h[:, self.i, self.j]
        if br.down is None:
            return g
        low = _conv_sp(h, br.down, stride=2) if h.shape[0] else h[:, ::2, ::2]
        del h
        for b in self.low:
            low = b.step(low)
        gl = low[:, (self.i - 1) // 2, self.j // 2]  # row r <- low row (r - 1) // 2
        self.fb = g if self.fb is None else torch.cat([self.fb, g], 0)
        self.lb = gl if self.lb is None else torch.cat([self.lb, gl], 0)
        n = min(self.fb.shape[0], self.lb.shape[0])
        out = self.fb[:n] + self.lb[:n]
        self.fb, self.lb = self.fb[n:], self.lb[n:]
        return out


def _half_plane_cl(br, x):
    """HalfPlaneBranch on channels-last frames x (T, H, W, n_inputs) (H, W even),
    before its final shift: (full (T, H, W, C), low (T, H/2, W/2, C) or None);
    the output at row i is full[i - s] + low[(i - s - 1) // 2] (s = box_radius
    + 1, rows i - s >= 1), as the reference's padding and shifts give."""
    h = _lin(x, br.inp)

    def block(blk, x):
        d = blk.s_dil
        H = x.shape[1]
        hs = _conv_sp(x, blk.spatial, padding=(d, d))[:, :H]  # rows r - d and r
        t = _conv_t(F.gelu(hs), blk.temporal, padding=blk.t_dil)
        out = _residual(x, blk.alpha, F.gelu(t))
        return _pointwise_cl(blk.mlp, out)

    for blk in br.full:
        h = block(blk, h)
    low = None
    if br.down is not None:
        low = _conv_sp(h, br.down, stride=2)
        for blk in br.low:
            low = block(blk, low)
    return h, low


class StreamPredictor:
    """``predict.predict_trial``'s baselines by streaming (module docstring)."""

    def __init__(self, model: SlabBlindSpotNet, wcfg, device: torch.device, amp_dtype=None,
                 chunk: int = 256, max_px: int = 0, pixel_block: int = 8192, keep_future_on_gpu: bool | None = None,
                 layout: str = "ncdhw", mem_fraction: float = 0.6, bytes_per_voxel: int = 0):
        if layout not in ("ncdhw", "cl"):
            raise ValueError(f"layout must be ncdhw or cl, not {layout!r}")
        self.cl = layout == "cl"
        # GPU memory planning: a rectangle of A pixels in chunks of L frames needs about
        # bytes_per_voxel * A * L; rectangles are split and chunks shortened to stay within
        # mem_fraction of the free memory (and halved on running out)
        self.mem_fraction = float(mem_fraction)
        self.bytes_per_voxel = int(bytes_per_voxel) or (1000 if self.cl else 1600)
        self.m, self.cfg = model, model.cfg
        self.w = wcfg
        self.device, self.amp = device, amp_dtype
        self.chunk = int(chunk)
        self.max_px = int(max_px)
        self.pixel_block = int(pixel_block)
        self.keep_future_on_gpu = keep_future_on_gpu
        self.timings: dict = {}

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.amp, enabled=self.amp is not None)

    # ------------------------------------------------------------- inputs
    def _maps(self, tr, st, pad: int):
        """Whole-trial maps padded by `pad` on every side, as predict_trial."""
        R, C = tr.n_rows, tr.n_cols
        mean_p = np.full((R + 2 * pad, C + 2 * pad), np.nan, np.float32)
        mean_p[pad:pad + R, pad:pad + C] = np.where(tr.valid, tr.mean_image, np.nan)
        vf_p = np.zeros_like(mean_p)
        vf_p[pad:pad + R, pad:pad + C] = np.nan_to_num(tr.varfac)
        var_p = np.full_like(mean_p, np.nan)
        var_p[pad:pad + R, pad:pad + C] = st["var_movie"]
        var_p = np.where(np.isfinite(mean_p), var_p, np.nan).astype(np.float32)
        return mean_p, vf_p, var_p

    def _movie_rect(self, movie, f0, f1, r0, r1, c0, c1):
        """movie[f0:f1, r0:r1, c0:c1] with NaN outside the trial (frames and
        grid); the movie is NaN at pixels that are not valid and on discarded
        frames already (``predict``), as predict_trial reads its bands."""
        T, R, C = movie.shape
        out = np.full((f1 - f0, r1 - r0, c1 - c0), np.nan, np.float32)
        a, b = max(f0, 0), min(f1, T)
        ra, rb, ca, cb = max(r0, 0), min(r1, R), max(c0, 0), min(c1, C)
        if a < b and ra < rb and ca < cb:  # the movie is masked (predict)
            out[a - f0:b - f0, ra - r0:rb - r0, ca - c0:cb - c0] = movie[a:b, ra:rb, ca:cb]
        return out

    def _x(self, movie_np, maps, rect_rc, pad, scale, gain):
        """make_inputs of a movie block (L, H, W) over rows/cols rect_rc."""
        r0, r1, c0, c1 = rect_rc
        mean_p, vf_p, var_p = maps
        dev = self.device
        sl = (slice(r0 + pad, r1 + pad), slice(c0 + pad, c1 + pad))
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)[None]  # noqa: E731
        return make_inputs(t(movie_np), t(mean_p[sl]), t(vf_p[sl]), scale, t(var_p[sl]), gain,
                           varfac_norm=self.cfg.varfac_norm)

    # ----------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, tr, st, rows: np.ndarray, cols: np.ndarray, movie: np.ndarray | None = None,
                masked: bool = False):
        """Baseline and noise variance at the valid pixels (rows, cols): (frame,
        N) float32 in counts and counts^2, as ``predict.predict_trial``.
        ``movie`` (frame, row, col) in counts with NaN where not valid or
        discarded; default read from ``tr``. ``masked``: the movie is NaN already
        at every pixel that is not valid and on every discarded frame (as
        ``registered.load_registered`` makes it), so it is used as it is;
        otherwise a masked copy is made."""
        cfg, w = self.cfg, self.w
        dev = self.device
        check_supported(cfg, w)
        self._valid = np.asarray(tr.valid, bool)
        self._discard = np.asarray(tr.discard, bool)
        if movie is None:
            movie, masked = tr._f["movie"][tr.channel], False  # an HDF5 dataset or an array
        if not masked:  # NaN where predict_trial reads NaN: pixels not valid, discarded frames
            movie = np.array(movie, np.float32)
            movie[:, ~self._valid] = np.nan
            movie[self._discard] = np.nan
        elif movie.dtype != np.float32:
            movie = movie.astype(np.float32)
        T, R, C = movie.shape
        sR, op = w.s_reach, w.out_px
        index = np.full((R, C), -1, np.int64)
        index[rows, cols] = np.arange(len(rows))
        cuda = dev.type == "cuda"
        L = max(1, self.chunk)
        max_px = self.max_px
        if max_px <= 0 and cuda:  # rectangles that fit a chunk of L frames in a share of the free memory
            max_px = int(self.mem_fraction * torch.cuda.mem_get_info()[0] / (self.bytes_per_voxel * L))
        rects = plan_rects(index >= 0, op, sR, max_px)
        pad = sR + op + 2  # maps padded so every rectangle is a plain slice
        maps = self._maps(tr, st, pad)
        s = float(tr.scale)
        scale = torch.full((1,), s, device=dev)
        gain = torch.full((1,), float(st["gain"]), device=dev)
        mu_out = np.full((T, len(rows)), np.nan, np.float32)
        var_out = np.full((T, len(rows)), np.nan, np.float32)
        tim = {"future": 0.0, "past": 0.0, "same_frame": 0.0, "fusion": 0.0, "inputs": 0.0, "n_rects": len(rects),
               "rect_px": [r.area for r in rects], "max_px": max_px, "chunks": [], "oom_retries": 0}
        for rect in rects:
            # output pixels of this rectangle: the valid pixels of its tiles' regions, in index order
            sel = []
            for r0, c0 in rect.tiles:
                sub = index[r0:r0 + op, c0:c0 + op]
                sel.append(sub[sub >= 0])
            px = np.sort(np.concatenate(sel)) if sel else np.zeros(0, np.int64)
            if not len(px):
                continue
            Lr = L
            if cuda:  # a rectangle larger than planned (one row band): shorter chunks
                torch.cuda.empty_cache()
                free = torch.cuda.mem_get_info()[0] - T * len(px) * cfg.width * 2  # the future branch's store
                Lr = max(16, min(L, int(self.mem_fraction * free / (self.bytes_per_voxel * rect.area))))
            self._shrink = 1.0
            while True:
                try:
                    (self._rect_cl if self.cl else self._rect)(movie, maps, pad, rect, index, rows, cols, px, Lr,
                                                               scale, gain, mu_out, var_out, tim)
                    tim["chunks"].append(Lr)
                    break
                except (torch.OutOfMemoryError, RuntimeError) as e:
                    if not _is_oom(e) or Lr <= 16:
                        raise
                # retry outside the except block, so the failed pass's tensors are freed
                tim["oom_retries"] += 1
                Lr = max(16, Lr // 2)
                self._shrink /= 2  # the passes that do not follow the chunk shrink too
                torch.cuda.empty_cache()
        self.timings = tim
        return mu_out, var_out

    def _store(self, shape, dtype, budget: float):
        """An empty store on the GPU if it takes less than `budget` of the free
        memory, else in pinned host memory."""
        dev = self.device
        n = int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()
        on_gpu = dev.type != "cuda" or n < budget * torch.cuda.mem_get_info()[0]
        return torch.empty(shape, dtype=dtype, device=dev if on_gpu else "cpu",
                           pin_memory=(not on_gpu and dev.type == "cuda"))

    def _rect_cl(self, movie, maps, pad, rect, index, rows, cols, px, L, scale, gain, mu_out, var_out, tim):
        """One rectangle on channels-last frames, in passes that each choose
        their own chunks: (0) window statistics (and edge features) at the
        output pixels over the whole trial; (1) pooled (and motion) features
        per tile on the tile's own window, in chunks of up to 512 frames;
        (2) the future branch backwards; (3) the past branch and the four
        half-plane branches forwards over the same input chunks, both carried
        in state (no halo), and the fusion as their outputs become available."""
        import time
        cfg, w = self.cfg, self.w
        dev = self.device
        T = movie.shape[0]
        tR, sR, op = w.t_reach, w.s_reach, w.out_px
        shift = cfg.gap + 1
        s = float(scale[0])
        half = self.amp or torch.float32  # the dtype the branches compute in (autocast's)
        pr, pc = (rows[px] - rect.r0).astype(np.int64), (cols[px] - rect.c0).astype(np.int64)
        prt, pct = torch.from_numpy(pr).to(dev), torch.from_numpy(pc).to(dev)
        N = len(px)
        rc = (rect.r0, rect.r1, rect.c0, rect.c1)
        rr, cc = rows[px], cols[px]
        mean_p, vf_p, var_p = maps
        tt = lambda q: torch.from_numpy(np.ascontiguousarray(q)).to(dev)  # noqa: E731
        mi, vf, vm = (tt(q[rr + pad, cc + pad])[None, :, None] for q in (mean_p, vf_p, var_p))
        B = max(1, self.pixel_block)
        span = cfg.gap + cfg.window_reach
        Bs = B
        if dev.type == "cuda":  # ~16 float32 values per pixel and frame in the statistics pass
            Bs = int(0.2 * torch.cuda.mem_get_info()[0] / ((T + 2 * span) * 4 * 16))
            Bs = max(64, int(min(B, Bs) * self._shrink))  # halved on each retry
        # the rectangle's movie on the GPU once when it fits; inputs are then sliced there
        rmov = None
        if dev.type == "cuda" and (T + 2 * tR) * rect.area * 4 < 0.15 * torch.cuda.mem_get_info()[0]:
            rmov = tt(self._movie_rect(movie, -tR, T + tR, *rc))
        rsl = (slice(rect.r0 + pad, rect.r1 + pad), slice(rect.c0 + pad, rect.c1 + pad))
        rmaps = [tt(q[rsl]) for q in maps]

        def xin(f0, f1, win=None):
            """make_inputs of frames f0..f1-1 over the rectangle, or over the window
            win = (r0, r1, c0, c1) of the trial grid inside it."""
            mp = rmaps
            if rmov is not None:
                mv_ = rmov[f0 + tR:f1 + tR]
                if win is not None:
                    a0, a1, b0, b1 = win[0] - rect.r0, win[1] - rect.r0, win[2] - rect.c0, win[3] - rect.c0
                    mv_, mp = mv_[:, a0:a1, b0:b1], [q[a0:a1, b0:b1] for q in rmaps]
                return make_inputs(mv_[None], mp[0][None], mp[1][None], scale, mp[2][None], gain,
                                   varfac_norm=cfg.varfac_norm)
            w_ = win or rc
            return self._x(self._movie_rect(movie, f0, f1, *w_), maps, w_, pad, scale, gain)

        # 0. per-pixel temporal statistics over the whole trial (window stats, edge features)
        t0 = time.time()
        wst = [self._store((T, N), torch.float32, 0.1) for _ in range(3)]
        est = gst = None
        if cfg.edge_features:
            est, gst = self._store((6, T, N), torch.float32, 0.1), self._store((2, T, N), torch.float32, 0.1)
        for k0 in range(0, N, Bs):
            k1 = min(k0 + Bs, N)
            if rmov is not None:  # frames -span .. T + span - 1 at the output pixels, on the GPU
                mvt = rmov[tR - span:tR + T + span][:, prt[k0:k1], pct[k0:k1]]
            else:
                mv = np.full((T + 2 * span, k1 - k0), np.nan, np.float32)
                mv[span:span + T] = movie[:, rr[k0:k1], cc[k0:k1]]  # (masked: NaN on discarded frames)
                mvt = tt(mv)
            x = make_inputs(mvt[None, :, :, None], mi[:, k0:k1], vf[:, k0:k1], scale, vm[:, k0:k1], gain,
                            varfac_norm=cfg.varfac_norm)
            for st_, a in zip(wst, blind_window_stats(x[:, 0], x[:, 1], cfg.gap, cfg.window_reach)):
                st_[:, k0:k1] = a[0, span:span + T, :, 0].to(st_.device)
            if cfg.edge_features:
                e, g = edge_features(x[:, 0], x[:, 1], x[:, 2, 0], cfg.gap, cfg.window_reach, cfg.edge_near,
                                     (span, span + T))
                est[:, :, k0:k1] = e[0, :, :, :, 0].to(est.device)
                gst[:, :, k0:k1] = g[0, :, :, :, 0].to(gst.device)
            del x
        # the per-pixel inputs the fusion reads (maps only; constant in time)
        xpx = make_inputs(torch.full((1, 1, N, 1), float("nan"), device=dev), mi, vf, scale, vm, gain,
                          varfac_norm=cfg.varfac_norm)  # (1, 5, 1, N, 1)
        tim["pixel_stats"] = tim.get("pixel_stats", 0.0) + time.time() - t0
        # 1. pooled (and motion) features per tile, on the tile's own window
        t0 = time.time()
        pf = None
        if cfg.box:
            pf = self._store((cfg.n_pool + cfg.n_motion, T, N), half, 0.15)
            pos = np.full(index.max() + 1, -1, np.int64)
            pos[px] = np.arange(N)
            ns = op + 2 * sR
            Lp = min(512, max(L, 16))  # the planned chunk (halved on retries), at most 512
            for r0, c0 in rect.tiles:
                sub = index[r0:r0 + op, c0:c0 + op]
                ok = sub >= 0
                if not ok.any():
                    continue
                ii, jj = (torch.from_numpy(v).to(dev) for v in np.nonzero(ok))
                k = torch.from_numpy(pos[sub[ok]]).to(pf.device)
                wrc = (r0 - sR, r0 - sR + ns, c0 - sR, c0 - sR + ns)
                for a in range(0, T, Lp):
                    b = min(a + Lp, T)
                    x = xin(a - 3, b + 4, wrc)
                    region = torch.arange(sR, sR + op, device=dev)
                    feats = [pooled_features_at(x, 3, 3 + b - a, cfg.box_radius, cfg.pool_scales, region, region)]
                    if cfg.motion_features:
                        mf = motion_features(x, 3, 3 + b - a, cfg.box_radius, cfg.pool_scales)
                        feats.append(mf[:, :, :, sR:sR + op, sR:sR + op])
                    fe = torch.cat(feats, 1)[0][:, :, ii, jj]  # (npool, frames, K)
                    pf[:, a:b, k] = fe.to(half).to(pf.device)
                    del x, feats, fe
        tim["same_frame"] += time.time() - t0
        # 2. future branch, backwards over trial frames [shift, T + tR): kept at the output pixels
        t0 = time.time()
        hf_store = self._store((cfg.width, T, N), half, 0.25)
        fut = _CausalStateCL(self.m.future)
        hi = T + tR
        while hi > shift:
            lo = max(hi - L, shift)
            ti = time.time()
            x = xin(lo, hi)
            tim["inputs"] += time.time() - ti
            with self._autocast():
                g = self._branch(fut, x, pr, pc, flip=True)
            del x
            a, b = max(lo - shift, 0), min(hi - shift, T)
            if a < b:
                hf_store[:, a:b] = g[:, a + shift - lo:b + shift - lo].to(half).to(hf_store.device)
            del g
            hi = lo
        del fut
        tim["future"] += time.time() - t0
        # 3. past branch and half-planes forwards on the same input chunks; fusion as outputs arrive
        R = cfg.hp_temporal_reach if cfg.box else 0
        past = _CausalStateCL(self.m.past)
        hps = []
        if cfg.box:
            H, W = rect.r1 - rect.r0, rect.c1 - rect.c0
            if H % 2 or W % 2:
                raise ValueError("rectangle sizes must be even")
            idx = torch.arange(H * W).view(H, W)
            for k in range(4):
                br = self.m.hp_views[k] if self.m.hp_views is not None else self.m.hp
                ridx = torch.rot90(idx, k, (0, 1))
                inv = torch.empty(H * W, dtype=torch.long)
                inv[ridx.flatten()] = torch.arange(H * W)
                p = inv[torch.from_numpy(pr * W + pc)]
                hps.append((k, _HalfPlaneStreamCL(br, (p // ridx.shape[1]).to(dev) - (cfg.box_radius + 1),
                                                  (p % ridx.shape[1]).to(dev))))
        pend_p, pend_h = None, [None] * len(hps)  # outputs from frame out_t on
        got_p = 0  # past outputs for frames < got_p are in pend_p (or fused)
        got_h = [-R] * len(hps)  # half-plane streams start at frame -R
        f_in = -tR
        out_t = 0
        f_end = T + R  # last input frame + 1 the half-planes need; the past branch needs T - shift
        while out_t < T:
            t0 = time.time()
            if f_in < f_end:
                fe = min(f_in + L, f_end)
                ti = time.time()
                x = xin(f_in, fe)
                xc = x[0].permute(1, 2, 3, 0).contiguous()  # (frames, H, W, n_inputs)
                del x
                tim["inputs"] += time.time() - ti
                with self._autocast():
                    if f_in < T - shift:  # past branch: pre-shift frame f -> output f + shift
                        n_p = min(fe, T - shift) - f_in
                        full, low = past.step(xc[:n_p])
                        g = full[:, prt, pct]
                        if low is not None:
                            f = past.b.cfg.spatial_factor
                            g = g + low[:, prt // f, pct // f]
                        del full, low
                        first = f_in + shift
                        g = g.permute(2, 0, 1)[:, max(out_t - first, 0):]
                        if first + n_p > out_t:
                            pend_p = g if pend_p is None else torch.cat([pend_p, g], 1)
                        got_p = first + n_p
                    tt0 = time.time()
                    a = max(f_in, -R)  # half-planes see frames -R .. T + R - 1
                    if hps and fe > a:
                        for n_k, (k, hs) in enumerate(hps):
                            g = hs.step(torch.rot90(xc[a - f_in:], k, (1, 2)).contiguous())  # (n, N, C)
                            first = got_h[n_k]
                            got_h[n_k] += g.shape[0]
                            g = g.permute(2, 0, 1)[:, max(out_t - first, 0):]
                            if got_h[n_k] > out_t and g.shape[1]:
                                pend_h[n_k] = g if pend_h[n_k] is None else torch.cat([pend_h[n_k], g], 1)
                    tim["same_frame"] += time.time() - tt0
                del xc
                f_in = fe
            tim["past"] += time.time() - t0
            ready = min([got_p] + got_h + [T])
            piece = max(L, 128)  # frames fused at a time
            if ready - out_t < min(piece, T - out_t) and f_in < f_end:
                continue
            t1 = min(ready, T, out_t + piece)
            n = t1 - out_t
            if n <= 0:
                raise RuntimeError("stream engine: no output frames became ready")
            tf = time.time()
            hp_c, pend_p = pend_p[:, :n], pend_p[:, n:]
            extra = []
            if hps:
                hh = torch.cat([ph[:, :n] for ph in pend_h], 0)  # direction order 0..3, as the reference
                pend_h = [ph[:, n:] for ph in pend_h]
                extra = [hh, pf[:, out_t:t1].to(dev)]
            mus, vars_ = [], []
            for k0 in range(0, N, B):
                k1 = min(k0 + B, N)
                ws = [st_[out_t:t1, k0:k1].to(dev) for st_ in wst]
                with self._autocast():
                    mu, var = self._head_cl(
                        hp_c[:, :, k0:k1], hf_store[:, out_t:t1, k0:k1].to(dev), [e[:, :, k0:k1] for e in extra],
                        None if est is None else est[:, out_t:t1, k0:k1].to(dev),
                        None if gst is None else gst[:, out_t:t1, k0:k1].to(dev),
                        ws, xpx[:, :, :, k0:k1].expand(-1, -1, n, -1, -1), scale.float(), gain.float())
                mus.append(mu.float().cpu().numpy())
                vars_.append(var.float().cpu().numpy())
            mu_out[out_t:t1, px] = np.concatenate(mus, 1) * s
            var_out[out_t:t1, px] = np.concatenate(vars_, 1) * s * s
            tim["fusion"] += time.time() - tf
            out_t = t1

    def _head_cl(self, hp, hf, extra, edge, gates, ws, x, s, g):
        """``SlabBlindSpotNet.head`` on flat pixels, channels last: the same
        features (the model's ``_features``) and layers (its 1 x 1 x 1 weights
        as matrix products). hp, hf (C, n, P); extra, edge (C', n, P); gates
        (2, n, P); ws = window mean, variance, count (n, P); x (1, 5, n, P, 1)
        inputs at the voxels; s, g (1,). Returns mu and var (n, P), scale units."""
        m, cfg = self.m, self.cfg
        gs = (g / s).view(-1, 1, 1, 1)
        wm, wv, nn_ = (a[None, :, :, None] for a in ws)
        feats = m._features(x.float(), wm, wv, nn_, gs).to(hp.dtype)[0, :, :, :, 0]  # (6, n, P)
        parts = [hp, hf, feats] + [e.to(hp.dtype) for e in extra + ([] if edge is None else [edge])]
        z = torch.cat([q.permute(1, 2, 0) for q in parts], -1)  # (n, P, inputs), ModelConfig.fusion_inputs order
        h = F.gelu(_lin(F.gelu(_lin(z, m.fuse[0])), m.fuse[2]))
        del z
        h = _pointwise_cl(m.fuse_more, h)
        mi = x[0, 2, :, :, 0].float()
        wm = torch.where(ws[2] > 0, ws[0], mi)
        mix = torch.tanh(_lin(h, m.head_mix)[..., 0].float())
        base = (mi + mix * (wm - mi)).clamp_min(1e-4)
        c_mu, c_var = _lin(h, m.head_mu)[..., 0].float(), _lin(h, m.head_var)[..., 0].float()
        if m.head_edge is not None:
            e, gt = _lin(h, m.head_edge).float(), gates.float()
            c_mu = c_mu + gt[0] * e[..., 0] + gt[1] * e[..., 1]
            c_var = c_var + gt[0] * e[..., 2] + gt[1] * e[..., 3]
        mu = base * torch.exp(soft_cap(c_mu, cfg.max_log_correction))
        var = (gs[0, 0] * mu + m.log_floor.exp() / s.view(-1, 1) ** 2) * torch.exp(soft_cap(c_var, cfg.max_log_var))
        return mu, var

    def _rect(self, movie, maps, pad, rect, index, rows, cols, px, L, scale, gain, mu_out, var_out, tim):
        """One rectangle in chunks of L frames: the future branch backwards
        (kept at the output pixels), then the past branch forwards with the
        same-frame parts and the fusion; fills mu_out and var_out at px."""
        import time
        cfg, w = self.cfg, self.w
        dev = self.device
        T = movie.shape[0]
        tR = w.t_reach
        shift = cfg.gap + 1
        s = float(scale[0])
        pr, pc = rows[px] - rect.r0, cols[px] - rect.c0  # positions in the rectangle
        pr, pc = pr.astype(np.int64), pc.astype(np.int64)
        prt, pct = torch.from_numpy(pr).to(dev), torch.from_numpy(pc).to(dev)
        N = len(px)
        rc = (rect.r0, rect.r1, rect.c0, rect.c1)
        # 1. future branch, backwards over trial frames [shift, T + tR): keep it at the output pixels
        t0 = time.time()
        on_gpu = self.keep_future_on_gpu
        if on_gpu is None:
            need = T * N * cfg.width * torch.empty((), dtype=self.amp or torch.float32).element_size()
            on_gpu = dev.type != "cuda" or need < 0.25 * torch.cuda.mem_get_info()[0]
        hf_store = (torch.empty((cfg.width, T, N), dtype=self.amp or torch.float32,
                                device=dev if on_gpu else "cpu", pin_memory=(not on_gpu and dev.type == "cuda")))
        fut = (_CausalStateCL if self.cl else _CausalState)(self.m.future)
        hi = T + tR  # exclusive end of the reversed sequence's first chunk
        while hi > shift:  # pre-shift frames shift..T-1+shift give outputs 0..T-1
            lo = max(hi - L, shift)
            ti = time.time()
            xm = self._movie_rect(movie, lo, hi, *rc)
            x = self._x(xm, maps, rc, pad, scale, gain)
            tim["inputs"] += time.time() - ti
            with self._autocast():
                g = self._branch(fut, x, pr, pc, flip=True)  # (C, frames lo..hi-1, N)
            del x
            # pre-shift frame f -> output frame f - shift
            a, b = max(lo - shift, 0), min(hi - shift, T)
            if a < b:
                g = g[:, a + shift - lo:b + shift - lo]
                hf_store[:, a:b] = g.to(hf_store.dtype) if on_gpu else g.to(hf_store.dtype).cpu()
            del g
            hi = lo
        del fut
        tim["future"] += time.time() - t0
        # 2. past branch forwards from frame -tR; same-frame parts and fusion per chunk of output frames
        past = (_CausalStateCL if self.cl else _CausalState)(self.m.past)
        f_in = -tR  # next pre-shift frame to process (zero state before it, as the window's padding)
        pending = None  # past branch at output frames out_t.., (C, frames, N)
        out_t = 0
        while out_t < T:
            t1 = min(out_t + L, T)  # output frames out_t..t1-1 this round
            t0 = time.time()
            while f_in < t1 - shift:  # pre-shift frame f -> output frame f + shift
                fe = min(f_in + L, t1 - shift)
                ti = time.time()
                xm = self._movie_rect(movie, f_in, fe, *rc)
                x = self._x(xm, maps, rc, pad, scale, gain)
                tim["inputs"] += time.time() - ti
                with self._autocast():
                    g = self._branch(past, x, pr, pc)
                del x
                first = f_in + shift
                f_in = fe
                if fe + shift <= out_t:  # pre-trial outputs only
                    continue
                g = g[:, max(out_t - first, 0):]
                pending = g if pending is None else torch.cat([pending, g], 1)
            tim["past"] += time.time() - t0
            n = t1 - out_t
            hp_chunk, pending = pending[:, :n], pending[:, n:]
            mu, var = self._fuse(movie, maps, rc, pad, rect, index, px, pr, pc, prt, pct, out_t, t1, hp_chunk,
                                 hf_store[:, out_t:t1], scale, gain, tim)
            mu_out[out_t:t1, px] = mu * s
            var_out[out_t:t1, px] = var * s * s
            out_t = t1

    def _branch(self, state, x, pr, pc, flip=False):
        """A causal branch's next chunk (before its shift) at the output pixels
        (rows pr, cols pc of the rectangle): (C, frames, N). ``flip``: the
        future branch, which runs on the chunk in reversed time."""
        if not self.cl:
            dev = x.device
            prt, pct = torch.from_numpy(pr).to(dev), torch.from_numpy(pc).to(dev)
            h = state.step(x.flip(2) if flip else x)
            if flip:
                h = h.flip(2)
            return h[0][:, :, prt, pct]
        xc = x[0].permute(1, 2, 3, 0)  # (L, H, W, n_inputs)
        xc = (xc.flip(0) if flip else xc).contiguous()
        full, low = state.step(xc)
        dev = x.device
        prt, pct = torch.from_numpy(pr).to(dev), torch.from_numpy(pc).to(dev)
        g = full[:, prt, pct]
        if low is not None:
            f = state.b.cfg.spatial_factor
            g = g + low[:, prt // f, pct // f]
        if flip:
            g = g.flip(0)
        return g.permute(2, 0, 1)

    def _fuse(self, movie, maps, rc, pad, rect, index, px, pr, pc, prt, pct, t0, t1, hp, hf, scale, gain, tim):
        """mu and var (frames, N) in scale units for output frames t0..t1 of one
        rectangle, given the past and future branches there (C, frames, N)."""
        import time
        cfg, w = self.cfg, self.w
        dev = self.device
        n = t1 - t0
        N = len(px)
        tc = time.time()
        extra = []
        if cfg.box:
            R = max(cfg.hp_temporal_reach, 4)  # the half-planes' reach; the pooled features need t-3..t+4
            xm = self._movie_rect(movie, t0 - R, t1 + R, *rc)
            x = self._x(xm, maps, rc, pad, scale, gain)  # (1, 5, n + 2R, H, W)
            with self._autocast():
                if self.cl:
                    hh = self._half_planes_cl(x, pr, pc)[:, :, R:R + n]
                else:
                    hh = self._half_planes(x, prt, pct)[:, :, R:R + n]  # (1, 4 hp_width, n, N, 1)
            # pooled (and motion) features per tile, on the tile's own window
            op, sR = w.out_px, w.s_reach
            npool = cfg.n_pool + cfg.n_motion
            pf = torch.zeros((1, npool, n, N, 1), dtype=torch.float32, device=dev)
            pos = {int(p): k for k, p in enumerate(px)}
            for r0, c0 in rect.tiles:
                sub = index[r0:r0 + op, c0:c0 + op]
                ok = sub >= 0
                if not ok.any():
                    continue
                wr, wc = r0 - sR - rect.r0, c0 - sR - rect.c0
                ns = op + 2 * sR
                xt = x[:, :, R - 3:R + n + 4, wr:wr + ns, wc:wc + ns]
                feats = [pooled_features(xt, 3, 3 + n, cfg.box_radius, cfg.pool_scales)]
                if cfg.motion_features:
                    feats.append(motion_features(xt, 3, 3 + n, cfg.box_radius, cfg.pool_scales))
                fe = torch.cat(feats, 1)[:, :, :, sR:sR + op, sR:sR + op][:, :, :, :sub.shape[0], :sub.shape[1]]
                ii, jj = np.nonzero(ok)
                k = torch.from_numpy(np.array([pos[int(p)] for p in sub[ok]], np.int64)).to(dev)
                pf[:, :, :, k, 0] = fe[:, :, :, torch.from_numpy(ii).to(dev), torch.from_numpy(jj).to(dev)]
            del x
            extra = [hh, pf]
        tim["same_frame"] += time.time() - tc
        tf = time.time()
        # per-pixel inputs at the output pixels: frames t0 - (gap + reach) .. t1 + gap + reach
        span = cfg.gap + cfg.window_reach
        rr = np.asarray(prt.cpu().numpy() + rect.r0)
        cc = np.asarray(pct.cpu().numpy() + rect.c0)
        T = movie.shape[0]
        a, b = t0 - span, t1 + span
        mv = np.full((b - a, N), np.nan, np.float32)
        aa, bb = max(a, 0), min(b, T)
        mv[aa - a:bb - a] = movie[aa:bb, rr, cc]  # (masked: NaN on discarded frames)
        mean_p, vf_p, var_p = maps
        t = lambda q: torch.from_numpy(np.ascontiguousarray(q)).to(dev)  # noqa: E731
        mi, vf, vm = (t(q[rr + pad, cc + pad])[None, :, None] for q in (mean_p, vf_p, var_p))
        mus, vars_ = [], []
        B = max(1, self.pixel_block)
        for k0 in range(0, N, B):
            k1 = min(k0 + B, N)
            x = make_inputs(t(mv[:, k0:k1])[None, :, :, None], mi[:, k0:k1], vf[:, k0:k1], scale, vm[:, k0:k1], gain,
                            varfac_norm=cfg.varfac_norm)  # (1, 5, b - a, n_px, 1)
            wm, wv, nn_ = blind_window_stats(x[:, 0], x[:, 1], cfg.gap, cfg.window_reach)
            core = slice(span, span + n)
            edge = gates = None
            if cfg.edge_features:
                edge, gates = edge_features(x[:, 0], x[:, 1], x[:, 2, 0], cfg.gap, cfg.window_reach, cfg.edge_near,
                                            (span, span + n))
            p = {"hp": hp[None, :, :, k0:k1, None], "hf": hf[None, :, :, k0:k1, None].to(hp.device),
                 "wm": wm[:, core], "wv": wv[:, core], "n": nn_[:, core],
                 "extra": [e[:, :, :, k0:k1] for e in extra], "edge": edge, "gates": gates,
                 "x": x[:, :, core], "s": scale.float(), "g": gain.float()}
            if extra:
                p["extra"] = [e.to(extra[0].dtype) for e in p["extra"]]
            with self._autocast():
                mu, var = self.m.head(p)
            mus.append(mu[0, :, :, 0].float().cpu().numpy())
            vars_.append(var[0, :, :, 0].float().cpu().numpy())
        tim["fusion"] += time.time() - tf
        return np.concatenate(mus, 1), np.concatenate(vars_, 1)

    def _half_planes(self, x, prt, pct):
        """SlabBlindSpotNet._half_planes at the output pixels only: (1, 4 *
        hp_width, frames, N, 1), one direction at a time."""
        m = self.m
        H, W = x.shape[3:]
        xp = F.pad(x, (0, W % 2, 0, H % 2))
        outs = []
        for k in range(4):
            br = m.hp_views[k] if m.hp_views is not None else m.hp
            o = torch.rot90(br(torch.rot90(xp, k, (3, 4))), -k, (3, 4))[..., :H, :W]
            outs.append(o[:, :, :, prt, pct][..., None])
            del o
        return torch.cat(outs, 1)

    def _half_planes_cl(self, x, pr, pc):
        """``_half_planes`` on channels-last frames, at the output pixels only:
        each direction runs on the rotated rectangle and is read at the output
        pixels' rotated positions (no rotation back): (1, 4 * hp_width,
        frames, N, 1)."""
        m = self.m
        H, W = x.shape[3:]
        if H % 2 or W % 2:
            raise ValueError("rectangle sizes must be even")
        s = self.cfg.box_radius + 1
        dev = x.device
        idx = torch.arange(H * W).view(H, W)
        outs = []
        for k in range(4):
            br = m.hp_views[k] if m.hp_views is not None else m.hp
            xr = torch.rot90(x[0], k, (2, 3)).permute(1, 2, 3, 0).contiguous()  # (T, H', W', n_inputs)
            ridx = torch.rot90(idx, k, (0, 1))
            Wr = ridx.shape[1]
            inv = torch.empty(H * W, dtype=torch.long)
            inv[ridx.flatten()] = torch.arange(H * W)
            pos = inv[torch.from_numpy(pr * W + pc)]
            i = (pos // Wr).to(dev) - s
            j = (pos % Wr).to(dev)
            full, low = _half_plane_cl(br, xr)
            g = full[:, i, j]
            if low is not None:
                g = g + low[:, (i - 1) // 2, j // 2]
            outs.append(g)
            del full, low
        return torch.cat(outs, 2).permute(2, 0, 1)[None, :, :, :, None]
