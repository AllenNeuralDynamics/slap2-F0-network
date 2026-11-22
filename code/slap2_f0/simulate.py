"""Simulated trials on the 80 Hz export grid, with a known, noise-free baseline.

Real exported trials serve as templates, so the simulation keeps their
geometry, brightness, events and noise statistics:

* **baseline** = the template's whole-trial mean minus its mean activity (the
  best constant baseline, see ``ceiling``), times a slow multiplicative drift:
  bleaching, a global fluctuation and a few smooth spatial fields that each
  fluctuate in time (Ornstein-Uhlenbeck processes, 0.5-8 s);
* **events** = the template's own fitted activity (GIAnT's footprints and
  kinetics on the trial grid), shifted circularly in time by a random offset
  and scaled by a random amplitude, so they no longer line up with anything
  in the baseline;
* **noise** = photon noise with the template's measured variance/mean ratio,
  then filtered by a small separable kernel (7 taps in time, 3 in each spatial
  axis) fitted to the template's measured correlations at 1 and 2 frames and
  1 pixel and to the inflation of long averages. Photon events are drawn at
  the rate that gives the right variance after the kernel, with their size
  raised (and rate lowered) to match the template's measured skew;
* **excess** = how much more the template's residual (movie - activity)
  varies than that photon noise, per pixel, fitted to its variance of single
  frames and of 64-frame means (``stats``: var_res, var64_res). A fast part
  scales the photon noise (it has the same time structure); a slow part is
  added to the baseline as spatially smooth fluctuations over 1-20 s. Bright
  pixels of real trials vary more than photon noise explains, at both time
  scales; a network trained without this read the extra spread as activity
  and over-subtracted on real data (v3 run, worst in the brightest sessions);
* **fast shared brightness** (optional, ``fast_global_sd``, ``fast_field_sd``):
  a global and a few regional (smooth field) multiplicative fluctuations with
  time constants of 1.5-20 frames, part of the drift factor. Real trials
  carry such changes, shared by every pixel in the frame (field-wide
  fractional brightness SD 2-5% per frame, much of it faster than the 8
  hidden frames, also in the GFP control): a slab blind spot cannot see them,
  a box blind spot can (``model``);
* **silent regions** (optional, ``silent_trial`` and ``silent_frac``): no
  activity in random smooth patches of the field, or anywhere in the virtual
  trial. Bright labeled pixels then vary (the excess) without any events, as
  in a GFP control, so the template's footprint geometry no longer says which
  pixels are active and only the events themselves do. Without them, a
  network fine-tuned on sensor sessions (where every bright pixel in the loss
  has activity) set the baseline of GFP dendrites too low (v4 run);
* **realistic low counts** (optional, ``photon_zeros``): every voxel is drawn
  from a zero-inflated distribution with its mean and variance, exact zeros
  with probability exp(-mean / c), else gamma (see ``_photon_counts``). Real
  movies are photon counts: never negative, and dim pixels are exactly 0 in
  many frames (37% at 0.5-2 counts, 27% at 2-4, 9% at 4-8 in 780125), most of
  all at ROI edges and in small ROIs; the filtered Poisson noise above is
  negative in a few percent of voxels and never exactly 0. On such pixels
  SiLo's moving median sits far below the mean and makes false sources. The
  draws come from one correlated Gaussian field (a Gaussian copula) whose
  kernels are stronger than the template's, so that the counts after the
  zero-inflated mapping, which weakens correlations, keep the template's
  correlations in time and space (``Template.pz_kernels``);
* **shuffled activity timing** (optional, ``shuffle_prob``): the template's
  activity timeline is cut into blocks at quiet frames and reordered, which
  breaks the recordings' stimulus rhythm while keeping every event;
* **residual motion** (optional, ``motion_sd_px``): the sample moves rigidly
  by a sub-pixel displacement d(t) (rows, columns; one Ornstein-Uhlenbeck
  process per axis). The baseline and the activity move with it (cubic
  spline shift of each frame); the scan does not, so the valid pixels and
  the noise stay where they are. Before the shift, pixels outside the valid
  ones take the nearest valid pixel's value: the sample there is only
  unscanned, not dark. ``truth`` is the moved baseline, ``truth_static`` the
  baseline without motion, and the whole-trial maps are those of the moving
  trial (the mean image is blurred by the motion). With ``motion_substeps``
  the motion is drawn at several steps per frame and each frame pools them
  through the frame-formation kernel, as the 80 Hz frames pool 200 Hz
  samples: the frame sees the kernel-weighted mean displacement and is
  blurred by the displacements' spread within the kernel (to second order);
* **focus fluctuations** (optional, ``focus_sd_px2``): the frames' blur
  changes over time, b_r d2/drow2 + b_c d2/dcol2 of the scene, with an
  isotropic part (focus) and a part per axis, each fast and slow;
* **local motion** (optional, ``local_sd_px``): on top of the rigid
  displacement, a smooth displacement field that changes over time (a few
  smooth spatial fields per axis, each weighted by a fast and a slow OU
  process), applied per frame by cubic-spline warping like the rigid shift.
  Registration removes rigid motion; what is left is mostly local.

Each draw is a new virtual trial; the window generator returns everything the
training windows of real data have (movie, activity, mean image, variance map,
masks) plus ``truth``, the noise-free baseline. Whole-trial inputs (mean image
and variance map) are computed from the virtual trial's known statistics plus
the sampling noise a real whole-trial average would have.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import h5py
import numpy as np

from .io import TrialFile


@dataclass
class SimConfig:
    amp: tuple[float, float] = (0.5, 1.5)  # activity amplitude, uniform
    # drift sizes: real trials show little drift (a blind temporal average of
    # the residual barely beats a constant, see ceiling), so these are modest
    bleach: float = 0.05  # largest fractional loss over the trial
    bleach_tau_s: tuple[float, float] = (10.0, 60.0)
    global_sd: float = 0.02  # largest SD of the global slow fluctuation
    global_tau_s: tuple[float, float] = (1.0, 8.0)
    n_fields: int = 2  # smooth spatial fields with their own fluctuation
    field_sd: float = 0.02
    field_px: tuple[float, float] = (15.0, 40.0)  # spatial scale of the fields
    field_tau_s: tuple[float, float] = (0.5, 5.0)
    rate_hz: float = 80.0
    excess: float = 1.0  # scale of the template's excess variance (0 = photon noise and the drift above only)
    n_slow: int = 6  # slow baseline processes that carry the excess's slow part
    slow_tau_s: tuple[float, float] = (1.0, 20.0)  # their time constants, log-uniform
    slow_max_sd: float = 1.0  # cap on a pixel's slow-fluctuation SD, relative to its mean (excess_maps)
    slow_px: tuple[float, float] = (2.0, 6.0)  # spatial scale of their fields
    # activity-free regions: with probability silent_trial a virtual trial has no
    # activity at all; otherwise a fraction (uniform in silent_frac) of its valid
    # pixels, in smooth patches of silent_px, has none (the defaults keep all)
    silent_trial: float = 0.0
    silent_frac: tuple[float, float] = (0.0, 0.0)
    silent_px: tuple[float, float] = (20.0, 60.0)
    # fast shared brightness changes (fractions; 0 = none, as in v1-v5): SDs drawn
    # uniform in (0, sd) per virtual trial, time constants log-uniform in frames
    fast_global_sd: float = 0.0
    n_fast_fields: int = 3
    fast_field_sd: float = 0.0
    fast_field_px: tuple[float, float] = (15.0, 60.0)
    fast_tau_frames: tuple[float, float] = (1.5, 20.0)
    # dark sources: a fraction (uniform in dark_frac) of the footprint-support
    # pixels, in patches of dark_px, get their baseline scaled by a factor
    # uniform in dark_factor while their activity stays, as at real sources on
    # nearly dark pixels (the movie mostly 0, with large events; 0 = none)
    dark_frac: tuple[float, float] = (0.0, 0.0)
    dark_px: tuple[float, float] = (2.0, 6.0)
    dark_factor: tuple[float, float] = (0.02, 0.3)
    # shuffled activity timing: the template's activity is cut into blocks of
    # shuffle_frames (uniform, cut where the summed activity is low, so events
    # stay whole) and the blocks are put in random order, which breaks the
    # stimulus rhythm of the recordings (2 s at 2 Hz, then 1 s off) while
    # keeping every event; probability shuffle_prob per virtual trial (0 = never)
    shuffle_prob: float = 0.0
    shuffle_frames: tuple[float, float] = (20.0, 60.0)
    # realistic low counts (photon_zeros): each voxel is drawn, through a
    # Gaussian copula, from a zero-inflated distribution with its mean and
    # variance whose share of exact zeros is exp(-mean / c), c the template
    # pixel's mean count per photon event, measured from its share of zero
    # frames (stats zero_movie), and whose positive part is gamma (real dim
    # pixels: positive-part skew 3.1-3.5, gamma 2.9-3.2, lognormal 7-9). The
    # copula's field has kernels chosen so that the counts keep the template's
    # correlations (Template.pz_kernels). Real movies are never negative and dim
    # pixels are exactly 0 in many frames, most of all at ROI edges and in small
    # ROIs, where SiLo's moving median makes false sources
    photon_zeros: bool = False
    # photon_zeros as up to 2026-11-21 (v7-v9, S0-S3): only voxels with at least
    # 2% zeros were redrawn, lognormal above zero, through a field with the
    # template's own kernels and independent of the other voxels' noise. Its
    # counts had about half the template's correlations (878030: next column
    # 0.23 against 0.39, next row 0.12 against 0.23, 1 frame 0.43 against 0.60)
    # and too much skew (6.2 against 4.8). Kept to reproduce those runs
    pz_legacy: bool = False
    # photon-count augmentation: per virtual trial, the template's gain (counts
    # per photon) times a factor log-uniform in photon_gain (1, 1 = off). A factor
    # k > 1 keeps the baseline and the activity in counts and divides the photons
    # by k: photon noise variance k x gain x mean, photon events k x larger in
    # counts at the same skew per photon (so the counts' skew grows by sqrt(k)),
    # k x the counts per photon event of realistic low counts (more exact zeros).
    # Normalized by the trial scale, this is the noise of a dimmer session. The
    # excess's slow part (a baseline fluctuation in counts^2) stays; its fast part
    # stays a factor on the photon noise. The window reports the gain it used, for
    # the network's input and the loss weights. With photon_zeros the copula's
    # kernels stay the template's (Template.pz_kernels at its own gain), so the
    # counts' correlations come out a little weaker where k adds zeros.
    # Why: below about 1 photon per frame at footprint support (776270, 876767,
    # 878030; training mice 0.4-1.0) the S0-S3 networks set the baseline 5-10% low
    # in activity-free simulations, in every pixel class, also on flat fields
    # without any structure; at 4x the photons the bias falls to 0.4-2%
    # (docs/plan.md, "Low photon counts: the baseline sits low").
    photon_gain: tuple[float, float] = (1.0, 1.0)
    # residual motion (0 = none): per axis (rows, columns) an OU process with SD
    # uniform in motion_sd_px (pixels) and time constant log-uniform in
    # motion_tau_frames; the baseline and activity move, the scan does not
    motion_sd_px: tuple[float, float] = (0.0, 0.0)
    motion_tau_frames: tuple[float, float] = (1.5, 20.0)
    # 0: each frame sees one displacement and no blur. n > 0: the motion is drawn at
    # n steps per frame and pooled through the frame-formation kernel (the
    # template's temporal noise kernel, interpolated): a frame sees the kernel-
    # weighted mean displacement, blurred by the displacements' spread in the kernel
    motion_substeps: int = 0
    # focus fluctuations (0 = none): the frame's blur changes by b(t) per axis,
    # frame = scene + b_r d2/drow2 + b_c d2/dcol2 (px^2; > 0 blurrier than the
    # mean, < 0 sharper). b = an isotropic part with SD uniform in focus_sd_px2
    # plus a part per axis with focus_aniso times that SD; each the sum of a fast
    # and a slow OU process of equal variance (time constants log-uniform in
    # focus_tau_frames). Measured on the viewer trials: 0.1-0.3 px^2 in all,
    # slow part mostly isotropic, fast part not
    focus_sd_px2: tuple[float, float] = (0.0, 0.0)
    focus_aniso: float = 0.7
    focus_tau_frames: tuple[tuple[float, float], tuple[float, float]] = ((1.5, 8.0), (40.0, 320.0))
    # local (non-rigid) motion (0 = none): per axis, n_local_fields smooth spatial
    # fields (spacing uniform in local_px) weighted by OU processes, a fast and a
    # slow one each (local_tau_frames, log-uniform), whose SDs are drawn uniform in
    # local_sd_px; the fields are scaled so that the local displacement's SD per
    # pixel is that SD on average. Measured on the viewer trials beyond the
    # simulator's level (local_motion.py, 64 px blocks): slow 0.07-0.22 px in all,
    # fast 0.12-0.2 px in 803121 and 878030
    local_sd_px: tuple[float, float] = (0.0, 0.0)
    local_px: tuple[float, float] = (20.0, 60.0)
    n_local_fields: int = 3
    local_tau_frames: tuple[tuple[float, float], tuple[float, float]] = ((1.5, 8.0), (40.0, 320.0))


def _fill_nearest(valid: np.ndarray):
    """Index arrays that give every pixel the nearest valid pixel (None if
    all or none are valid)."""
    if valid.all() or not valid.any():
        return None
    from scipy import ndimage
    return ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)


def _shift(img: np.ndarray, d, fill) -> np.ndarray:
    """img (R, C) moved by d = (rows, cols) pixels: out[x] = img[x - d], cubic
    spline, edges continued; fill (from _fill_nearest) first replaces the
    pixels outside the valid ones."""
    from scipy import ndimage
    if fill is not None:
        img = img[fill[0], fill[1]]
    return ndimage.shift(img, (float(d[0]), float(d[1])), order=3, mode="nearest").astype(np.float32)


def _blur(img: np.ndarray, c) -> np.ndarray:
    """img (R, C) blurred by a small displacement covariance c = (rows, cols,
    rows-cols) in px^2: img + c_rr/2 d2/drow2 + c_cc/2 d2/dcol2 + c_rc d2/drow dcol
    (second order in the displacement; edges continued)."""
    p = np.pad(img, 1, mode="edge")
    drr = p[2:, 1:-1] - 2 * img + p[:-2, 1:-1]
    dcc = p[1:-1, 2:] - 2 * img + p[1:-1, :-2]
    drc = (p[2:, 2:] - p[2:, :-2] - p[:-2, 2:] + p[:-2, :-2]) / 4
    return (img + 0.5 * (c[0] * drr + c[1] * dcc) + c[2] * drc).astype(np.float32)


def _warp(img: np.ndarray, d, loc: np.ndarray, w: np.ndarray, fill) -> np.ndarray:
    """img (R, C) moved by the rigid displacement d (rows, cols) plus the local
    displacement field sum_j loc[:, j] * w[:, j] (loc (2, J, R, C), w (2, J)):
    out[x] = img[x - D(x)], cubic spline, edges continued."""
    from scipy import ndimage
    if fill is not None:
        img = img[fill[0], fill[1]]
    D = np.einsum("aj,ajrc->arc", w.astype(np.float32), loc)
    if d is not None:
        D = D + np.asarray(d, np.float32)[:, None, None]
    rr, cc = np.indices(img.shape, dtype=np.float32)
    return ndimage.map_coordinates(img, [rr - D[0], cc - D[1]], order=3, mode="nearest").astype(np.float32)


def _move(img: np.ndarray, r: dict, t: int, fill, loc=None) -> np.ndarray:
    """img as frame t sees it: moved to its displacement (rigid, plus the local
    field when loc, the local fields on img's region, is given), blurred by the
    displacement's spread within the frame (motion with substeps) and by the
    focus change."""
    if loc is not None:
        out = _warp(img, r["motion"][:, t] if "motion" in r else None, loc, r["local_proc"][:, :, t], fill)
    elif "motion" in r:
        out = _shift(img, r["motion"][:, t], fill)
    else:
        out = (img[fill[0], fill[1]] if fill is not None else img).astype(np.float32)
    c = r["motion_cov"][:, t].astype(np.float64) if "motion_cov" in r else None
    if "focus" in r:
        f = np.array([2 * r["focus"][0, t], 2 * r["focus"][1, t], 0.0])
        c = f if c is None else c + f
    if c is not None:
        out = _blur(out, c)
    return out


def _pooled_motion(d: np.ndarray, kt: np.ndarray, n: int, T: int):
    """Frame displacements from motion drawn at n steps per frame, d (2, T n + 2 h):
    each frame pools the steps within the temporal kernel kt (taps at whole
    frames, interpolated to the steps; h = n * (len(kt) // 2)). Returns the
    weighted mean (2, T) and the weighted covariance (3, T): rows, cols, rows-cols."""
    k = len(kt) // 2
    h = n * k
    w = np.interp(np.arange(-h, h + 1) / n, np.arange(-k, k + 1), kt)
    w = w / w.sum()
    win = np.lib.stride_tricks.sliding_window_view(d, 2 * h + 1, axis=1)[:, ::n][:, :T]  # (2, T, 2h+1)
    mean = win @ w
    dev = win - mean[..., None]
    cov = np.stack([(dev[0] ** 2) @ w, (dev[1] ** 2) @ w, (dev[0] * dev[1]) @ w])
    return mean.astype(np.float32), cov.astype(np.float32)


def _spatial_tap(rho: float) -> float:
    """a such that the kernel [a, 1, a] gives neighbour correlation rho."""
    rho = float(np.clip(rho, 0.0, 0.7))
    return 0.0 if rho == 0 else (1 - np.sqrt(1 - 2 * rho ** 2)) / (2 * rho)


def noise_kernels(st: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized (sum 1) temporal, row and column kernels matching the measured
    correlations. Temporal: 7 symmetric taps [k3, k2, k1, 1, k1, k2, k3] fitted to
    the correlations at 1 and 2 frames and to the inflation of long averages
    (F64 ~ (sum k)^2 / sum k^2), which sets the variance of SiLo's 8-frame
    averages. Spatial: [a, 1, a] per axis from the neighbour correlations."""
    t1, t2 = float(np.clip(st["rho_t1"], 0, 0.9)), float(np.clip(st["rho_t2"], -0.2, 0.8))
    F = float(st.get("F64", np.nan))
    g = np.linspace(-0.5, 1.5, 81)
    k1, k2, k3 = np.meshgrid(np.linspace(0, 1.5, 61), g, g, indexing="ij", sparse=True)
    r0 = 1 + 2 * (k1 ** 2 + k2 ** 2 + k3 ** 2)
    r1 = 2 * (k1 + k1 * k2 + k2 * k3)
    r2 = 2 * k2 + k1 ** 2 + 2 * k1 * k3
    err = (r1 / r0 - t1) ** 2 + (r2 / r0 - t2) ** 2
    if np.isfinite(F) and F > 1:
        err = err + 0.05 * (((1 + 2 * (k1 + k2 + k3)) ** 2 / r0 - F) / F) ** 2
    err = err + 1e-6 * (k1 ** 2 + k2 ** 2 + k3 ** 2)  # prefer short kernels when the lags leave k3 free (white noise)
    i = np.unravel_index(np.argmin(err), err.shape)
    a1, a2, a3 = k1.flat[i[0]], k2.flat[i[1]], k3.flat[i[2]]
    kt = np.array([a3, a2, a1, 1.0, a1, a2, a3])
    a_r, a_c = _spatial_tap(st["rho_r"]), _spatial_tap(st["rho_c"])
    kr, kc = np.array([a_r, 1.0, a_r]), np.array([a_c, 1.0, a_c])
    return kt / kt.sum(), kr / kr.sum(), kc / kc.sum()


def _filter(x: np.ndarray, kt: np.ndarray, kr: np.ndarray, kc: np.ndarray) -> np.ndarray:
    """Separable 'valid' correlation of a (T, R, C) array (output shrinks by the kernel sizes - 1)."""
    for ax, k in ((0, kt), (1, kr), (2, kc)):
        n = x.shape[ax] - len(k) + 1
        out = np.zeros(x.shape[:ax] + (n,) + x.shape[ax + 1:], np.float32)
        for i, w in enumerate(k):
            if w:
                out += np.float32(w) * np.take(x, np.arange(i, i + n), axis=ax)
        x = out
    return x


def _ou(n: int, tau_frames: np.ndarray, sd: np.ndarray, rng) -> np.ndarray:
    """Stationary AR(1) processes, one per row of the output (len(tau) x n)."""
    phi = np.exp(-1.0 / np.maximum(tau_frames, 1e-3))
    e = rng.standard_normal((len(phi), n)) * (sd * np.sqrt(1 - phi ** 2))[:, None]
    x = np.empty_like(e)
    x[:, 0] = rng.standard_normal(len(phi)) * sd
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x


def kernel_var(kt: np.ndarray, n: int) -> float:
    """Variance of n-frame means of white noise filtered by kt, relative to the
    single-frame variance."""
    ac = np.correlate(kt, kt, "full") / (kt ** 2).sum()
    lags = np.arange(-(len(kt) - 1), len(kt))
    return float((np.maximum(n - np.abs(lags), 0) * ac).sum() / n ** 2)


def ou_var(tau: float, n: int) -> float:
    """Variance of n-frame means of a unit stationary OU process (tau in frames)."""
    x = n / tau
    return float(2 / x * (1 - (1 - np.exp(-x)) / x))


def _smooth_valid(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mean over the valid pixels of each 3x3 neighbourhood."""
    v = valid.astype(np.float64)
    x = np.pad(np.where(valid, a, 0.0), 1)
    w = np.pad(v, 1)
    R, C = a.shape
    num = sum(x[i:i + R, j:j + C] for i in range(3) for j in range(3))
    den = sum(w[i:i + R, j:j + C] for i in range(3) for j in range(3))
    return np.where(valid, num / np.maximum(den, 1), 0.0)


def excess_maps(st: dict, kt: np.ndarray, valid: np.ndarray, cfg: "SimConfig | None" = None):
    """Per-pixel excess of the residual over the simulated photon noise
    (variance p = gain * mean): a factor on the photon noise variance (fast)
    and the variance of a slow baseline fluctuation (slow), from
        var_res   = (1 + fast) * p        + slow
        var64_res = (1 + fast) * p * k64  + slow * c64
    with k64 the photon kernel's and c64 the slow processes' variance of
    64-frame means. Each pixel's 64-frame variance rests on ~25 blocks, so both
    maps are averaged over 3x3 valid neighbours. Zeros if the stats lack the
    maps (older caches)."""
    z = np.zeros(valid.shape, np.float32)
    if "var_res" not in st or "var64_res" not in st:
        return z, z
    cfg = cfg or SimConfig()
    taus = np.exp(np.linspace(*np.log(cfg.slow_tau_s), 9)) * cfg.rate_hz
    c64 = float(np.mean([ou_var(t, 64) for t in taus]))
    k64 = kernel_var(kt, 64)
    ph = st["gain"] * np.maximum(np.nan_to_num(st["mean_movie"]), 0)
    v1, v64 = np.nan_to_num(st["var_res"]), np.nan_to_num(st["var64_res"])
    ok = valid & np.isfinite(st["var_res"]) & np.isfinite(st["var64_res"]) & (ph > 0)
    slow = (v64 - k64 * v1) / (c64 - k64)
    fast = np.where(ph > 0, (v1 - slow) / np.maximum(ph, 1e-9) - 1, 0)
    slow = np.maximum(_smooth_valid(slow, ok), 0)
    # at most slow_max_sd x the pixel's mean: where the residual moves far more than
    # the movie (GIAnT fit artifacts, see stats.unfit_pixels) the estimate is not a baseline
    slow = np.minimum(slow, (cfg.slow_max_sd * np.maximum(np.nan_to_num(st["mean_movie"]), 0)) ** 2)
    fast = np.clip(_smooth_valid(fast, ok), -0.3, 3.0)
    return np.where(ok, fast, 0).astype(np.float32), np.where(ok, slow, 0).astype(np.float32)


def _field(R: int, C: int, spacing: float, rng) -> np.ndarray:
    """Smooth random field with unit SD: bilinear upsampling of a coarse Gaussian grid."""
    nr, nc = int(np.ceil(R / spacing)) + 2, int(np.ceil(C / spacing)) + 2
    g = rng.standard_normal((nr, nc))
    rr, cc = np.arange(R) / spacing + 0.5, np.arange(C) / spacing + 0.5
    g = np.stack([np.interp(rr, np.arange(nr), g[:, j]) for j in range(nc)], 1)
    g = np.stack([np.interp(cc, np.arange(nc), g[i]) for i in range(R)], 0)
    return ((g - g.mean()) / (g.std() + 1e-9)).astype(np.float32)


class Template:
    """A real exported trial prepared for simulation."""

    def __init__(self, tf: TrialFile, st: dict):
        self.tf, self.st = tf, st
        self.T, self.R, self.C = tf.n_frames, tf.n_rows, tf.n_cols
        self.kt, self.kr, self.kc = noise_kernels(st)
        self.ksum2 = float((self.kt ** 2).sum() * (self.kr ** 2).sum() * (self.kc ** 2).sum())
        self.Ft = float(1.0 / (self.kt ** 2).sum())  # variance inflation of long averages
        # skew: fewer, larger photon events than Poisson at the measured gain
        # (as with variable pulse heights) raise the skew at the same variance;
        # q scales the event size and divides the event rate by q^2
        k3 = float((self.kt ** 3).sum() * (self.kr ** 3).sum() * (self.kc ** 3).sum()) / self.ksum2 ** 1.5
        bg = tf.valid & ~tf.labeled & np.isfinite(st["mean_movie"]) & (st["mean_movie"] > 0)
        skew1 = float(np.median(k3 / np.sqrt(st["mean_movie"][bg] * self.ksum2 / st["gain"]))) if bg.any() else np.nan
        q = float(st.get("skew", np.nan)) / skew1 if np.isfinite(skew1) and skew1 > 0 else 1.0
        self.q = float(np.clip(q, 1.0, 3.0)) if np.isfinite(q) else 1.0
        mean_movie = np.where(tf.valid, st["mean_movie"], np.nan)
        floor = 0.01 * tf.scale
        self.B = np.where(tf.valid, np.maximum(mean_movie - st["mean_act"], floor), np.nan).astype(np.float32)
        self.mean_act = np.nan_to_num(st["mean_act"]).astype(np.float32)
        self.var_act = np.maximum(np.nan_to_num(st["mean_act2"]) - self.mean_act ** 2, 0).astype(np.float32)
        self.n_valid = max(1.0, float(np.nanmedian(np.where(tf.valid, st["n_valid"], np.nan))))
        self.gain = float(st["gain"])
        self.fast, self.slow = excess_maps(st, self.kt, tf.valid)
        self.zc = zero_count_scale(st, tf.valid)  # None without zero_movie (older stats)
        at = st.get("act_trace")
        self.act_trace = np.asarray(at, np.float64) if at is not None and np.size(at) == self.T else None
        unfit = st.get("unfit")  # GIAnT fit artifacts (stats.sanitize): their per-frame activity is not used
        self.unfit = np.asarray(unfit, bool) if unfit is not None and np.any(unfit) else None
        self._pz = {}

    def pz_kernels(self, excess: float = 1.0):
        """Temporal, row and column kernels (sum 1) of the Gaussian field behind
        photon_zeros, and their sum of squares. The zero-inflated mapping
        weakens correlations: output correlation A(rho) = sum_k c_k^2 rho^k / V for
        a latent correlation rho, c_k the mapping's normalized Hermite
        coefficients and V its variance, pooled over a sample of the template's
        background pixels as stats.trial_stats pools them. The latent
        correlations at 1 and 2 frames and 1 row and column are A^-1 of the
        template's, and the temporal kernel's target for the inflation of
        64-frame means is adjusted until the counts' inflation matches."""
        if excess in self._pz:
            return self._pz[excess]
        st, tf = self.st, self.tf
        bg = tf.valid & ~tf.labeled & np.isfinite(self.B) & (self.B > 0)
        idx = np.flatnonzero(bg)
        idx = np.random.default_rng(0).choice(idx, min(300, len(idx)), replace=False)
        mu = self.B.ravel()[idx].astype(np.float64)
        var = self.gain * mu * np.maximum(1 + self.fast.ravel()[idx] * excess, 0.05)
        A = _copula_attenuation(mu, var, self.zc.ravel()[idx].astype(np.float64))

        def inv(r):  # latent correlation giving output correlation r (bisection; A is increasing on [0, 1))
            r = float(r)
            if not np.isfinite(r) or r <= 0:
                return r
            lo, hi = 0.0, 0.999
            if A(hi) <= r:
                return hi
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if A(mid) < r else (lo, mid)
            return 0.5 * (lo + hi)

        lat = dict(st, rho_t1=inv(st["rho_t1"]), rho_t2=inv(st["rho_t2"]), rho_r=inv(st["rho_r"]), rho_c=inv(st["rho_c"]))
        F = float(st.get("F64", np.nan))
        kt, kr, kc = noise_kernels(lat)
        if np.isfinite(F) and F > 1:
            Fl = F
            for _ in range(6):
                ac = np.correlate(kt, kt, "full")[len(kt):] / (kt ** 2).sum()  # latent correlation at lags 1..
                tau = np.arange(1, len(ac) + 1)
                Fo = 1 + 2 * float(np.sum((1 - tau / 64) * np.array([A(a) for a in ac])))
                if abs(Fo - F) < 0.01 * F:
                    break
                Fl = max(Fl + (F - Fo), 1.01)
                kt = noise_kernels(dict(lat, F64=Fl))[0]
        ksum2 = float((kt ** 2).sum() * (kr ** 2).sum() * (kc ** 2).sum())
        self._pz[excess] = (kt, kr, kc, ksum2)
        return self._pz[excess]


def zero_count_scale(st: dict, valid: np.ndarray):
    """Per pixel c, the mean count per photon event, such that the share of
    zero frames is exp(-mean / c) (a compound Poisson process: zero when no
    photon event lands); from the measured share of zero frames and the mean
    where both are informative (0.01-0.99), averaged over 3x3 valid neighbours,
    and the template's median elsewhere. None without zero_movie."""
    if "zero_movie" not in st:
        return None
    z = np.asarray(st["zero_movie"], np.float64)
    m = np.asarray(st["mean_movie"], np.float64)
    ok = valid & np.isfinite(z) & np.isfinite(m) & (z > 0.01) & (z < 0.99) & (m > 0)
    if not ok.any():
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(ok, -m / np.log(np.where(ok, z, 0.5)), 0.0)
    cs = _smooth_valid(c, ok)
    has = _smooth_valid(ok.astype(np.float64), valid) > 0
    out = np.where(has, cs, float(np.median(c[ok])))
    return np.clip(out, 0.2, 50.0).astype(np.float32)


def _photon_zeros(movie: np.ndarray, mu: np.ndarray, var: np.ndarray, zc: np.ndarray, g: np.ndarray,
                  z_min: float = 0.02) -> np.ndarray:
    """Voxels whose expected share of zeros exp(-mu / c) is at least z_min
    get a zero-inflated value with mean mu and variance var: 0 with that
    probability, otherwise lognormal. The draw goes through g, a correlated
    unit Gaussian field (the template's noise kernels), as a Gaussian copula,
    so the noise keeps its time and space structure. Other voxels keep movie."""
    import torch

    z = np.exp(-np.maximum(mu, 0) / zc)
    z = np.minimum(z, var / np.maximum(var + mu ** 2, 1e-12) - 1e-3)  # a positive part must exist
    use = (z >= z_min) & np.isfinite(movie) & (mu > 0)
    if not use.any():
        return movie
    zu, mu_u, v_u = z[use], mu[use], var[use]
    mp = mu_u / (1 - zu)  # mean and variance of the positive part
    vp = np.maximum((v_u + mu_u ** 2) / (1 - zu) - mp ** 2, 1e-6)
    s2 = np.log1p(vp / mp ** 2)
    u = torch.special.ndtr(torch.from_numpy(g[use].astype(np.float64))).numpy()
    w = np.clip((u - zu) / (1 - zu), 1e-9, 1 - 1e-9)
    pos = np.exp(np.log(mp) - s2 / 2 + np.sqrt(s2) * torch.special.ndtri(torch.from_numpy(w)).numpy())
    out = movie.copy()
    out[use] = np.where(u < zu, 0.0, pos).astype(np.float32)
    return out


_GAMMA_TABLE = None


def _gamma_table():
    """log Q(Phi(x); k), Q the standard gamma's quantile (shape k, scale 1), on a
    grid of log k (0.02-2000) and x (-7..7)."""
    global _GAMMA_TABLE
    if _GAMMA_TABLE is None:
        from scipy import special
        lk = np.linspace(np.log(0.02), np.log(2000.0), 241)
        x = np.linspace(-7.0, 7.0, 561)
        K, X = np.meshgrid(np.exp(lk), x, indexing="ij")
        q = special.gammaincinv(K, special.ndtr(X))
        small = (special.log_ndtr(X) + special.gammaln(K + 1)) / K  # Q ~ (u Gamma(k + 1))^(1/k) as u -> 0
        with np.errstate(divide="ignore"):
            lq = np.where(q > 1e-30, np.log(np.maximum(q, 1e-300)), small)
        _GAMMA_TABLE = (lk, x, lq.ravel())
    return _GAMMA_TABLE


def gamma_quantile(k: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The standard gamma's quantile (shape k, scale 1) at probability Phi(x):
    bilinear interpolation of log Q in (log k, x) (relative error < 0.5% for
    |x| <= 5), Wilson-Hilferty above k = 2000. Fast enough for whole windows,
    unlike scipy's gammaincinv."""
    lk, xg, lq = _gamma_table()
    k, x = np.asarray(k, np.float64), np.asarray(x, np.float64)
    a = (np.log(np.clip(k, np.exp(lk[0]), np.exp(lk[-1]))) - lk[0]) / (lk[1] - lk[0])
    b = (np.clip(x, xg[0], xg[-1]) - xg[0]) / (xg[1] - xg[0])
    i = np.minimum(a.astype(np.int64), len(lk) - 2)
    j = np.minimum(b.astype(np.int64), len(xg) - 2)
    fa, fb = a - i, b - j
    n = len(xg)
    i0 = i * n + j
    v = (1 - fa) * ((1 - fb) * lq[i0] + fb * lq[i0 + 1]) + fa * ((1 - fb) * lq[i0 + n] + fb * lq[i0 + n + 1])
    q = np.exp(v)
    big = k > np.exp(lk[-1])
    if big.any():
        kb, xb = k[big], x[big]
        q[big] = kb * np.maximum(1 - 1 / (9 * kb) + xb / (3 * np.sqrt(kb)), 0) ** 3
    return q


def _photon_counts(mu: np.ndarray, var: np.ndarray, zc: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Voxel values with mean mu (> 0) and variance var: 0 with probability
    z = exp(-mu / zc) (capped so that a positive part exists), otherwise gamma
    with the positive part's mean and variance; drawn through g, a unit Gaussian
    field, as a Gaussian copula (value = F^-1(Phi(g)))."""
    from scipy import special
    z = np.exp(-np.maximum(mu, 0) / zc)
    z = np.clip(np.minimum(z, var / np.maximum(var + mu ** 2, 1e-12) - 1e-3), 0.0, None)
    mp = mu / (1 - z)  # mean and variance of the positive part
    vp = np.maximum((var + mu ** 2) / (1 - z) - mp ** 2, 1e-9 * np.maximum(mp, 1e-6) ** 2)
    u = special.ndtr(g)
    w = np.clip((u - z) / (1 - z), 1e-12, 1 - 1e-12)
    return np.where(u < z, 0.0, vp / mp * gamma_quantile(mp ** 2 / vp, special.ndtri(w)))


def _copula_attenuation(mu: np.ndarray, var: np.ndarray, zc: np.ndarray, K: int = 40):
    """A(rho): the correlation of _photon_counts at two voxels of the same pixel
    whose latent Gaussians correlate rho, pooled over pixels (sum of covariances
    over sum of variances, as stats.trial_stats pools neighbour correlations):
    sum_p sum_k c_pk^2 rho^k / sum_p V_p, with c_pk the normalized Hermite
    coefficients of pixel p's mapping and V_p its variance (truncated at K terms,
    which slightly underestimates A as rho -> 1)."""
    x = np.linspace(-8.0, 8.0, 4001)
    wts = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi) * (x[1] - x[0])
    n = len(mu)
    h = _photon_counts(np.repeat(mu, len(x)), np.repeat(var, len(x)), np.repeat(zc, len(x)),
                       np.tile(x, n)).reshape(n, len(x))
    m = h @ wts
    V = float(np.sum((h ** 2) @ wts - m ** 2))
    c2 = np.empty(K)
    prev, cur = np.ones_like(x), x.copy()  # normalized probabilists' Hermite polynomials He_k / sqrt(k!)
    for k in range(1, K + 1):
        c2[k - 1] = float(np.sum((h @ (cur * wts)) ** 2))
        prev, cur = cur, (x * cur - np.sqrt(k) * prev) / np.sqrt(k + 1)
    pw = np.arange(1, K + 1)
    return lambda rho: float(np.sum(c2 * float(rho) ** pw) / V)


def _shuffle_map(tp: "Template", frames: tuple[float, float], rng) -> np.ndarray:
    """A permutation of the template's frames: blocks of frames (length
    uniform in frames, each cut moved to the next quiet frame - summed activity
    in its lowest 30% - so events stay whole) in random order."""
    T = tp.T
    a = tp.act_trace if tp.act_trace is not None else np.zeros(T)
    quiet = a <= np.quantile(a, 0.3)
    cuts = [0]
    while True:
        nxt = cuts[-1] + int(rng.uniform(*frames))
        if nxt >= T:
            break
        q = np.flatnonzero(quiet[nxt: nxt + int(frames[0])])
        nxt += int(q[0]) if len(q) else 0
        if nxt >= T:
            break
        cuts.append(nxt)
    cuts.append(T)
    blocks = [np.arange(c0, c1) for c0, c1 in zip(cuts[:-1], cuts[1:])]
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])


def draw(tp: Template, cfg: SimConfig, rng) -> dict:
    """One virtual trial: activity shift and amplitude, drift processes and fields."""
    T, fps = tp.T, cfg.rate_hz
    t = np.arange(T) / fps
    tau_b = rng.uniform(*cfg.bleach_tau_s)
    bleach = -rng.uniform(0, cfg.bleach) * (1 - np.exp(-t / tau_b)) / (1 - np.exp(-t[-1] / tau_b) + 1e-9)
    K = cfg.n_fields
    taus = np.concatenate([[rng.uniform(*cfg.global_tau_s)], rng.uniform(*cfg.field_tau_s, K)]) * fps
    sds = np.concatenate([[rng.uniform(0, cfg.global_sd)], rng.uniform(0, cfg.field_sd, K)])
    ou = _ou(T, taus, sds, rng)
    proc = np.vstack([bleach + ou[0], ou[1:]]).astype(np.float32)  # (1 + K, T): global, then per field
    fields = np.stack([_field(tp.R, tp.C, rng.uniform(*cfg.field_px), rng) for _ in range(K)]) if K else \
        np.zeros((0, tp.R, tp.C), np.float32)
    cov = np.atleast_2d(np.cov(proc))
    r = {"shift": int(rng.integers(T)), "amp": float(rng.uniform(*cfg.amp)), "proc": proc, "fields": fields,
         "proc_mean": proc.mean(1), "proc_cov": cov, "excess": float(cfg.excess)}
    if cfg.excess > 0 and cfg.n_slow > 0 and tp.slow.any():
        # slow processes with unit variance per pixel: fields normalized so that sum_k field_k^2 = 1
        ks = cfg.n_slow
        taus_s = np.exp(rng.uniform(*np.log(cfg.slow_tau_s), ks)) * fps
        sp = _ou(T, taus_s, np.ones(ks), rng).astype(np.float32)
        sf = np.stack([_field(tp.R, tp.C, rng.uniform(*cfg.slow_px), rng) for _ in range(ks)])
        sf /= np.sqrt((sf ** 2).sum(0)) + 1e-6
        r.update(slow_proc=sp, slow_fields=sf, slow_mean=sp.mean(1), slow_cov=np.atleast_2d(np.cov(sp)))
    if cfg.silent_trial > 0 or cfg.silent_frac[1] > 0:  # drawn last, so the draws above stay as they were
        if rng.random() < cfg.silent_trial:
            r["act_mask"] = np.zeros((tp.R, tp.C), np.float32)
        else:
            frac = rng.uniform(*cfg.silent_frac)
            fld = _field(tp.R, tp.C, rng.uniform(*cfg.silent_px), rng)
            v = tp.tf.valid
            thr = np.quantile(fld[v], frac) if v.any() and frac > 0 else -np.inf
            r["act_mask"] = (fld > thr).astype(np.float32)
    if cfg.fast_global_sd > 0 or (cfg.fast_field_sd > 0 and cfg.n_fast_fields > 0):  # drawn last, see above
        kf = cfg.n_fast_fields if cfg.fast_field_sd > 0 else 0
        taus_f = np.exp(rng.uniform(*np.log(cfg.fast_tau_frames), 1 + kf))
        sds_f = np.concatenate([[rng.uniform(0, cfg.fast_global_sd)], rng.uniform(0, cfg.fast_field_sd, kf)])
        fp = _ou(T, taus_f, sds_f, rng).astype(np.float32)
        proc = r["proc"].copy()
        proc[0] += fp[0]  # the global process
        if kf:
            ff = np.stack([_field(tp.R, tp.C, rng.uniform(*cfg.fast_field_px), rng) for _ in range(kf)])
            proc = np.vstack([proc, fp[1:]])
            r["fields"] = np.concatenate([r["fields"], ff]) if len(r["fields"]) else ff
        r.update(proc=proc, proc_mean=proc.mean(1), proc_cov=np.atleast_2d(np.cov(proc)))
    if cfg.dark_frac[1] > 0:  # drawn last, see above
        frac = rng.uniform(*cfg.dark_frac)
        sup = tp.tf.support & tp.tf.valid
        dark = np.ones((tp.R, tp.C), np.float32)
        if sup.any() and frac > 0:
            fld = _field(tp.R, tp.C, rng.uniform(*cfg.dark_px), rng)
            thr = np.quantile(fld[sup], 1 - frac)
            pick = sup & (fld >= thr)
            dark[pick] = rng.uniform(*cfg.dark_factor, int(pick.sum())).astype(np.float32)
        r["dark"] = dark
    if cfg.shuffle_prob > 0 and rng.random() < cfg.shuffle_prob:  # drawn last, see above
        r["tmap"] = _shuffle_map(tp, cfg.shuffle_frames, rng)
    if cfg.motion_sd_px[1] > 0:  # drawn last, see above
        taus_m = np.exp(rng.uniform(*np.log(cfg.motion_tau_frames), 2))
        sds_m = rng.uniform(*cfg.motion_sd_px, 2)
        n = int(cfg.motion_substeps)
        if n <= 0:
            r["motion"] = _ou(T, taus_m, sds_m, rng).astype(np.float32)  # (2, T): rows, cols in pixels
        else:  # drawn at n steps per frame, pooled by the frame-formation kernel
            h = n * (len(tp.kt) // 2)
            d = _ou(T * n + 2 * h, taus_m * n, sds_m, rng)
            r["motion"], r["motion_cov"] = _pooled_motion(d, tp.kt, n, T)
    if cfg.focus_sd_px2[1] > 0:  # drawn last, see above
        sd = rng.uniform(*cfg.focus_sd_px2)
        sds = np.array([sd, cfg.focus_aniso * sd, cfg.focus_aniso * sd]) / np.sqrt(2)  # isotropic, rows, cols
        b = 0.0
        for lo_hi in cfg.focus_tau_frames:  # fast and slow, equal variance
            b = b + _ou(T, np.exp(rng.uniform(*np.log(lo_hi), 3)), sds, rng)
        r["focus"] = np.stack([b[0] + b[1], b[0] + b[2]]).astype(np.float32)  # (2, T): b_r, b_c in px^2
    if cfg.local_sd_px[1] > 0 and cfg.n_local_fields > 0:  # drawn last, see above
        J = cfg.n_local_fields
        flds = np.stack([np.stack([_field(tp.R, tp.C, rng.uniform(*cfg.local_px), rng) for _ in range(J)])
                         for _ in range(2)])  # (2, J, R, C), unit SD each
        flds /= np.sqrt(J)  # unit-variance weights then give a displacement of unit SD per pixel on average
        w = 0.0
        for lo_hi in cfg.local_tau_frames:  # fast and slow, each with its own SD
            w = w + _ou(T, np.exp(rng.uniform(*np.log(lo_hi), 2 * J)),
                        np.full(2 * J, rng.uniform(*cfg.local_sd_px)), rng)
        r["local_fields"] = flds.astype(np.float32)
        r["local_proc"] = w.reshape(2, J, T).astype(np.float32)  # (2, J, T)
    if cfg.photon_gain != (1.0, 1.0):  # drawn last, see above
        lo, hi = (float(v) for v in cfg.photon_gain)
        r["gain_k"] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if hi > lo else lo
    if cfg.photon_zeros and tp.zc is not None:
        r["photon_zeros"] = True
        r["pz_legacy"] = bool(cfg.pz_legacy)
    return r


def _drift_maps(r: dict, rows, cols) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel mean and variance over the trial of the drift factor (minus 1)."""
    f = r["fields"][:, rows][:, :, cols]  # (K, nr, nc)
    coef = np.concatenate([np.ones((1,) + f.shape[1:], np.float32), f])  # weights of the processes
    mean = np.tensordot(r["proc_mean"], coef, 1)
    var = np.einsum("kl,kij,lij->ij", r["proc_cov"], coef, coef)
    return mean.astype(np.float32), np.maximum(var, 0).astype(np.float32)


def window(tp: Template, r: dict, t0: int, nt: int, r0: int, nr: int, c0: int, nc: int, rng) -> dict:
    """A window of the virtual trial; arrays in counts, NaN (movie) or 0 outside the trial."""
    T, R, C = tp.T, tp.R, tp.C
    tf = tp.tf
    ts, te = max(t0, 0), min(t0 + nt, T)
    rs, re = max(r0, 0), min(r0 + nr, R)
    cs, ce = max(c0, 0), min(c0 + nc, C)
    out = {k: np.full((nt, nr, nc), np.nan, np.float32) for k in ("movie", "truth", "noisy")}
    out["activity"] = np.zeros((nt, nr, nc), np.float32)
    for k in ("mean_image", "mean_act", "var_map"):
        out[k] = np.full((nr, nc), np.nan, np.float32)
    out["varfac"] = np.zeros((nr, nc), np.float32)
    out["labeled"] = np.zeros((nr, nc), bool)
    out["support"] = np.zeros((nr, nc), bool)
    if not (ts < te and rs < re and cs < ce):
        return out
    dt, dr, dc = slice(ts - t0, te - t0), slice(rs - r0, re - r0), slice(cs - c0, ce - c0)
    rows, cols = np.arange(rs, re), np.arange(cs, ce)
    valid = tf.valid[rs:re, cs:ce]
    # the scene is computed on the window plus the margin that motion can move into it
    mo = r.get("motion")
    moving = mo is not None or "focus" in r or "local_fields" in r
    reach = np.abs(mo).max() if mo is not None else 0.0
    if "local_fields" in r:  # bound on the local displacement: max |field| times the summed max |weight|
        reach += float(np.abs(r["local_fields"]).max() * np.abs(r["local_proc"]).max(2).sum(1).max())
    m = int(np.ceil(reach)) + 3 if reach > 0 else (2 if moving else 0)
    xs, xe, ys, ye = max(rs - m, 0), min(re + m, R), max(cs - m, 0), min(ce + m, C)
    loc = r["local_fields"][:, :, xs:xe, ys:ye] if "local_fields" in r else None
    wr, wc = slice(rs - xs, re - xs), slice(cs - ys, ce - ys)  # the window within that region
    valid_x = tf.valid[xs:xe, ys:ye]
    B = np.nan_to_num(tp.B[xs:xe, ys:ye])
    dk = r["dark"][xs:xe, ys:ye] if "dark" in r else None
    if dk is not None:  # dark sources: less baseline, the same activity
        B = B * dk
    # drift factor on the window
    f = r["fields"][:, xs:xe, ys:ye]
    p = r["proc"][:, ts:te]
    d = 1.0 + p[0][:, None, None] + (np.tensordot(p[1:].T, f, 1) if len(f) else 0.0)
    truth = B[None] * d
    # the excess's slow part: baseline fluctuations with the template's per-pixel variance
    sa = np.sqrt(tp.slow[xs:xe, ys:ye] * r.get("excess", 0.0))
    if dk is not None:
        sa = sa * dk  # the slow excess scales with the baseline it rides on
    if "slow_fields" in r:
        sg = r["slow_fields"][:, xs:xe, ys:ye] * sa[None]
        truth = truth + np.tensordot(r["slow_proc"][:, ts:te].T, sg, 1)
    # activity, shifted circularly in time
    src = (np.arange(ts, te) - r["shift"]) % T
    if "tmap" in r:  # shuffled timeline
        src = r["tmap"][src]
    act = np.empty((te - ts, xe - xs, ye - ys), np.float32)
    brk = np.flatnonzero(np.diff(src) != 1)
    pieces = np.split(np.arange(te - ts), brk + 1)
    for idx in pieces:
        a, b = src[idx[0]], src[idx[-1]] + 1
        act[idx] = tf._f["activity"][tf.channel, a:b, xs:xe, ys:ye]
    amask = r["act_mask"][xs:xe, ys:ye] if "act_mask" in r else np.float32(1.0)  # 0 in silent regions
    act *= r["amp"] * amask
    act[:, ~valid_x] = 0
    if tp.unfit is not None:  # as in the sanitized stats: no activity at GIAnT's fit artifacts
        act[:, tp.unfit[xs:xe, ys:ye]] = 0  # (on 804730 path 2 up to 3e5 counts on a movie of ~13)
    fill = None
    if moving:  # the sample moves (or changes focus), the scan does not
        fill = _fill_nearest(valid_x)
        out["truth_static"] = np.full((nt, nr, nc), np.nan, np.float32)
        out["truth_static"][dt, dr, dc] = np.where(valid[None], truth[:, wr, wc], np.nan)
        truth = np.stack([_move(truth[i], r, ts + i, fill, loc)[wr, wc] for i in range(te - ts)])
        act = np.stack([_move(np.nan_to_num(act[i]), r, ts + i, fill, loc)[wr, wc] for i in range(te - ts)])
        act[:, ~valid] = 0
    else:  # the region is the window
        truth, act = truth[:, wr, wc], act[:, wr, wc]
    S = np.where(valid[None], truth + act, 0).astype(np.float32)
    gain = tp.gain * r.get("gain_k", 1.0)  # counts per photon (photon_gain: fewer photons for the same counts)
    out["gain"] = np.float32(gain)
    # photon noise through the kernel; the signal itself is not blurred
    pt, ps_r, ps_c = len(tp.kt) // 2, len(tp.kr) // 2, len(tp.kc) // 2
    Se = np.pad(S, ((pt, pt), (ps_r, ps_r), (ps_c, ps_c)), mode="edge")
    gp = gain * tp.q / tp.ksum2  # event size; variance gp^2 * lam * ksum2 = gain * S
    lam = np.maximum(Se, 0) * tp.ksum2 / (gain * tp.q ** 2)
    noise = _filter((gp * (rng.poisson(lam) - lam)).astype(np.float32), tp.kt, tp.kr, tp.kc)
    fac = np.maximum(1 + tp.fast[rs:re, cs:ce] * r.get("excess", 0.0), 0.05)  # the excess's fast part
    movie = S + noise * np.sqrt(fac)[None]
    movie[:, ~valid] = np.nan
    movie[tf.discard[ts:te]] = np.nan
    truth = np.where(valid[None], truth, np.nan)
    out["movie"][dt, dr, dc] = movie
    out["truth"][dt, dr, dc] = truth
    out["noisy"][dt, dr, dc] = movie - act
    out["activity"][dt, dr, dc] = act
    # whole-trial inputs from the virtual trial's statistics plus sampling noise
    dmean, dvar = _drift_maps(r, rows, cols)
    am = amask[wr, wc] if np.ndim(amask) else amask
    mact, vact = tp.mean_act[rs:re, cs:ce] * r["amp"] * am, tp.var_act[rs:re, cs:ce] * (r["amp"] * am) ** 2
    Bx, B = B, B[wr, wc]
    M = B * (1 + dmean) + mact
    svar = np.zeros_like(M)
    if "slow_fields" in r:
        M = M + np.tensordot(r["slow_mean"], sg[:, wr, wc], 1)
        svar = np.maximum(np.einsum("kl,kij,lij->ij", r["slow_cov"], sg[:, wr, wc], sg[:, wr, wc]), 0)
    vmot = 0.0
    if moving:  # the moving trial's mean and variance: the static mean moved by 128 of its displacements
        dmx, _ = _drift_maps(r, np.arange(xs, xe), np.arange(ys, ye))
        Mx = Bx * (1 + dmx) + tp.mean_act[xs:xe, ys:ye] * r["amp"] * amask
        if "slow_fields" in r:
            Mx = Mx + np.tensordot(r["slow_mean"], sg, 1)
        kept = np.flatnonzero(~tf.discard) if (~tf.discard).any() else np.arange(T)
        sh = np.stack([_move(Mx, r, k, fill, loc)[wr, wc] for k in kept[np.linspace(0, len(kept) - 1, 128).astype(int)]])
        M, vmot = sh.mean(0), sh.var(0)
    var_mean = gain * np.maximum(M, 0) * fac * tp.Ft / tp.n_valid
    g = rng.standard_normal((re - rs + 2 * ps_r, ce - cs + 2 * ps_c)).astype(np.float32)[None]
    g = _filter(g, np.ones(1), tp.kr, tp.kc)[0] / np.sqrt((tp.kr ** 2).sum() * (tp.kc ** 2).sum())
    mi = M + np.sqrt(var_mean) * g
    vm = (B ** 2 * dvar + svar + vact + vmot + gain * np.maximum(M, 0) * fac) * \
        np.maximum(1 + np.sqrt(2 * tp.Ft / tp.n_valid) * rng.standard_normal(M.shape), 0.1)
    out["mean_image"][dr, dc] = np.where(valid, mi, np.nan)
    out["mean_act"][dr, dc] = np.where(valid, mact, np.nan)  # the virtual trial's mean activity
    out["var_map"][dr, dc] = np.where(valid, vm, np.nan)
    out["varfac"][dr, dc] = np.nan_to_num(tf.varfac[rs:re, cs:ce])
    out["labeled"][dr, dc] = tf.labeled[rs:re, cs:ce] & valid
    out["support"][dr, dc] = tf.support[rs:re, cs:ce] & valid
    if r.get("photon_zeros"):  # last, so the draws above are as without it
        ge = rng.standard_normal(Se.shape).astype(np.float32)
        var = gain * np.maximum(S, 0) * fac[None]
        zc = np.broadcast_to(tp.zc[rs:re, cs:ce] * np.float32(r.get("gain_k", 1.0)), S.shape)
        if r.get("pz_legacy"):
            g = _filter(ge, tp.kt, tp.kr, tp.kc) / np.sqrt(tp.ksum2)
            mz = _photon_zeros(movie, S, var, zc, g)
        else:  # every voxel, through one field whose kernels make the counts correlate as the template
            kt, kr, kc, ks2 = tp.pz_kernels(r.get("excess", 0.0))
            g = _filter(ge, kt, kr, kc) / np.float32(np.sqrt(ks2))
            use = np.isfinite(movie) & (S > 0)
            mz = movie.copy()
            mz[use] = _photon_counts(S[use].astype(np.float64), var[use].astype(np.float64),
                                     zc[use].astype(np.float64), g[use].astype(np.float64))
        out["movie"][dt, dr, dc] = mz
        out["noisy"][dt, dr, dc] = mz - act
    return out


def write_trial(tp: Template, out_fn, cfg: SimConfig | None = None, seed: int = 0, compress: bool = True) -> dict:
    """A whole simulated trial in the export format, plus truth/baseline (for checks and
    for running the ceiling and stats tools on simulations)."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    r = draw(tp, cfg, rng)
    w = window(tp, r, 0, tp.T, 0, tp.R, 0, tp.C, rng)
    tf = tp.tf
    with h5py.File(out_fn, "w") as f:
        kw = dict(chunks=(1, min(256, tp.T), min(64, tp.R), min(64, tp.C)))
        if compress:
            kw.update(compression="gzip", compression_opts=4)
        f.create_dataset("movie", data=w["movie"][None], **kw)
        f.create_dataset("activity", data=w["activity"][None], **kw)
        f.create_dataset("truth/baseline", data=w["truth"], chunks=kw["chunks"][1:],
                         **{k: v for k, v in kw.items() if k != "chunks"})
        if "truth_static" in w:  # the baseline without motion and focus changes; the displacement (rows, cols; pixels)
            f.create_dataset("truth/baseline_static", data=w["truth_static"], chunks=kw["chunks"][1:],
                             **{k: v for k, v in kw.items() if k != "chunks"})
        if "motion" in r:
            f["truth/motion"] = r["motion"]
            if "motion_cov" in r:  # spread of the displacement within each frame's kernel (px^2: rows, cols, rows-cols)
                f["truth/motion_cov"] = r["motion_cov"]
        if "focus" in r:  # focus change per frame (px^2: b_r, b_c)
            f["truth/focus"] = r["focus"]
        if "local_fields" in r:  # local displacement = sum_j fields[:, j] * proc[:, j, t] (rows, cols; pixels)
            f["truth/local_fields"] = r["local_fields"].astype(np.float16)
            f["truth/local_proc"] = r["local_proc"]
        f["mean_image"] = w["mean_image"][None]
        f["varfac"] = tf.varfac
        f["mask/valid"] = tf.valid.astype(np.uint8)
        f["mask/labeled"] = tf.labeled.astype(np.uint8)
        f["mask/footprint_support"] = tf.support.astype(np.uint8)
        f["frame/discard"] = tf.discard.astype(np.uint8)
        for k, v in tf.attrs.items():
            try:
                f.attrs[k] = v
            except TypeError:
                pass
        f.attrs["simulated"] = 1
        f.attrs["template"] = str(tf.path)
        f.attrs["sim_amp"] = r["amp"]
        f.attrs["sim_gain"] = float(w.get("gain", tp.gain))  # counts per photon used (photon_gain)
        f.attrs["sim_config"] = str(asdict(cfg))
    return r


def add_check_arguments(p) -> None:
    p.add_argument("root", help="folder with exported trial files (templates)")
    p.add_argument("--out", default="results/simcheck", help="simulated trials and simcheck.csv go here")
    p.add_argument("--stats_cache", default="", help="folder for cached trial stats (default: <out>/stats)")
    p.add_argument("--max_files", type=int, default=1, help="templates per session; 0 = all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep", action="store_true", help="keep the simulated trial files (about 2x the export's size)")


def check(args) -> None:
    """Simulate whole trials from templates and compare their measured noise
    statistics with the templates' (the simulator should reproduce them)."""
    import csv
    from pathlib import Path

    from .io import list_trials
    from .stats import SCALARS, load_stats, trial_stats

    out = Path(args.out)
    files = list_trials(args.root)
    if args.max_files:
        per, keep = {}, []
        for f in files:
            s = f.parent.parent.name
            if per.get(s, 0) < args.max_files:
                keep.append(f)
                per[s] = per.get(s, 0) + 1
        files = keep
    stats = load_stats(files, args.stats_cache or out / "stats")
    rows = []
    for k, fn in enumerate(files):
        st = stats[str(fn)]
        tp = Template(TrialFile(fn), st)
        sim_fn = out / fn.parent.parent.name / fn.parent.name / fn.name
        sim_fn.parent.mkdir(parents=True, exist_ok=True)
        write_trial(tp, sim_fn, seed=args.seed + k)
        ss = trial_stats(sim_fn)
        if not args.keep:
            sim_fn.unlink()  # whole simulated trials are large; keep only the statistics
        row = {"template": f"{fn.parent.parent.name}/{fn.parent.name}/{fn.name}",
               "kernel_t": " ".join(f"{v:.3f}" for v in tp.kt), "a_r": float(tp.kr[0] / tp.kr[1]),
               "a_c": float(tp.kc[0] / tp.kc[1]), "Ft_kernel": tp.Ft, "skew_q": tp.q}
        for s in SCALARS:
            row[f"real_{s}"], row[f"sim_{s}"] = st[s], ss[s]
        rows.append(row)
        print(row["template"] + ": " + " | ".join(f"{s} {st[s]:.3f}/{ss[s]:.3f}" for s in SCALARS if s != "n_frames")
              + f" | kernel F {tp.Ft:.2f}, skew factor {tp.q:.2f}", flush=True)
    with open(out / "simcheck.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'simcheck.csv'} (real/sim)")


def excess_summary(tf: TrialFile, st: dict) -> dict[str, float]:
    """How much more the trial's residual (movie - activity) varies than photon
    noise: the measured variance of single frames and of 8- and 64-frame means
    over the photon-noise prediction (gain * mean, through the trial's noise
    kernel), median over pixel classes (all valid, background, labeled,
    footprint support) and over all or the brightest 10% of valid pixels; plus the
    simulator's fast and slow excess on the brightest 10% (slow as SD over the
    mean). NaN without the residual maps (older stats caches)."""
    out: dict[str, float] = {}
    if "var64_res" not in st:
        return out
    kt = noise_kernels(st)[0]
    mm = np.nan_to_num(st["mean_movie"])
    ph = st["gain"] * np.maximum(mm, 0)
    valid = tf.valid & np.isfinite(st["var_res"]) & (ph > 0)
    if not valid.any():
        return out
    top = valid & (mm >= np.quantile(mm[valid], 0.9))
    classes = {"all": valid, "bg": valid & ~tf.labeled, "lab": valid & tf.labeled, "sup": valid & tf.support}
    with np.errstate(invalid="ignore", divide="ignore"):
        for n, key in ((1, "var_res"), (8, "var8_res"), (64, "var64_res")):
            ratio = st[key] / (ph * kernel_var(kt, n))
            fin = np.isfinite(ratio)  # 64-frame means need at least 8 blocks
            for cname, cm in classes.items():
                for bname, bm in (("", cm & fin), ("top10_", cm & top & fin)):
                    out[f"{cname}_{bname}r{n}"] = float(np.median(ratio[bm])) if bm.any() else float("nan")
    fast, slow = excess_maps(st, kt, tf.valid)
    out["top10_fast"] = float(np.median(fast[top]))
    out["top10_slow_sd_rel"] = float(np.median(np.sqrt(slow[top]) / np.maximum(mm[top], 1e-6)))
    out["top10_mean_over_scale"] = float(np.median(mm[top]) / tf.scale)
    out["valid_px"] = float(valid.sum())
    out["support_px"] = float((valid & tf.support).sum())
    return out
