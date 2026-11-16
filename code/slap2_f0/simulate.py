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
* **silent regions** (optional, ``silent_trial`` and ``silent_frac``): no
  activity in random smooth patches of the field, or anywhere in the virtual
  trial. Bright labeled pixels then vary (the excess) without any events, as
  in a GFP control, so the template's footprint geometry no longer says which
  pixels are active and only the events themselves do. Without them, a
  network fine-tuned on sensor sessions (where every bright pixel in the loss
  has activity) set the baseline of GFP dendrites too low (v4 run).

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
    slow_px: tuple[float, float] = (2.0, 6.0)  # spatial scale of their fields
    # activity-free regions: with probability silent_trial a virtual trial has no
    # activity at all; otherwise a fraction (uniform in silent_frac) of its valid
    # pixels, in smooth patches of silent_px, has none (the defaults keep all)
    silent_trial: float = 0.0
    silent_frac: tuple[float, float] = (0.0, 0.0)
    silent_px: tuple[float, float] = (20.0, 60.0)


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
    B = np.nan_to_num(tp.B[rs:re, cs:ce])
    # drift factor on the window
    f = r["fields"][:, rs:re, cs:ce]
    p = r["proc"][:, ts:te]
    d = 1.0 + p[0][:, None, None] + (np.tensordot(p[1:].T, f, 1) if len(f) else 0.0)
    truth = B[None] * d
    # the excess's slow part: baseline fluctuations with the template's per-pixel variance
    sa = np.sqrt(tp.slow[rs:re, cs:ce] * r.get("excess", 0.0))
    if "slow_fields" in r:
        sg = r["slow_fields"][:, rs:re, cs:ce] * sa[None]
        truth = truth + np.tensordot(r["slow_proc"][:, ts:te].T, sg, 1)
    # activity, shifted circularly in time
    src = (np.arange(ts, te) - r["shift"]) % T
    act = np.empty((te - ts, re - rs, ce - cs), np.float32)
    brk = np.flatnonzero(np.diff(src) < 0)
    pieces = np.split(np.arange(te - ts), brk + 1)
    for idx in pieces:
        a, b = src[idx[0]], src[idx[-1]] + 1
        act[idx] = tf._f["activity"][tf.channel, a:b, rs:re, cs:ce]
    amask = r["act_mask"][rs:re, cs:ce] if "act_mask" in r else np.float32(1.0)  # 0 in silent regions
    act *= r["amp"] * amask
    act[:, ~valid] = 0
    S = np.where(valid[None], truth + act, 0).astype(np.float32)
    # photon noise through the kernel; the signal itself is not blurred
    pt, ps_r, ps_c = len(tp.kt) // 2, len(tp.kr) // 2, len(tp.kc) // 2
    Se = np.pad(S, ((pt, pt), (ps_r, ps_r), (ps_c, ps_c)), mode="edge")
    gp = tp.gain * tp.q / tp.ksum2  # event size; variance gp^2 * lam * ksum2 = gain * S
    lam = np.maximum(Se, 0) * tp.ksum2 / (tp.gain * tp.q ** 2)
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
    mact, vact = tp.mean_act[rs:re, cs:ce] * r["amp"] * amask, tp.var_act[rs:re, cs:ce] * (r["amp"] * amask) ** 2
    M = B * (1 + dmean) + mact
    svar = np.zeros_like(M)
    if "slow_fields" in r:
        M = M + np.tensordot(r["slow_mean"], sg, 1)
        svar = np.maximum(np.einsum("kl,kij,lij->ij", r["slow_cov"], sg, sg), 0)
    var_mean = tp.gain * np.maximum(M, 0) * fac * tp.Ft / tp.n_valid
    g = rng.standard_normal((re - rs + 2 * ps_r, ce - cs + 2 * ps_c)).astype(np.float32)[None]
    g = _filter(g, np.ones(1), tp.kr, tp.kc)[0] / np.sqrt((tp.kr ** 2).sum() * (tp.kc ** 2).sum())
    mi = M + np.sqrt(var_mean) * g
    vm = (B ** 2 * dvar + svar + vact + tp.gain * np.maximum(M, 0) * fac) * \
        np.maximum(1 + np.sqrt(2 * tp.Ft / tp.n_valid) * rng.standard_normal(M.shape), 0.1)
    out["mean_image"][dr, dc] = np.where(valid, mi, np.nan)
    out["mean_act"][dr, dc] = np.where(valid, mact, np.nan)  # the virtual trial's mean activity
    out["var_map"][dr, dc] = np.where(valid, vm, np.nan)
    out["varfac"][dr, dc] = np.nan_to_num(tf.varfac[rs:re, cs:ce])
    out["labeled"][dr, dc] = tf.labeled[rs:re, cs:ce] & valid
    out["support"][dr, dc] = tf.support[rs:re, cs:ce] & valid
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
