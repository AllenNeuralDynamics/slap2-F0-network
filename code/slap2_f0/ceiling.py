"""How much is there to learn? Reference baselines on exported trials.

For each trial, compares simple baseline predictors against the training
target y = (movie - activity) / scale, on the pixels the loss uses, split by
pixel class (footprint support, unlabeled background). The predictors bracket
what the network can achieve:

    mean_image         the network's starting point (whole-trial movie mean,
                       so it includes the mean activity)
    target_mean        oracle: per-pixel whole-trial mean of the target (the
                       best constant baseline; mean_image minus mean activity)
    blind_target       oracle: per-pixel mean of the target over the network's
                       temporal context (gap < |dt| <= gap + reach), i.e. a
                       blind-spot temporal average of residuals with no events
    blind_target_3x3   the same, averaged over 3 x 3 pixels
    blind_movie        the same average of the movie (what the network sees:
                       its context still holds events)
    blind_movie_3x3
    clipped_mean       non-learned candidate: whole-trial movie mean leaving out
                       frames whose 8-frame average is > 2 sigma above it
    silo               today's SiLo localization baseline (moving mean over 8
                       frames, then moving median over 321), on a subsample

All predictors share the target's photon noise, so differences in mean squared
error are differences in baseline error. ``target_mean`` vs ``mean_image`` is
the mean activity the network must remove; ``blind_target`` vs ``target_mean``
is how much the baseline drifts on the time scale of the context (minus the
extra estimation noise of a shorter average); ``blind_movie`` vs
``blind_target`` is the cost of events in the context.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from .io import list_trials


def _blind_mean(v: np.ndarray, n: np.ndarray, gap: int, reach: int) -> np.ndarray:
    """Mean over frames gap < |dt| <= gap + reach along axis 0 (NaN where empty).
    v holds values with missing entries as 0, n the counts (0/1 or sums)."""
    T = v.shape[0]
    cs = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(v, 0, dtype=np.float64)])
    cn = np.concatenate([np.zeros((1,) + n.shape[1:]), np.cumsum(n, 0, dtype=np.float64)])
    t = np.arange(T)

    def box(a, b):  # inclusive [a, b], clipped
        a = np.clip(a, 0, T)
        b = np.clip(b + 1, 0, T)
        return cs[b] - cs[a], cn[b] - cn[a]

    s1, n1 = box(t - gap - reach, t - gap - 1)
    s2, n2 = box(t + gap + 1, t + gap + reach)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((s1 + s2) / (n1 + n2)).astype(np.float32)


def _movmean(a: np.ndarray, k: int) -> np.ndarray:
    """NaN-aware moving mean over k frames along axis 0 (SiLo's IMs for k = 8)."""
    if k <= 1:
        return a
    T = a.shape[0]
    fin = np.isfinite(a)
    cs = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(np.where(fin, a, 0), 0, dtype=np.float64)])
    cn = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(fin, 0, dtype=np.float64)])
    lo, hi = k // 2, k - k // 2
    i0 = np.clip(np.arange(T) - lo, 0, T)
    i1 = np.clip(np.arange(T) + hi, 0, T)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = ((cs[i1] - cs[i0]) / (cn[i1] - cn[i0])).astype(np.float32)
    out[~fin] = np.nan
    return out


def clipped_mean(m: np.ndarray, scale: float, background: np.ndarray, k: float = 2.0, smooth: int = 8,
                 dilate: int = 8, iters: int = 3) -> np.ndarray:
    """Whole-trial mean per pixel with event frames left out: frames where the
    `smooth`-frame average exceeds the current estimate by k sigma (photon
    noise model, var = g * mean in counts, g and the averaging factor measured
    on background pixels), widened by `dilate` frames each side. m is the movie
    in units of `scale` (frames x rows x cols, NaN = missing)."""
    fin = np.isfinite(m)
    ims = _movmean(m, smooth)
    bg = m[:, background]
    mu_bg, var_bg = np.nanmean(bg, 0), np.nanvar(bg, 0)
    ok = mu_bg > 0
    g = float(np.median(var_bg[ok] / mu_bg[ok])) * scale  # counts per photon-ish
    c = float(np.median(np.nanvar(ims[:, background], 0)[ok] / var_bg[ok]))  # variance factor of the average
    mu = np.nanmean(m, 0)
    for _ in range(iters):
        sig = np.sqrt(np.maximum(c * g * mu / scale, 1e-12))
        flag = (ims > mu + k * sig).astype(np.float32)
        flag = (_movmean(flag, 2 * dilate + 1) if dilate else flag) > 0
        keep = fin & ~flag
        mu = np.where(keep, m, 0).sum(0) / np.maximum(keep.sum(0), 1)
    return mu.astype(np.float32)


def _box3(v: np.ndarray) -> np.ndarray:
    """3 x 3 spatial box sum over the last two axes (zero padded)."""
    p = np.pad(v, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros_like(v)
    for dr in range(3):
        for dc in range(3):
            out += p[:, dr: dr + v.shape[1], dc: dc + v.shape[2]]
    return out


def _silo(m: np.ndarray, smooth: int = 8, window: int = 321, stride: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """SiLo baseline for columns of m (frames x pixels) at every ``stride``-th frame."""
    T = m.shape[0]
    fin = np.isfinite(m)
    cs = np.concatenate([np.zeros((1, m.shape[1])), np.cumsum(np.where(fin, m, 0), 0)])
    cn = np.concatenate([np.zeros((1, m.shape[1])), np.cumsum(fin, 0)])
    lo, hi = smooth // 2, smooth - smooth // 2
    a = np.clip(np.arange(T) - lo, 0, T)
    b = np.clip(np.arange(T) + hi, 0, T)
    with np.errstate(invalid="ignore", divide="ignore"):
        mm = (cs[b] - cs[a]) / (cn[b] - cn[a])
    half = window // 2
    pad = np.full((half, m.shape[1]), np.nan)
    mp = np.concatenate([pad, mm, pad])
    frames = np.arange(0, T, stride)
    out = np.empty((len(frames), m.shape[1]), np.float32)
    for k, t in enumerate(frames):
        out[k] = np.nanmedian(mp[t: t + window], axis=0)
    return frames, out


def trial_ceiling(fn: Path, gap: int = 8, reach: int = 254, channel: int = 0, silo_pixels: int = 300,
                  seed: int = 0, smooth: int = 1) -> list[dict]:
    """Rows of reference-baseline errors per pixel class. With smooth > 1 the
    target (and time-varying predictors) are first averaged over that many
    frames, as SiLo does before Z-scoring (8 at 80 Hz): photon noise shrinks
    and baseline errors weigh more."""
    with h5py.File(fn, "r") as f:
        movie = f["movie"][channel].astype(np.float32)
        act = f["activity"][channel].astype(np.float32)
        mean_image = f["mean_image"][channel].astype(np.float32)
        valid = f["mask/valid"][:].astype(bool) & np.isfinite(mean_image)
        labeled = f["mask/labeled"][:].astype(bool)
        support = f["mask/footprint_support"][:].astype(bool)
        discard = f["frame/discard"][:].astype(bool)
        attrs = {k: f.attrs[k] for k in f.attrs}
    label = attrs.get("label", fn.parent.parent.name)
    label = label.decode() if isinstance(label, bytes) else str(np.asarray(label).squeeze())
    scale = float(np.nanmedian(mean_image[valid]))
    movie[:, ~valid] = np.nan
    movie[discard] = np.nan
    y = (movie - act) / scale
    m = movie / scale
    mi = mean_image / scale
    fin = np.isfinite(y)
    n = fin.astype(np.float32)
    y0 = np.where(fin, y, 0).astype(np.float32)
    m0 = np.where(fin, m, 0).astype(np.float32)

    preds = {
        "mean_image": None,
        "target_mean": (y0.sum(0) / np.maximum(n.sum(0), 1)).astype(np.float32),
        "blind_target": _blind_mean(y0, n, gap, reach),
        "blind_target_3x3": _blind_mean(_box3(y0), _box3(n), gap, reach),
        "blind_movie": _blind_mean(m0, n, gap, reach),
        "blind_movie_3x3": _blind_mean(_box3(m0), _box3(n), gap, reach),
        "clipped_mean": clipped_mean(m, scale, valid & ~labeled),
    }

    if smooth > 1:
        y = _movmean(y, smooth)
        for k, p in preds.items():
            if p is not None and p.ndim == 3:
                preds[k] = _movmean(p, smooth)

    classes = {"support": valid & support, "unlabeled": valid & ~labeled,
               "loss": valid & (support | ~labeled), "labeled_unfitted": valid & labeled & ~support}
    rng = np.random.default_rng(seed)
    rows = []
    for cname, cmask in classes.items():
        if not cmask.any():
            continue
        yy = y[:, cmask]
        ok = np.isfinite(yy)
        row = {"file": str(fn), "label": label, "class": cname, "smooth": smooth, "n_pixels": int(cmask.sum()),
               "n_frames": int(y.shape[0]), "scale": scale, "gain": float(np.asarray(attrs.get("activityGain", np.nan)).ravel()[0]),
               "target_mean": float(np.nanmean(yy)), "mean_image_mean": float(mi[cmask].mean()),
               "activity_mean": float(np.nanmean(act[:, cmask][ok]) / scale)}
        for pname, p in preds.items():
            if p is None:
                pp = np.broadcast_to(mi[cmask], yy.shape)
            elif p.ndim == 2:
                pp = np.broadcast_to(p[cmask], yy.shape)
            else:
                pp = p[:, cmask]
            good = ok & np.isfinite(pp)
            d = (yy - pp)[good]
            row[f"mse_{pname}"] = float(np.mean(d ** 2))
            row[f"bias_{pname}"] = float(np.mean(d))
        # SiLo on a pixel subsample (slow: a moving median)
        idx = np.flatnonzero(cmask.ravel())
        sub = rng.choice(idx, size=min(silo_pixels, idx.size), replace=False)
        ms = m.reshape(m.shape[0], -1)[:, sub]
        ys = y.reshape(y.shape[0], -1)[:, sub]
        frames, sb = _silo(ms)
        d = ys[frames] - sb
        good = np.isfinite(d)
        row["mse_silo"] = float(np.mean(d[good] ** 2))
        row["bias_silo"] = float(np.mean(d[good]))
        # mean image and target_mean on the same subsample and frames, to compare with silo
        for pname in ("mean_image", "target_mean"):
            p = mi if pname == "mean_image" else preds["target_mean"]
            ds = ys[frames] - p.ravel()[sub][None]
            row[f"mse_{pname}_silo_subset"] = float(np.mean(ds[good] ** 2))
        rows.append(row)
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="slap2-f0 ceiling", description=__doc__.split("\n\n")[0])
    add_arguments(p)
    run(p.parse_args(argv))


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("root", help="folder with exported trial files (searched recursively)")
    p.add_argument("--out", default="ceiling.csv")
    p.add_argument("--gap", type=int, default=8)
    p.add_argument("--reach", type=int, default=254, help="temporal reach of the context beyond the gap, frames")
    p.add_argument("--max_files", type=int, default=0, help="per session; 0 = all")
    p.add_argument("--smooth", type=int, default=1, help="average target over this many frames first (SiLo: 8)")


def run(args) -> None:
    files = list_trials(args.root)
    if args.max_files:
        per: dict[str, int] = {}
        keep = []
        for f in files:
            s = f.parent.parent.name
            if per.get(s, 0) < args.max_files:
                keep.append(f)
                per[s] = per.get(s, 0) + 1
        files = keep
    rows = []
    for fn in files:
        r = trial_ceiling(fn, gap=args.gap, reach=args.reach, smooth=args.smooth)
        rows += r
        for x in r:
            if x["class"] == "loss":
                print(f"{fn.parent.parent.name}/{fn.parent.name}/{fn.name}: " + json.dumps(
                    {k: round(v, 3) for k, v in x.items() if k.startswith("mse_") or k.startswith("bias_")}), flush=True)
    if rows:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out} ({len(rows)} rows)")
