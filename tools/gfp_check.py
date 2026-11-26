"""Ground-truth check of network baselines on a GFP control with injected events, independent of GIAnT's fit
(docs/plan.md, "S6", 2026-11-26).

Two registered movies of the same trial: the original GFP recording (no activity), and a copy with injected
events of known size (GIAnT-Python's injection tool, GIAnT-Python#22; for 847983: 60 sources per path, 1 Hz,
sizes 0.25-8). Use a trial where the copy's registration equals the original's (trial 1 of 847983: within
0.02 px), so copy - original is the injected activity at every pixel.

  no-activity bias   on the original: sum(movie - baseline) / sum(movie); positive = baseline below the movie
  leak               at the injected sources (peak pixel and neighbours within 1.5 px): the share of the injected
                     mean activity that appears in the baseline, sum(B_copy - B_orig) / sum(M_copy - M_orig)
  following          within-source slope of (B_copy - B_orig) on (M_copy - M_orig), 8-frame means: the share of
                     an event's time course that the baseline follows
  bias at sources    on the copy against the truth (the original movie): sum(M_orig - B_copy) / sum(M_orig)
  rings              the same quantities in rings around the source peaks (beyond 5 px nothing is injected)

The first and last 2 s of the trial and frames discarded for motion in either movie are left out.

usage: python tools/gfp_check.py NAME=BASELINE_ROOT [...] --inj <injection dir> --cache <cache dir>
                                 [--trial E1T1] [--copy 847983_inj] [--orig 847983_orig]

<injection dir> holds the copy and the original (folders --copy and --orig), each with
motion_correction/<trial>DMD{1,2}_ALIGNMENTDATA.h5 and <trial>DMD{1,2}_REGISTERED_DOWNSAMPLED-80Hz.h5; the
copy also holds injection_truth.h5. Each BASELINE_ROOT holds `slap2-f0 predict-registered` outputs for both
movies: BASELINE_ROOT/<copy>/<trial>DMD{1,2}_BASELINE-80Hz.h5, and the same under <orig>. The movies at the
pixels valid in every model's baselines are cached in <cache dir> (tools/gfp_mse.py reads that cache).
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

from slap2_f0.registered import discard_frames

EDGE_S, SMOOTH = 2.0, 8


def smooth(x, n=SMOOTH):
    ok = np.isfinite(x)
    k = np.ones(n)
    num = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, np.where(ok, x, 0.0))
    den = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, ok.astype(float))
    return np.where(den >= n / 2, num / np.maximum(den, 1), np.nan)


def movie_at(inj, trial, ds, dmd, idx):
    """Channel-1 frames of the registered movie at flat page indices idx: (frames, len(idx)); and the discards."""
    stem = f"{trial}DMD{dmd}"
    with h5py.File(inj / ds / "motion_correction" / f"{stem}_ALIGNMENTDATA.h5", "r") as f:
        n_ch = int(np.asarray(f["numChannels"]).ravel()[0])
        hz = float(np.asarray(f["alignHz"]).ravel()[0])
        disc = discard_frames(np.asarray(f["recNegErr"]).ravel(), hz)
    with h5py.File(inj / ds / "motion_correction" / f"{stem}_REGISTERED_DOWNSAMPLED-80Hz.h5", "r") as f:
        d = f["data"]
        n = d.shape[0] // n_ch
        out = np.empty((n, len(idx)), np.float32)
        for i0 in range(0, n, 128):
            i1 = min(n, i0 + 128)
            out[i0:i1] = d[i0 * n_ch:i1 * n_ch][::n_ch].reshape(i1 - i0, -1)[:, idx]
    return out, disc, hz


def baseline_at(root, trial, ds, dmd, idx=None):
    with h5py.File(Path(root) / ds / f"{trial}DMD{dmd}_BASELINE-80Hz.h5", "r") as f:
        b = f["baseline"]
        if idx is None:  # valid pixels: finite in a middle frame
            return np.isfinite(b[b.shape[0] // 2]), b.shape
        out = np.empty((b.shape[0], len(idx)), np.float32)
        for i0 in range(0, b.shape[0], 256):
            out[i0:i0 + 256] = b[i0:i0 + 256].reshape(-1, b.shape[1] * b.shape[2])[:, idx]
    return out


def load_dmd(a, dmd, roots):
    """Movies at the pixels valid in every model's baselines (cached), kept frames, source cores."""
    valid = None
    for root in roots:
        for ds in (a.copy, a.orig):
            v, shape = baseline_at(root, a.trial, ds, dmd)
            valid = v if valid is None else (valid & v)
    idx = np.flatnonzero(valid.ravel())
    cf = a.cache / f"{a.trial}DMD{dmd}_{len(idx)}.npz"
    if cf.exists():
        z = np.load(cf)
        if np.array_equal(z["idx"], idx):
            return {k: z[k] for k in z.files} | {"shape": shape}
    mi, di, hz = movie_at(a.inj, a.trial, a.copy, dmd, idx)
    mo, do, _ = movie_at(a.inj, a.trial, a.orig, dmd, idx)
    keep = ~(di | do)
    e = int(round(EDGE_S * hz))
    keep[:e] = False
    keep[-e:] = False
    a.cache.mkdir(parents=True, exist_ok=True)
    np.savez(cf, idx=idx, m_inj=mi, m_orig=mo, keep=keep, hz=hz)
    return {"idx": idx, "m_inj": mi, "m_orig": mo, "keep": keep, "hz": hz, "shape": shape}


def peaks(a, dmd):
    """Zero-based MATLAB (row, col) of each injected source's peak pixel on this path."""
    with h5py.File(a.inj / a.copy / "injection_truth.h5", "r") as f:
        return f[f"path{dmd}/peak_pixel_rc"][:]


def cores(a, dmd, d):
    """Indices (into the valid-pixel list) of each source's core: valid pixels within 1.5 px of its peak pixel.
    Truth positions are zero-based MATLAB (row, col); a stored page is (MATLAB col, row)."""
    pk = peaks(a, dmd)
    B = d["shape"][2]
    pa, pb = np.divmod(d["idx"], B)  # page coordinates of the valid pixels: pa = MATLAB col, pb = MATLAB row
    out = []
    for r, c in pk:
        m = np.flatnonzero(np.hypot(pa - c, pb - r) <= 1.5)
        if len(m) >= 5:
            out.append(m)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("models", nargs="+", metavar="NAME=BASELINE_ROOT")
    p.add_argument("--inj", type=Path, required=True, help="folder holding the copy and the original")
    p.add_argument("--cache", type=Path, required=True, help="cache of the movies at the valid pixels")
    p.add_argument("--trial", default="E1T1", help="trial stem of the files (default E1T1)")
    p.add_argument("--copy", default="847983_inj", help="folder of the copy with injected events")
    p.add_argument("--orig", default="847983_orig", help="folder of the original")
    a = p.parse_args()
    models = [m.split("=", 1) for m in a.models]
    roots = [r for _, r in models]
    res = {name: [] for name, _ in models}
    for dmd in (1, 2):
        d = load_dmd(a, dmd, roots)
        k = d["keep"]
        mi, mo = d["m_inj"], d["m_orig"]
        cs = cores(a, dmd, d)
        mean_o = np.nanmean(mo[k], 0)
        inj = np.nanmean(mi[k], 0) - mean_o  # injected mean activity per pixel
        bright = np.argsort(np.argsort(mean_o)) / len(mean_o)
        core_px = np.unique(np.concatenate(cs))
        src_act = np.array([inj[c].mean() for c in cs])
        src_rel = src_act / np.array([mean_o[c].mean() for c in cs])
        print(f"DMD{dmd}: {len(d['idx'])} valid px, {int(k.sum())} frames kept, {len(cs)} sources; injected mean activity "
              f"at the cores: median {100 * np.median(src_rel):.1f}% of the baseline (10-90%: {100 * np.quantile(src_rel, 0.1):.1f}-{100 * np.quantile(src_rel, 0.9):.1f}%)")
        # copy - original away from sources should be ~0 (registration check)
        print(f"   check, copy minus original: at cores {100 * inj[core_px].sum() / mean_o[core_px].sum():+.2f}% of the baseline; "
              f"at the dimmest half of pixels {100 * inj[bright < 0.5].sum() / mean_o[bright < 0.5].sum():+.2f}%")
        pk = peaks(a, dmd)
        pa, pb = np.divmod(d["idx"], d["shape"][2])
        dist = np.min(np.hypot(pa[:, None] - pk[None, :, 1], pb[:, None] - pk[None, :, 0]), 1)
        for name, root in models:
            bo = baseline_at(root, a.trial, a.orig, dmd, d["idx"])
            bi = baseline_at(root, a.trial, a.copy, dmd, d["idx"])
            r = {"dmd": dmd}
            bo_m, bi_m = np.nanmean(bo[k], 0), np.nanmean(bi[k], 0)
            ok = np.isfinite(bo_m) & np.isfinite(bi_m)
            for lab, m in (("all", ok), ("dim third", ok & (bright < 1 / 3)), ("mid third", ok & (bright >= 1 / 3) & (bright < 2 / 3)),
                           ("bright third", ok & (bright >= 2 / 3))):
                r["bias " + lab] = 100 * (mean_o[m] - bo_m[m]).sum() / mean_o[m].sum()
            # control: the same quantities in rings around the source peaks (beyond 5 px nothing is injected)
            for lo, hi in ((0, 1.5), (1.5, 3), (3, 5), (5, 8), (8, 1e9)):
                s_ = ok & (dist >= lo) & (dist < hi)
                n_ = mean_o[s_].sum()
                r[f"ring {lo:g}-{min(hi, 99):g} px: injected / B_orig - truth / B_copy - truth"] = (
                    100 * inj[s_].sum() / n_, 100 * (bo_m - mean_o)[s_].sum() / n_, 100 * (bi_m - mean_o)[s_].sum() / n_)
            cp = core_px[ok[core_px]]
            r["bias cores, original"] = 100 * (mean_o[cp] - bo_m[cp]).sum() / mean_o[cp].sum()
            r["bias cores, copy vs truth"] = 100 * (mean_o[cp] - bi_m[cp]).sum() / mean_o[cp].sum()
            r["leak"] = (bi_m[cp] - bo_m[cp]).sum() / inj[cp].sum()
            order = np.argsort(src_rel)
            t3 = np.array_split(order, 3)
            for lab, o in zip(("leak, least active third", "leak, middle third", "leak, most active third"), t3):
                r[lab] = sum((bi_m[cs[j]] - bo_m[cs[j]]).sum() for j in o) / sum(inj[cs[j]].sum() for j in o)
            # following: within-source temporal slope, 8-frame means
            num = den = 0.0
            for c in cs:
                x = np.where(k, (mi[:, c] - mo[:, c]).mean(1), np.nan)
                y = np.where(k, np.nanmean(bi[:, c] - bo[:, c], 1), np.nan)
                xs, ys = smooth(x[:, None])[:, 0], smooth(y[:, None])[:, 0]
                g = np.isfinite(xs) & np.isfinite(ys)
                xs, ys = xs[g] - xs[g].mean(), ys[g] - ys[g].mean()
                num += (xs * ys).sum()
                den += (xs * xs).sum()
            r["following"] = num / den
            res[name].append(r)

    keys = [key for key in res[models[0][0]][0] if key != "dmd"]
    print("\nmean of the two paths (DMD1, DMD2 in brackets); bias in % of the movie, positive = baseline below the truth")
    print(f"{'':30s}" + "".join(f"{name:>24s}" for name, _ in models))
    for key in keys:
        pct = key.startswith("bias")
        if key.startswith("ring"):
            print(key + ", % of the movie, mean of the two paths: " + "   ".join(
                f"{name} " + " / ".join(f"{x:+.2f}" for x in np.mean([r[key] for r in res[name]], 0)) for name, _ in models))
            continue
        cells = []
        for name, _ in models:
            v = [r[key] for r in res[name]]
            cells.append((f"{np.mean(v):+.2f}% ({v[0]:+.1f}, {v[1]:+.1f})" if pct else f"{np.mean(v):.3f} ({v[0]:.2f}, {v[1]:.2f})").rjust(24))
        print(f"{key:30s}" + "".join(cells))


if __name__ == "__main__":
    main()
