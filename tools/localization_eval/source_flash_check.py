"""What the sources' flashes look like, from source_patches.py's output: are
they synapse-sized disks with sensor kinetics, or edge, line or dot artifacts?

usage: python source_flash_check.py OUT_DIR [--path Path2] [--summary A/experiment_summary.h5] [--events 30]

Per source (category 0 both runs, 1 only run A, 2 only run B, 3 control pixel):
  - events: the largest 3-frame means of the centre 3x3 trace above its trial
    median, at least 20 frames apart;
  - flash image: the mean over events of the 3-frame mean minus frames
    t-25..t-10, normalised to its peak; its size from a 2-D Gaussian fit
    (sigma along the major and minor axes, orientation; NaN where the fit fails);
  - kinetics: the event-triggered centre trace, its width at half maximum and
    its decay to half from the peak, in 80 Hz frames;
  - zeros: share of frames where the centre pixel is exactly 0;
  - synchrony: share of events within +-1 frame of an event of a source more
    than 10 px away;
  - motion (with --summary): the frame's registration shift at the events, as
    a percentile of all frames' shifts (50 = unrelated to motion);
  - ring_z and tail_z, the selection-free test: picking any pixel's largest
    frames yields a "flash" there, so the signal is measured only outside the
    selection window, in a ring at Chebyshev distance 3-4 px (same frames) and
    in the decay tail t+3..t+8 (centre 3x3), each as a z-score against the
    same statistic at 100 sets of random times. A synapse-sized disk with
    sensor kinetics has both; noise picked at one pixel has neither. A source
    passes with both above 3.
Writes OUT_DIR/<path>_flash.npz (images, traces, measures) and prints medians per category.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import curve_fit


def gauss2d(xy, a, r0, c0, s1, s2, th, b):
    r, c = xy
    ct, st = np.cos(th), np.sin(th)
    u = (r - r0) * ct + (c - c0) * st
    v = -(r - r0) * st + (c - c0) * ct
    return a * np.exp(-0.5 * (u / s1) ** 2 - 0.5 * (v / s2) ** 2) + b


def fit_size(img):
    """Major and minor sigma (px), orientation (deg, 0 = along rows) and centre offset of a 2-D Gaussian fit."""
    n = img.shape[0]
    rr, cc = np.mgrid[:n, :n].astype(float)
    ok = np.isfinite(img)
    if ok.sum() < 20:
        return [np.nan] * 4
    c = n // 2
    p0 = [np.nanmax(img), c, c, 1.5, 1.5, 0.0, 0.0]
    try:
        p, _ = curve_fit(gauss2d, (rr[ok], cc[ok]), img[ok], p0=p0, maxfev=4000,
                         bounds=([0, c - 4, c - 4, 0.3, 0.3, -np.pi, -1], [10, c + 4, c + 4, 8, 8, np.pi, 1]))
    except (RuntimeError, ValueError):
        return [np.nan] * 4
    s1, s2, th = abs(p[3]), abs(p[4]), p[5]
    if s2 > s1:
        s1, s2, th = s2, s1, th + np.pi / 2
    return [s1, s2, np.degrees(th) % 180, float(np.hypot(p[1] - c, p[2] - c))]


def selection_free_z(p, ev, T, rng, n_null=100):
    """Ring (Chebyshev distance 3-4 px, frames t-1..t+1) and tail (centre 3x3,
    t+3..t+8) signal at the events, as z-scores against random times."""
    w = p.shape[1]
    c = w // 2
    yy, xx = np.mgrid[:w, :w] - c
    cheb = np.maximum(abs(yy), abs(xx))
    ring = (cheb >= 3) & (cheb <= 4)

    def stat(ts):
        pre = np.stack([np.nanmean(p[t - 25:t - 10], 0) for t in ts])
        win = np.stack([np.nanmean(p[t - 1:t + 2], 0) for t in ts]) - pre
        tail = np.array([np.nanmean(p[t + 3:t + 9, c - 1:c + 2, c - 1:c + 2]) for t in ts]) - \
            np.nanmean(pre[:, c - 1:c + 2, c - 1:c + 2], (1, 2))
        return np.nanmean(win[:, ring]), np.nanmean(tail)

    obs = np.array(stat(ev))
    null = np.array([stat(rng.integers(25, T - 41, len(ev))) for _ in range(n_null)])
    return tuple((obs - np.nanmean(null, 0)) / np.nanstd(null, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--path", default="Path2")
    ap.add_argument("--summary", default="")
    ap.add_argument("--events", type=int, default=30)
    args = ap.parse_args()
    out = Path(args.out)
    P = np.load(out / f"{args.path}_patches.npy", mmap_mode="r")
    S = np.load(out / f"{args.path}_sources.npz")
    cat, bounds, xy = S["cat"], S["bounds"], S["xy"]
    n, T, w, _ = P.shape
    c = w // 2
    trial = np.searchsorted(bounds, np.arange(T), side="right") - 1
    motion = None
    if args.summary:  # registration shift per 80 Hz frame, from the summary's native-rate frame info
        with h5py.File(args.summary, "r") as f:
            g = f[args.path]
            sx = np.asarray(g["frame_info/offlineXshifts"][()]).ravel()
            sy = np.asarray(g["frame_info/offlineYshifts"][()]).ravel()
            nf = np.asarray(g["frame_info/trial_num_frames"][()]).ravel()
        nb = np.concatenate([[0], np.cumsum(nf)])
        motion = np.empty(T)
        for k in range(len(bounds) - 1):
            t80 = np.arange(bounds[k + 1] - bounds[k])
            idx = nb[k] + np.minimum(((t80 + 0.5) * nf[k] / len(t80)).astype(int), nf[k] - 1)
            mx, my = sx[idx] - np.nanmedian(sx[nb[k]:nb[k + 1]]), sy[idx] - np.nanmedian(sy[nb[k]:nb[k + 1]])
            motion[bounds[k]:bounds[k + 1]] = np.hypot(mx, my)
    ev_all, flash, trace, meas = [], [], [], []
    lags = np.arange(-20, 41)
    rng = np.random.default_rng(0)
    for k in range(n):
        p = np.asarray(P[k], np.float32)
        ctr = np.nanmean(p[:, c - 1:c + 2, c - 1:c + 2], (1, 2))
        dff = ctr.copy()
        for t in range(len(bounds) - 1):
            seg = slice(bounds[t], bounds[t + 1])
            dff[seg] -= np.nanmedian(ctr[seg])
        sm = np.convolve(np.nan_to_num(dff, nan=-1e9), np.ones(3) / 3, "same")
        order = np.argsort(-sm)
        ev = []
        for t in order:
            if len(ev) >= args.events or sm[t] < -1e6:
                break
            if 25 <= t < T - 41 and all(abs(t - e) >= 20 for e in ev) and trial[t - 25] == trial[t + 40]:
                ev.append(t)
        ev = np.array(sorted(ev))
        ev_all.append(ev)
        imgs = [np.nanmean(p[t - 1:t + 2], 0) - np.nanmean(p[t - 25:t - 10], 0) for t in ev]
        im = np.nanmean(imgs, 0) if imgs else np.full((w, w), np.nan)
        pk = np.nanmax(im[c - 2:c + 3, c - 2:c + 3]) if np.isfinite(im).any() else np.nan
        im = im / pk if pk and pk > 0 else im
        tr = np.nanmean([ctr[t + lags] - np.nanmean(ctr[t - 25:t - 10]) for t in ev], 0) if len(ev) else np.full(len(lags), np.nan)
        tpk = np.nanmax(tr[18:23]) if np.isfinite(tr).any() else np.nan
        tr = tr / tpk if tpk and tpk > 0 else tr
        above = tr >= 0.5
        i0 = 20
        lo = i0
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = i0
        while hi < len(tr) - 1 and above[hi + 1]:
            hi += 1
        s1, s2, th, off = fit_size(im)
        zeros = float(np.nanmean(p[:, c, c] == 0))
        mot = float(np.mean([np.mean(motion < motion[t]) * 100 for t in ev])) if motion is not None and len(ev) else np.nan
        rz, tz = selection_free_z(p, ev, T, rng) if len(ev) else (np.nan, np.nan)
        flash.append(im); trace.append(tr)
        meas.append([s1, s2, th, off, hi - lo + 1, hi - i0, zeros, mot, float(np.nanmean([sm[t] for t in ev])) if len(ev) else np.nan,
                     rz, tz])
    meas = np.array(meas)
    sync = np.full(n, np.nan)
    for k in range(n):
        far = np.hypot(*(xy - xy[k]).T) > 10
        others = np.concatenate([ev_all[j] for j in np.nonzero(far)[0]]) if far.any() else np.array([])
        if len(ev_all[k]) and len(others):
            sync[k] = np.mean([np.any(np.abs(others - t) <= 1) for t in ev_all[k]])
    cols = ["sigma_major", "sigma_minor", "orientation", "centre_offset", "fwhm_frames", "decay_frames", "zeros",
            "motion_pct", "event_size", "ring_z", "tail_z"]
    np.savez(out / f"{args.path}_flash.npz", flash=np.array(flash), trace=np.array(trace), lags=lags, meas=meas,
             cols=np.array(cols), sync=sync, cat=cat, events=np.array(ev_all, dtype=object))
    names = ["both runs", f"only {S['names'][0]}", f"only {S['names'][1]}", "control px"]
    print(f"{'':16s} " + " ".join(f"{h:>13s}" for h in cols + ["sync"]))
    for k, nm in enumerate(names):
        m = cat == k
        row = list(np.nanmedian(meas[m], 0)) + [np.nanmedian(sync[m])]
        print(f"{nm + f' ({m.sum()})':16s} " + " ".join(f"{v:13.2f}" for v in row))
    for k, nm in enumerate(names):
        m = cat == k
        if m.any():
            ok = (meas[m, 9] > 3) & (meas[m, 10] > 3)
            print(f"{nm}: pass the selection-free test (ring_z > 3 and tail_z > 3) {ok.sum()} of {m.sum()} ({ok.mean():.2f})")
    for k, nm in enumerate(names):  # orientation spread: lines along the scan would share one
        m = (cat == k) & np.isfinite(meas[:, 0]) & (meas[:, 0] > 1.3 * meas[:, 1])
        if m.any():
            th = np.radians(meas[m, 2]) * 2
            print(f"{nm}: {m.sum()} elongated (major > 1.3 x minor), mean orientation "
                  f"{np.degrees(np.angle(np.mean(np.exp(1j * th)))) / 2 % 180:.0f} deg, alignment {abs(np.mean(np.exp(1j * th))):.2f}")


if __name__ == "__main__":
    main()
