"""Selection-safe comparison of GIAnT's activity with the movie on an exported trial.

Events are picked on the MOVIE in a ring 3-5 px from each source's peak
pixel: pixels that carry almost none of the source's footprint, so neither
the core's photon noise nor the activity estimate's noise selects them (the
selection favours spatially broad events). At each event the amplitude is
the mean of frames -1..+1 minus the mean of frames -8..-3, for the movie and
for the activity, and the two are compared at <= 1.5 px and 1.5-2.5 px from
the peak.

This is the reference for ``code/matlab/ringEventRatios.m``, which the
exporter runs on every trial (attributes ``eventRatioCore``,
``eventRatio2px``, ``eventRatioSources``, ``eventRatioEvents``,
``eventRatioArtifactSources``). Both follow the analysis of the activity-fit
plan (``ring_eta.py``, 2026-11-24) line for line.

    python -m slap2_f0.ringqc <trial files or folders>

prints the ratios of exported trials, with the activity divided by the
trial's ``activityGain`` (older exports scaled it), so old and new exports
can be compared.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import median_filter

from .io import list_trials

RAD = 8  # box half-width around each peak pixel
EDGE = 160  # frames at each end where no event is picked
BINS = ((0.0, 1.5), (1.5, 2.5))


@dataclass
class RingEventRatios:
    core: float  # summed activity amplitude / summed movie amplitude, <= 1.5 px
    near: float  # the same at 1.5-2.5 px
    n_sources: int  # sources with at least 3 ring events
    n_events: int
    n_artifact_sources: int  # left out: a pixel within 4 px is a GIAnT fit artifact


def ring_event_ratios(movie: np.ndarray, activity: np.ndarray, valid: np.ndarray, peak_row, peak_col,
                      discard: np.ndarray, activity_scale: float = 1.0) -> RingEventRatios:
    """The ratios for one trial.

    movie, activity: (frame, row, col); the movie is NaN where a pixel is not
    imaged (discarded frames count as NaN). Neither is copied whole: only each
    source's box is taken out, so a large trial needs no memory beyond its two
    arrays. activity_scale multiplies the activity (1 / activityGain for an
    old export). valid: (row, col) bool.
    peak_row, peak_col: the sources' 0-based peak pixels. discard: (frame,)
    bool. A source counts when its 17 x 17 px box lies inside the arrays, its
    peak pixel is valid, no pixel within 4 px is a fit artifact (mean activity
    > 3 x max(mean movie, 1)), it has at least 12 valid ring and 5 valid core
    pixels, and at least 3 ring events: local maxima, at frames 160..T-161, of
    the ring's mean (frames with a NaN ring pixel left out) minus its 81-frame
    running median, above that signal's 99th percentile, with no discarded
    frame within -8..+16.
    """
    T, R, C = movie.shape
    discard = np.asarray(discard, bool)
    keep = np.flatnonzero(~discard)
    ma = np.zeros((R, C)); ms = np.zeros((R, C)); mn = np.zeros((R, C))
    for t0 in range(0, T, 256):  # time-means in blocks of frames (float64 sums, no full-size temporary)
        ma += activity[t0:t0 + 256].sum(0, dtype=np.float64)
        k = keep[(keep >= t0) & (keep < t0 + 256)]
        if len(k):
            blk = movie[k]
            ok = np.isfinite(blk)
            ms += np.where(ok, blk, 0).sum(0, dtype=np.float64); mn += ok.sum(0)
    ma *= activity_scale / T
    with np.errstate(invalid="ignore", divide="ignore"):
        mm = np.where(mn > 0, ms / np.maximum(mn, 1), np.nan)
        bad_px = ma > 3 * np.maximum(mm, 1)  # NaN (never imaged) compares False
    rr, cc = np.mgrid[-RAD:RAD + 1, -RAD:RAD + 1]
    dist = np.hypot(rr, cc)
    num_m = np.zeros(len(BINS)); num_a = np.zeros(len(BINS))
    out = RingEventRatios(np.nan, np.nan, 0, 0, 0)
    for r0, c0 in zip(np.asarray(peak_row, int), np.asarray(peak_col, int)):
        if not (RAD <= r0 < R - RAD and RAD <= c0 < C - RAD) or not valid[r0, c0]:
            continue
        box = (slice(r0 - RAD, r0 + RAD + 1), slice(c0 - RAD, c0 + RAD + 1))
        v = valid[box]
        if bad_px[box][dist <= 4].any():
            out.n_artifact_sources += 1
            continue
        ring = v & (dist >= 3) & (dist <= 5)
        core = v & (dist <= 1.5)
        if ring.sum() < 12 or core.sum() < 5:
            continue
        mr = np.asarray(movie[:, box[0], box[1]][:, ring], np.float64)
        mr[discard] = np.nan
        sel = np.where(np.isnan(mr).any(1), np.nan, mr.mean(1))
        if np.isnan(sel).all():  # every frame misses a ring pixel: no events
            continue
        sel = sel - median_filter(np.nan_to_num(sel, nan=np.nanmedian(sel)), size=81, mode="nearest")
        thr = np.nanpercentile(sel, 99)
        ev = [e for e in range(EDGE, T - EDGE) if sel[e] > thr and sel[e] >= sel[e - 1] and sel[e] > sel[e + 1]
              and not discard[e - 8:e + 17].any()]
        if len(ev) < 3:
            continue
        for e in ev:
            # no discarded frame in the window (see ev), so the movie needs no masking here
            mw = np.asarray(movie[e - 8:e + 17, box[0], box[1]], np.float64)
            aw = np.asarray(activity[e - 8:e + 17, box[0], box[1]], np.float64) * activity_scale
            am = mw[7:10].mean(0) - mw[:6].mean(0); aa = aw[7:10].mean(0) - aw[:6].mean(0)
            okp = v & np.isfinite(am)
            for i, (lo, hi) in enumerate(BINS):
                q = okp & (dist >= lo) & (dist < hi)
                num_m[i] += am[q].sum(); num_a[i] += aa[q].sum()
        out.n_sources += 1
        out.n_events += len(ev)
    if out.n_events:
        out.core, out.near = float(num_a[0] / num_m[0]), float(num_a[1] / num_m[1])
    return out


def trial_ring_event_ratios(path: str | Path, channel: int = 0) -> RingEventRatios:
    """The ratios of an exported trial, with its activity divided by its
    ``activityGain`` (GIAnT's activity, whatever the export applied)."""
    with h5py.File(path, "r") as f:
        valid = f["mask/valid"][:].astype(bool)
        discard = f["frame/discard"][:].astype(bool)
        if "sources" not in f:
            return RingEventRatios(np.nan, np.nan, 0, 0, 0)
        pr, pc = f["sources/peak_row"][:].astype(int), f["sources/peak_col"][:].astype(int)
        gain = float(np.ravel(f.attrs.get("activityGain", 1.0))[0])
        movie = f["movie"][channel]
        activity = f["activity"][channel]
    return ring_event_ratios(movie, activity, valid, pr, pc, discard, activity_scale=1.0 / gain)


def main(argv=None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    files = []
    for a in args:
        p = Path(a)
        files += list_trials(p) if p.is_dir() else [p]  # folders: exported trials only (not trial_table.h5)
    for fn in files:
        q = trial_ring_event_ratios(fn)
        print(f"{fn}: activity/movie {q.core:.4f} at <= 1.5 px, {q.near:.4f} at 1.5-2.5 px "
              f"({q.n_sources} sources, {q.n_events} ring events, {q.n_artifact_sources} at fit artifacts left out)")


if __name__ == "__main__":
    main()
