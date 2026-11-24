"""Read trial files written by ``code/matlab/exportF0TrainingData.m``.

Each file holds one trial of one imaging path on the trial's 80 Hz registered
grid, cropped to the box around the valid pixels. Arrays (C order):

    movie       (channel, frame, row, col)  registered movie, NaN where not
                                            imaged or discarded
    activity    (channel, frame, row, col)  GIAnT's fitted activity, same grid
                                            and timing; target = movie - activity
    mean_image  (channel, row, col)
    varfac      (row, col) or (frame, row, col); TrialFile.varfac is relative to its median over valid pixels
    mask/valid, mask/labeled, mask/footprint_support   (row, col) uint8
    frame/discard, frame/motion_row, ...               (frame,)
    sources/peak_row, sources/peak_col                 (source,)

Attributes on the root group hold QC and provenance (see the export function).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class Patch:
    """A window of a trial; positions outside the trial are NaN (movie) or 0."""

    movie: np.ndarray  # (frame, row, col) float32, NaN = no data
    activity: np.ndarray  # (frame, row, col) float32
    mean_image: np.ndarray  # (row, col) float32, NaN outside the trial
    varfac: np.ndarray  # (row, col) float32
    labeled: np.ndarray  # (row, col) bool
    support: np.ndarray  # (row, col) bool
    origin: tuple[int, int, int]  # (frame, row, col) of the window's corner


def read_mean_image(f: h5py.File, channel: int = 0) -> np.ndarray:
    """A trial's mean image (row, col) for one channel. Exports of
    one-channel recordings store it as (row, col), without the channel axis
    (MATLAB drops the trailing singleton when writing), so indexing it by
    channel would return its first row."""
    ds = f["mean_image"]
    if ds.ndim == 2:
        if channel != 0:
            raise IndexError(f"{f.filename}: one-channel mean image, no channel {channel}")
        return ds[:]
    return ds[channel]


class TrialFile:
    """Lazy access to one exported trial."""

    def __init__(self, path: str | Path, channel: int = 0):
        self.path = Path(path)
        self.channel = channel
        self._f = h5py.File(self.path, "r")
        self.n_channels, self.n_frames, self.n_rows, self.n_cols = self._f["movie"].shape
        self.attrs = {k: _scalar(v) for k, v in self._f.attrs.items()}
        self.valid = self._f["mask/valid"][:].astype(bool)
        self.labeled = self._f["mask/labeled"][:].astype(bool)
        self.support = self._f["mask/footprint_support"][:].astype(bool)
        self.discard = self._f["frame/discard"][:].astype(bool)
        self.mean_image = read_mean_image(self._f, channel).astype(np.float32)
        vf = self._f["varfac"]
        vf = (vf[:] if vf.ndim == 2 else np.nanmean(vf[:], axis=0)).astype(np.float32)
        # relative to its median over the imaged pixels, so the network input
        # does not depend on the window (idempotent: a simulated trial saved
        # from this keeps it)
        ok = self.valid & np.isfinite(vf) & (vf > 0)
        self.varfac_ref = float(np.median(vf[ok])) if ok.any() else 1.0
        self.varfac = vf / np.float32(self.varfac_ref)
        good = self.valid & np.isfinite(self.mean_image)
        # one scale per trial, so inputs and outputs are O(1)
        self.scale = float(np.nanmedian(self.mean_image[good])) if good.any() else 1.0

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "TrialFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def read(self, t0: int, nt: int, r0: int, nr: int, c0: int, nc: int) -> Patch:
        """Window starting at (t0, r0, c0) of size (nt, nr, nc); may extend past
        the trial on any side, where it is filled with NaN (movie) or 0."""
        movie = np.full((nt, nr, nc), np.nan, np.float32)
        activity = np.zeros((nt, nr, nc), np.float32)
        ts, te = max(t0, 0), min(t0 + nt, self.n_frames)
        rs, re = max(r0, 0), min(r0 + nr, self.n_rows)
        cs, ce = max(c0, 0), min(c0 + nc, self.n_cols)
        mean = np.full((nr, nc), np.nan, np.float32)
        vf = np.zeros((nr, nc), np.float32)
        lab = np.zeros((nr, nc), bool)
        sup = np.zeros((nr, nc), bool)
        if ts < te and rs < re and cs < ce:
            dt, dr, dc = slice(ts - t0, te - t0), slice(rs - r0, re - r0), slice(cs - c0, ce - c0)
            ch = self.channel
            movie[dt, dr, dc] = self._f["movie"][ch, ts:te, rs:re, cs:ce]
            activity[dt, dr, dc] = self._f["activity"][ch, ts:te, rs:re, cs:ce]
            valid = self.valid[rs:re, cs:ce]
            movie[dt, dr, dc][:, ~valid] = np.nan
            movie[dt, dr, dc][self.discard[ts:te]] = np.nan
            mean[dr, dc] = np.where(valid, self.mean_image[rs:re, cs:ce], np.nan)
            vf[dr, dc] = np.nan_to_num(self.varfac[rs:re, cs:ce])
            lab[dr, dc] = self.labeled[rs:re, cs:ce] & valid
            sup[dr, dc] = self.support[rs:re, cs:ce] & valid
        return Patch(movie, activity, mean, vf, lab, sup, (t0, r0, c0))


def _scalar(v):
    a = np.asarray(v)
    if a.dtype.kind in "SO":
        return a.astype(str).item() if a.size == 1 else a.astype(str).tolist()
    return a.item() if a.size == 1 else a.tolist()


def list_trials(root: str | Path) -> list[Path]:
    """All exported trial files (trial_<digits>.h5 with a movie) under a folder,
    at any depth. Other HDF5 files that sit next to them, such as GIAnT's
    trial_table.h5, are skipped."""
    out = []
    for p in sorted(Path(root).rglob("trial_*.h5")):
        if not p.stem[len("trial_"):].isdigit():
            continue
        try:
            with h5py.File(p, "r") as f:
                if "movie" in f and "activity" in f:
                    out.append(p)
        except OSError:
            continue
    return out


def _path_id(attr, folder: str) -> int | str:
    """An imaging path's number from the exporter's ``path`` attribute or,
    without it, from the folder name (Path2 -> 2), so both name one path."""
    if attr is not None:
        try:
            return int(float(attr))
        except (TypeError, ValueError):
            pass
    m = re.fullmatch(r"(?i)path_?(\d+)", folder)
    return int(m.group(1)) if m else folder


def filter_trials(files: list[Path], min_peak_r: float = 0.3, min_path_trials: int = 4) -> tuple[list[Path], list[str]]:
    """Leave out trials whose activity calibration failed, and imaging paths
    left with too few trials. Returns the kept files, in the order given,
    and one report line per trial or path left out.

    - Calibration: the exporter aligns GIAnT's activity with the movie
      (offset and blur; older exports also scaled it) and
      records how well its peaks match (attribute ``peakR``, a correlation,
      so the scale does not enter). Below
      ``min_peak_r`` the fit does not describe the movie: in the 2025-2026
      exports 34 of 3,026 trials scored 0.03-0.27, with the gain and blur
      pinned at their bounds, against 0.30 or more for the rest (median
      0.43). Their training target (movie - activity) is wrong wherever the
      activity is.
    - Paths: a path with fewer than ``min_path_trials`` trials left after
      the calibration cut is left out whole. This applies only to paths of
      calibrated exports (some trial has ``peakR``); trials of older or
      synthetic files without it are never left out.
    0 turns either rule off.
    """
    info = []
    for f in files:
        with h5py.File(f, "r") as h:
            a = {k: _scalar(h.attrs[k]) for k in ("label", "path", "trial", "peakR") if k in h.attrs}
        digits = f.stem[len("trial_"):]
        trial = int(a["trial"]) if "trial" in a else (int(digits) if digits.isdigit() else 0)
        info.append((str(a.get("label", f.parent.parent.name)), _path_id(a.get("path"), f.parent.name), trial,
                     float(a["peakR"]) if "peakR" in a else float("nan")))
    reasons, report = filter_records(info, min_peak_r, min_path_trials)
    return [f for f, why in zip(files, reasons) if why is None], report


def filter_records(records, min_peak_r: float = 0.3, min_path_trials: int = 4) -> tuple[list[str | None], list[str]]:
    """The rules of ``filter_trials`` on (session label, path, trial, peak r)
    records, e.g. from an export's ``export_summary.json``: why each record
    is left out (None: kept), and the report lines."""
    report, reasons = [], [None] * len(records)
    if min_peak_r > 0:
        for i, (s, p, t, r) in enumerate(records):
            if r < min_peak_r:  # NaN (no attribute) compares False and is kept
                reasons[i] = f"calibration failed (peak r {r:.2f} < {min_peak_r:g})"
                report.append(f"{s} path {p} trial {t}: {reasons[i]}")
    if min_path_trials > 0:
        kept: dict[tuple, int] = {}
        calibrated: set[tuple] = set()
        for i, (s, p, _, r) in enumerate(records):
            kept[(s, p)] = kept.get((s, p), 0) + (reasons[i] is None)
            if r == r:  # has peakR
                calibrated.add((s, p))
        for key in sorted(calibrated, key=str):
            if kept[key] < min_path_trials:
                report.append(f"{key[0]} path {key[1]}: left out whole, {kept[key]} usable trials "
                              f"(< {min_path_trials})")
                for i, (s, p, _, _) in enumerate(records):
                    if (s, p) == key and reasons[i] is None:
                        reasons[i] = f"path left out whole: {kept[key]} usable trials (< {min_path_trials})"
    return reasons, report
