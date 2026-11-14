"""Read trial files written by ``code/matlab/exportF0TrainingData.m``.

Each file holds one trial of one imaging path on the trial's 80 Hz registered
grid, cropped to the box around the valid pixels. Arrays (C order):

    movie       (channel, frame, row, col)  registered movie, NaN where not
                                            imaged or discarded
    activity    (channel, frame, row, col)  GIAnT's fitted activity, same grid
                                            and timing; target = movie - activity
    mean_image  (channel, row, col)
    varfac      (row, col) or (frame, row, col)
    mask/valid, mask/labeled, mask/footprint_support   (row, col) uint8
    frame/discard, frame/motion_row, ...               (frame,)
    sources/peak_row, sources/peak_col                 (source,)

Attributes on the root group hold QC and provenance (see the export function).
"""

from __future__ import annotations

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
        self.mean_image = self._f["mean_image"][channel].astype(np.float32)
        vf = self._f["varfac"]
        self.varfac = (vf[:] if vf.ndim == 2 else np.nanmean(vf[:], axis=0)).astype(np.float32)
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
