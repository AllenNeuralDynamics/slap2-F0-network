"""Decimation test (slap2-F0-network#33): s6_ft baselines decimated in time to
40, 20 and 10 Hz and interpolated back to the 80 Hz frames, plus the spectrum
of the full-rate baseline.

Decimation by factor k (2, 4, 8): the mean over blocks of k frames (the last
block may be shorter), placed at the block's centre, then linear
interpolation back to every frame (held constant before the first and after
the last centre). This is what a network that works at the lower rate and is
interpolated to the frames would give if its low-rate samples were the
full-rate baseline's block means.

usage: decimate.py [DS ...]  -> OUT/baselines/s6_ft_d{40,20,10}/<ds>/ (SRC and OUT: the paths below,
                               the 2026-11-27 run's; edit them for other data)
                                infspeed/decim/spectrum.json, decimation_error.json
Files keep the source's layout (/baseline (frame, A, B) float16, NaN outside
the valid pixels), chunks, compression and attributes, plus decimate_k and
decimate_hz. No giant silo record (.json) is written: giant silo accepts a
baseline file without one.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

SRC = {
    "780125_5tr": Path("D:/GIAnT_testdata/siloref/wp6/f0wire/780125_5tr/baselines"),
    "847983_inj": Path("D:/GIAnT_testdata/siloref/wp6/baselines/s6_ft/847983_inj"),
    "847983_orig": Path("D:/GIAnT_testdata/siloref/wp6/baselines/s6_ft/847983_orig"),
}
OUT = Path("D:/GIAnT_testdata/siloref/infspeed/decim")
FACTORS = {2: "d40", 4: "d20", 8: "d10"}
HZ = 80.0


def block_centres(T: int, k: int) -> np.ndarray:
    """Centre (in frames) of each block of k frames; the last may be shorter."""
    starts = np.arange(0, T, k)
    ends = np.minimum(starts + k, T)
    return (starts + ends - 1) / 2.0


def decimate(b: np.ndarray, k: int) -> np.ndarray:
    """b (T, N) float32 -> block means at the block centres, linearly
    interpolated back to frames 0..T-1 (constant beyond the end centres)."""
    T, N = b.shape
    nb = -(-T // k)
    pad = np.full((nb * k, N), np.nan, np.float32)
    pad[:T] = b
    with np.errstate(invalid="ignore"):
        means = np.nanmean(pad.reshape(nb, k, N), axis=1)
    c = block_centres(T, k)
    t = np.arange(T, dtype=np.float64)
    j = np.clip(np.searchsorted(c, t, side="right") - 1, 0, nb - 1)
    j1 = np.minimum(j + 1, nb - 1)
    span = np.where(j1 > j, c[j1] - c[j], 1.0)
    w = np.clip((t - c[j]) / span, 0.0, 1.0).astype(np.float32)[:, None]
    return (1 - w) * means[j] + w * means[j1]


def power_shares(b: np.ndarray, cuts=(5.0, 10.0, 20.0)) -> dict:
    """Share of each pixel's power (mean removed, Hann window) above each cut
    frequency: median over pixels and pooled over pixels."""
    T = b.shape[0]
    x = b - b.mean(0, keepdims=True)
    x *= np.hanning(T).astype(np.float32)[:, None]
    P = np.abs(np.fft.rfft(x, axis=0)) ** 2
    f = np.fft.rfftfreq(T, 1.0 / HZ)
    tot = P[f > 0].sum(0)
    ok = tot > 0
    out = {}
    for fc in cuts:
        above = P[f > fc].sum(0)
        out[f">{fc:g}Hz"] = {
            "median": float(np.median(above[ok] / tot[ok])),
            "p90": float(np.percentile(above[ok] / tot[ok], 90)),
            "pooled": float(above[ok].sum() / tot[ok].sum()),
        }
    # relative SD of the baseline over time (how much it moves at all)
    rel = b.std(0) / np.maximum(b.mean(0), 1e-6)
    out["rel_sd_median"] = float(np.median(rel))
    return out


def main():
    only = sys.argv[1:] or list(SRC)
    spec, err = {}, {}
    for ds in only:
        for f in sorted(SRC[ds].glob("*_BASELINE-80Hz.h5")):
            with h5py.File(f, "r") as h:
                d = h["baseline"]
                full = d[:].astype(np.float32)
                attrs = dict(h.attrs)
                chunks, comp, copts, shuffle = d.chunks, d.compression, d.compression_opts, d.shuffle
            T = full.shape[0]
            valid = np.isfinite(full).all(0)
            b = full[:, valid]  # (T, N)
            key = f"{ds}/{f.name.split('_BASELINE')[0]}"
            spec[key] = power_shares(b)
            err[key] = {}
            for k, tag in FACTORS.items():
                dec = decimate(b, k)
                rel = np.abs(dec - b) / np.maximum(np.abs(b), 1e-6)
                err[key][tag] = {
                    "max_abs_counts": float(np.abs(dec - b).max()),
                    "p999_abs_counts": float(np.percentile(np.abs(dec - b), 99.9)),
                    "rel_median": float(np.median(rel)),
                    "rel_p99": float(np.percentile(rel, 99)),
                    "rel_p999": float(np.percentile(rel, 99.9)),
                    "rel_max": float(rel.max()),
                }
                o = np.full(full.shape, np.nan, np.float16)
                o[:, valid] = np.clip(dec, -65000, 65000).astype(np.float16)
                dest = OUT / "baselines" / f"s6_ft_{tag}" / ds / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(dest, "w") as h:
                    h.create_dataset("baseline", data=o, chunks=chunks, compression=comp,
                                     compression_opts=copts, shuffle=shuffle)
                    h.attrs.update(attrs)
                    h.attrs["decimate_k"] = k
                    h.attrs["decimate_hz"] = HZ / k
                    h.attrs["decimate_source"] = str(f)
            print(key, T, int(valid.sum()), json.dumps(spec[key]), json.dumps(err[key]), flush=True)
    tag = "_".join(only)
    (OUT / f"spectrum_{tag}.json").write_text(json.dumps(spec, indent=1))
    (OUT / f"decimation_error_{tag}.json").write_text(json.dumps(err, indent=1))


if __name__ == "__main__":
    main()
