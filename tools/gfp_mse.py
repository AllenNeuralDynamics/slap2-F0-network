"""Time-resolved ground-truth error on a GFP control with injected events (companion of tools/gfp_check.py; reads
its cache; docs/plan.md, "S6", 2026-11-26).

8-frame means of (truth - baseline), where the truth is the ORIGINAL movie (no activity; its photon noise is common
to every model, so differences between models are meaningful). Reported against the trial mean image, as in the
held-out evaluation: 100 * (mse8_model / mse8_mean_image - 1), lower is better.
  original, all pixels      baseline of the original movie
  copy, source cores        baseline of the copy with injected events, at the injected sources' cores (peak pixel
                            and the valid pixels within 1.5 px); the mean image there is the copy's (it holds the
                            mean injected activity)
  oracle                    the original's own mean image (the best constant) in place of a model

usage: python tools/gfp_mse.py NAME=BASELINE_ROOT [...] --inj <injection dir> --cache <cache dir>
                               [--trial E1T1] [--copy 847983_inj] [--orig 847983_orig]

Same folders as tools/gfp_check.py, which must have run first with the same models (or a superset whose valid
pixels these models cover): of its cache files, the one with the most pixels valid in every model given is used.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

SMOOTH = 8


def smooth(x, keep, n=SMOOTH):
    """Running mean over n frames along axis 0, using kept, finite samples only (at least n/2 of them)."""
    ok = np.isfinite(x) & keep[:, None]
    num = np.cumsum(np.where(ok, x, 0.0), 0, dtype=np.float64)
    den = np.cumsum(ok, 0, dtype=np.float64)
    num = np.concatenate([np.zeros((1, x.shape[1])), num])
    den = np.concatenate([np.zeros((1, x.shape[1])), den])
    s, c = num[n:] - num[:-n], den[n:] - den[:-n]
    return np.where(c >= n / 2, s / np.maximum(c, 1), np.nan)


def baseline(root, trial, ds, dmd, idx):
    with h5py.File(Path(root) / ds / f"{trial}DMD{dmd}_BASELINE-80Hz.h5", "r") as f:
        b = f["baseline"]
        out = np.empty((b.shape[0], len(idx)), np.float32)
        for i0 in range(0, b.shape[0], 256):
            out[i0:i0 + 256] = b[i0:i0 + 256].reshape(-1, b.shape[1] * b.shape[2])[:, idx]
        return out, b.shape


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("models", nargs="+", metavar="NAME=BASELINE_ROOT")
    p.add_argument("--inj", type=Path, required=True, help="folder holding the copy and the original")
    p.add_argument("--cache", type=Path, required=True, help="tools/gfp_check.py's cache")
    p.add_argument("--trial", default="E1T1", help="trial stem of the files (default E1T1)")
    p.add_argument("--copy", default="847983_inj", help="folder of the copy with injected events")
    p.add_argument("--orig", default="847983_orig", help="folder of the original")
    a = p.parse_args()
    models = [m.split("=", 1) for m in a.models]
    rows = {}
    for dmd in (1, 2):
        # the cache whose pixels are valid in every model given (the largest that all models cover)
        best = None
        for cf in sorted(a.cache.glob(f"{a.trial}DMD{dmd}_*.npz")):
            z = np.load(cf)
            idx = z["idx"]
            good = True
            for _, root in models:
                for ds in (a.copy, a.orig):
                    with h5py.File(Path(root) / ds / f"{a.trial}DMD{dmd}_BASELINE-80Hz.h5", "r") as f:
                        b = f["baseline"]
                        good &= bool(np.isfinite(b[b.shape[0] // 2].ravel()[idx]).all())
            if good and (best is None or len(idx) > len(best["idx"])):
                best = {k: z[k] for k in z.files}
        if best is None:
            raise SystemExit(f"no cache file in {a.cache} covers every model on DMD{dmd}: run tools/gfp_check.py first")
        z = best
        idx, mi, mo, keep = z["idx"], z["m_inj"], z["m_orig"], z["keep"]
        with h5py.File(a.inj / a.copy / "injection_truth.h5", "r") as f:
            pk = f[f"path{dmd}/peak_pixel_rc"][:]
        _, shape = baseline(models[0][1], a.trial, a.orig, dmd, idx[:1])
        pa, pb = np.divmod(idx, shape[2])  # pa = MATLAB col, pb = MATLAB row
        core = np.unique(np.concatenate([np.flatnonzero(np.hypot(pa - c, pb - r) <= 1.5) for r, c in pk]))
        mean_o, mean_i = np.nanmean(mo[keep], 0), np.nanmean(mi[keep], 0)

        def mse(base, px=slice(None)):
            return float(np.nanmean(smooth((mo - base)[:, px], keep) ** 2))

        ref_all, ref_core = mse(mean_o[None]), mse(mean_i[None], core)
        rows.setdefault("oracle (the original's mean image)", []).append((0.0, 100 * (mse(mean_o[None], core) / ref_core - 1)))
        for name, root in models:
            bo, _ = baseline(root, a.trial, a.orig, dmd, idx)
            bi, _ = baseline(root, a.trial, a.copy, dmd, idx)
            rows.setdefault(name, []).append((100 * (mse(bo) / ref_all - 1), 100 * (mse(bi, core) / ref_core - 1)))
        print(f"DMD{dmd}: {len(idx)} px, {len(core)} core px, {int(keep.sum())} frames", flush=True)

    print("\n8-frame squared error against the truth, % relative to the mean image (lower is better); mean of the two paths (DMD1, DMD2)")
    print(f"{'':38s}{'original, all pixels':>28s}{'copy, source cores':>28s}")
    for name, v in rows.items():
        v = np.array(v)
        print(f"{name:38s}" + "".join(f"{v[:, j].mean():+8.2f}% ({v[0, j]:+.1f}, {v[1, j]:+.1f})".rjust(28) for j in range(2)))


if __name__ == "__main__":
    main()
