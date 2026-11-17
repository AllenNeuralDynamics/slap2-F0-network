"""Movie patches around the sources of two SILo runs (e.g. moving-median vs
network localization baselines), for checking what the sources look like.

usage: python source_patches.py MOCO_DIR A/experiment_summary.h5 B/experiment_summary.h5 OUT_DIR
           [--path Path2] [--dmd DMD2] [--radius 8] [--names main,net]

For every source of run A and every source of run B that A does not have
(one-to-one matches within 1.5 px), reads each trial's registered 80 Hz movie
(``*<dmd>_REGISTERED_DOWNSAMPLED-80Hz.tif``, SiLo's discards as NaN) and
writes OUT_DIR/<path>_patches.npy, float16 (source, frame, 2r+1, 2r+1) over
all trials in order, and OUT_DIR/<path>_sources.npz: coordinates (row, col),
category (0 both runs, 1 only A, 2 only B, 3 control pixel), SNR in each run, the trial
boundaries, and per source the mean image, valid mask, both runs' activity
images and GIAnT's spatial profile around it.
"""
import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from slap2_f0.registered import load_registered  # noqa: E402


def match(a, b, radius):
    """One-to-one matches within radius, closest pairs first: index pairs (i, j)."""
    if not len(a) or not len(b):
        return []
    d = np.hypot(*(np.subtract.outer(a[:, k], b[:, k]) for k in range(2)))
    i, j = np.nonzero(d <= radius)
    pairs, ua, ub = [], set(), set()
    for k in np.argsort(d[i, j], kind="stable"):
        if i[k] not in ua and j[k] not in ub:
            ua.add(i[k]); ub.add(j[k]); pairs.append((i[k], j[k]))
    return pairs


def load_run(fn, path):
    """Coordinates (row, col), SNR, profiles accessor, activity and mean image (row, col)."""
    f = h5py.File(fn, "r")
    g = f[path]
    xy = np.asarray(g["sources/spatial/coords"][()]).T[:, 1:3]  # rows [z, row, col]
    snr = np.asarray(g["sources/temporal/SNR"][()]).ravel()
    act = np.asarray(g["visualizations/act_im"][()])[:, :, 0].T
    mean = np.asarray(g["visualizations/mean_im"][()])[:, :, 0, 0].T
    return f, g, xy, snr, act, mean


def crop(img, r, c, rad, fill=np.nan):
    out = np.full((2 * rad + 1, 2 * rad + 1) + img.shape[2:], fill, np.float32)
    H, W = img.shape[:2]
    r0, c0 = r - rad, c - rad
    rs, re_, cs, ce = max(r0, 0), min(r0 + 2 * rad + 1, H), max(c0, 0), min(c0 + 2 * rad + 1, W)
    out[rs - r0:re_ - r0, cs - c0:ce - c0] = img[rs:re_, cs:ce]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("moco")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    ap.add_argument("--path", default="Path2")
    ap.add_argument("--dmd", default="DMD2")
    ap.add_argument("--radius", type=int, default=8)
    ap.add_argument("--names", default="main,net")
    ap.add_argument("--controls", type=int, default=0,
                    help="also this many control pixels (category 3) more than 6 px from every source, matched to "
                         "the only-A sources' brightness, share of zero frames and distance to the ROI edge "
                         "(in the first trial)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rad = args.radius
    fa, ga, xa, sa, act_a, mean_a = load_run(args.a, args.path)
    fb, gb, xb, sb, act_b, _ = load_run(args.b, args.path)
    pairs = match(xa, xb, 1.5)
    ia = {i: j for i, j in pairs}
    only_b = sorted(set(range(len(xb))) - {j for _, j in pairs})
    xy = np.vstack([xa, xb[only_b]]) if only_b else xa
    cat = np.array([0 if i in ia else 1 for i in range(len(xa))] + [2] * len(only_b))
    snr_a = np.concatenate([sa, np.full(len(only_b), np.nan)])
    snr_b = np.array([sb[ia[i]] if i in ia else np.nan for i in range(len(xa))] + list(sb[only_b]))
    src_run = np.array([0] * len(xa) + [1] * len(only_b))  # which run's profile
    src_idx = np.array(list(range(len(xa))) + only_b)
    rc = np.round(xy).astype(int)
    tifs = sorted(Path(args.moco).glob(f"*{args.dmd}_REGISTERED_DOWNSAMPLED-80Hz.tif"),
                  key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)])
    if args.controls:
        from scipy.ndimage import distance_transform_edt
        tif = tifs[0]
        movie, valid, _, _, _ = load_registered(tif, Path(str(tif).replace("_REGISTERED_DOWNSAMPLED-80Hz.tif",
                                                                               "_ALIGNMENTDATA.h5")))
        mean, zeros = np.nanmean(movie, 0), np.nanmean(movie == 0, 0)
        del movie
        edge = distance_transform_edt(valid)
        far = np.ones_like(valid)
        for r, c in rc:
            far[max(r - 6, 0):r + 7, max(c - 6, 0):c + 7] = False
        lost = rc[cat == 1]
        feats = np.stack([np.log(np.maximum(mean, 1e-3)), zeros, np.minimum(edge, 15) / 5], -1)
        cand = np.argwhere(valid & far & (edge >= 2))
        rng = np.random.default_rng(0)
        picks = []
        for k in rng.permutation(len(lost))[:args.controls] if len(lost) >= args.controls else \
                rng.integers(0, len(lost), args.controls):  # nearest unused candidate to each lost source's features
            f0 = feats[lost[k][0], lost[k][1]]
            d = np.abs(feats[cand[:, 0], cand[:, 1]] - f0).sum(1)
            for j in np.argsort(d):
                if not any(abs(cand[j] - q).max() <= 6 for q in picks):
                    picks.append(cand[j]); break
        picks = np.array(picks)
        xy = np.vstack([xy, picks]); rc = np.vstack([rc, picks])
        cat = np.concatenate([cat, np.full(len(picks), 3)])
        snr_a = np.concatenate([snr_a, np.full(len(picks), np.nan)]); snr_b = np.concatenate([snr_b, np.full(len(picks), np.nan)])
        src_run = np.concatenate([src_run, np.full(len(picks), -1)]); src_idx = np.concatenate([src_idx, np.full(len(picks), -1)])
    prof = []
    for k in range(len(xy)):
        if src_run[k] < 0:
            prof.append(np.zeros((2 * rad + 1, 2 * rad + 1), np.float32)); continue
        g = ga if src_run[k] == 0 else gb
        p = np.asarray(g["sources/spatial/profiles"][:, :, 0, src_idx[k]]).T
        prof.append(crop(p, rc[k, 0], rc[k, 1], rad, 0))
    print(f"{args.path}: {len(xa)} sources in {args.names.split(',')[0]} ({(cat == 1).sum()} only there), "
          f"{len(only_b)} only in {args.names.split(',')[1]}; {len(tifs)} trials", flush=True)
    chunks, bounds, valid_any = [], [0], None
    for tif in tifs:
        adata = Path(str(tif).replace("_REGISTERED_DOWNSAMPLED-80Hz.tif", "_ALIGNMENTDATA.h5"))
        movie, valid, discard, _, hz = load_registered(tif, adata)
        T = len(movie)
        p = np.stack([crop(movie.transpose(1, 2, 0), r, c, rad).transpose(2, 0, 1) for r, c in rc])
        chunks.append(p.astype(np.float16))
        valid_any = valid if valid_any is None else valid_any | valid
        bounds.append(bounds[-1] + T)
        print(f"  {tif.name}: {T} frames, {discard.sum()} discarded", flush=True)
        del movie
    patches = np.concatenate(chunks, 1)
    np.save(out / f"{args.path}_patches.npy", patches)
    np.savez(out / f"{args.path}_sources.npz", xy=xy, cat=cat, snr_a=snr_a, snr_b=snr_b, bounds=np.array(bounds),
             names=np.array(args.names.split(",")), radius=rad,
             mean=np.stack([crop(mean_a, r, c, rad) for r, c in rc]),
             valid=np.stack([crop(valid_any.astype(np.float32), r, c, rad, 0) for r, c in rc]),
             act_a=np.stack([crop(act_a, r, c, rad) for r, c in rc]),
             act_b=np.stack([crop(act_b, r, c, rad) for r, c in rc]),
             profile=np.stack(prof), act_full_a=act_a.astype(np.float32), act_full_b=act_b.astype(np.float32),
             mean_full=mean_a.astype(np.float32), valid_full=valid_any)
    print(f"wrote {out}: patches {patches.shape}", flush=True)


if __name__ == "__main__":
    main()
