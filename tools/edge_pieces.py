"""Edge bias from cut trials: no ground truth or GIAnT activity needed.

usage:
  python edge_pieces.py OUT.npz --trials T1.h5 [T2.h5 ...] --ckpt name=path [name=path ...] [--piece 512]
  python edge_pieces.py OUT.npz --registered MOCO_DIR --files E1T1DMD1,E1T2DMD2 --ckpt name=path ...

Each trial (exported or simulated trial files, or GIAnT's registered 80 Hz
movies) is predicted whole and in pieces of --piece frames, each piece as if
it were a trial of its own but with the whole trial's mean image, variance
maps and gain, so only the truncation in time differs. At a cut, the pieces'
baselines next to it are at a trial edge while the whole trial's are in its
middle, so (piece - whole) / whole by distance from the cut is the edge bias
alone, on real data. Beyond the model's temporal reach (263 frames for v6)
it must be 0. In simulated trials (truth/baseline) the bias against the
truth is also reported, for the whole trial by distance from its own edges
and for the pieces by distance from any edge. Pixels: all valid pixels, and
footprint support and background where the trial files have the masks.
"""
import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from slap2_f0.predict import load_checkpoint, predict_trial, prepare, valid_index  # noqa: E402
from slap2_f0.registered import load_registered, movie_stats  # noqa: E402

BINS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 400)]


class Piece:
    """Frames [a, b) of a trial held in memory, with the whole trial's maps."""

    def __init__(self, movie, discard, valid, mean_image, varfac, scale, a, b):
        self.n_frames, self.n_rows, self.n_cols = b - a, movie.shape[1], movie.shape[2]
        self.channel = 0
        self._f = {"movie": movie[None, a:b]}
        self.discard = discard[a:b]
        self.valid, self.mean_image, self.varfac, self.scale = valid, mean_image, varfac, scale


def load_trial(fn):
    """movie (T, R, C) float32, valid, discard, mean image, varfac (relative), scale, masks, truth or None."""
    from slap2_f0.io import TrialFile
    with TrialFile(fn) as tf:
        movie = tf._f["movie"][tf.channel].astype(np.float32)
        out = dict(movie=movie, valid=tf.valid, discard=tf.discard, mean_image=tf.mean_image, varfac=tf.varfac,
                   scale=tf.scale, support=tf.support, bg=tf.valid & ~tf.labeled)
        out["truth"] = tf._f["truth/baseline"][()].astype(np.float32) if "truth/baseline" in tf._f else None
    return out


def load_reg(moco, stem):
    movie, valid, discard, varfac, _ = load_registered(Path(moco) / f"{stem}_REGISTERED_DOWNSAMPLED-80Hz.tif",
                                                       Path(moco) / f"{stem}_ALIGNMENTDATA.h5")
    ok = ~discard
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(valid, np.nanmean(movie[ok], 0), np.nan).astype(np.float32)
    vf = np.nan_to_num(varfac.astype(np.float32))
    okv = valid & (vf > 0)
    vf = vf / np.float32(np.median(vf[okv]) if okv.any() else 1.0)
    good = valid & np.isfinite(mean)
    return dict(movie=movie, valid=valid, discard=discard, mean_image=mean, varfac=vf,
                scale=float(np.nanmedian(mean[good])), support=None, bg=None, truth=None)


def run_model(model, wcfg, d, st, device, amp, piece):
    T = d["movie"].shape[0]
    whole = Piece(d["movie"], d["discard"], d["valid"], d["mean_image"], d["varfac"], d["scale"], 0, T)
    rows, cols = valid_index(whole)
    full, _ = predict_trial(model, wcfg, whole, st, device, amp, rows=rows, cols=cols)
    cut = np.full_like(full, np.nan)
    starts = list(range(0, T, piece))
    if T - starts[-1] < piece // 2 and len(starts) > 1:  # no short last piece
        starts.pop()
    bounds = starts + [T]
    for a, b in zip(bounds[:-1], bounds[1:]):
        p = Piece(d["movie"], d["discard"], d["valid"], d["mean_image"], d["varfac"], d["scale"], a, b)
        cut[a:b], _ = predict_trial(model, wcfg, p, st, device, amp, rows=rows, cols=cols)
    return rows, cols, full, cut, bounds


def binned(per_frame, positions):
    """per_frame (T,), positions: list of (frame index array, distance array) -> bin means."""
    out = []
    for lo, hi in BINS:
        v = [per_frame[i][(dd >= lo) & (dd < hi)] for i, dd in positions]
        v = np.concatenate(v) if v else np.array([])
        out.append(np.nanmean(v) if v.size else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--trials", nargs="*", default=[])
    ap.add_argument("--registered", default="")
    ap.add_argument("--files", default="")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--piece", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    items = [("file", t) for t in args.trials] + [("reg", s) for s in args.files.split(",") if s and args.registered]
    models = []
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        model, wcfg, _ = load_checkpoint(path, device)
        amp = prepare(model, device, "auto")
        models.append((name, model, amp, wcfg))
    res = {}
    for kind, item in items:
        t0 = time.time()
        d = load_trial(item) if kind == "file" else load_reg(args.registered, item)
        st = movie_stats(d["movie"], d["valid"], d["discard"])
        label = Path(item).parent.parent.name + "/" + Path(item).name if kind == "file" else item
        for name, model, amp, wcfg in models:
            rows, cols, full, cut, bounds = run_model(model, wcfg, d, st, device, amp, args.piece)
            T = full.shape[0]
            keep = ~d["discard"]
            classes = {"all": np.ones(len(rows), bool)}
            if d["support"] is not None:
                classes["sup"] = d["support"][rows, cols]
                classes["bg"] = d["bg"][rows, cols]
            truth = d["truth"][:, rows, cols] if d["truth"] is not None else None
            cuts = bounds[1:-1]
            t = np.arange(T)
            after = [(np.nonzero((t >= c) & (t < c + 400))[0], t[(t >= c) & (t < c + 400)] - c) for c in cuts]
            before = [(np.nonzero((t < c) & (t >= c - 400))[0], c - 1 - t[(t < c) & (t >= c - 400)]) for c in cuts]
            for cname, cm in classes.items():
                if not cm.any():
                    continue
                f, c_ = full[:, cm], cut[:, cm]
                diff = 100 * np.nanmean(c_ - f, 1) / np.nanmean(f)
                diff[~keep] = np.nan
                r = {"cut_after": binned(diff, after), "cut_before": binned(diff, before)}
                if truth is not None:
                    tr = truth[:, cm]
                    bf = 100 * np.nanmean(f - tr, 1) / np.nanmean(tr)
                    bc = 100 * np.nanmean(c_ - tr, 1) / np.nanmean(tr)
                    bf[~keep] = np.nan
                    bc[~keep] = np.nan
                    trial_start = [(np.nonzero(t < 400)[0], t[t < 400])]
                    trial_end = [(np.nonzero(t >= T - 400)[0], T - 1 - t[t >= T - 400])]
                    mid = slice(400, T - 400)
                    r.update(full_start=binned(bf, trial_start), full_end=binned(bf, trial_end),
                             full_mid=float(np.nanmean(bf[mid])), piece_after=binned(bc, after),
                             piece_before=binned(bc, before))
                res[(label, name, cname)] = r
            print(f"{label} {name}: {len(cuts)} cuts, {time.time() - t0:.0f} s; all px, piece - whole after a cut: "
                  + " ".join(f"{v:+.2f}" for v in res[(label, name, 'all')]["cut_after"])
                  + " | before: " + " ".join(f"{v:+.2f}" for v in res[(label, name, 'all')]["cut_before"]), flush=True)
    np.savez(args.out, res=np.array([(k, v) for k, v in res.items()], dtype=object), bins=np.array(BINS))
    # summary: mean over trials per model and pixel class
    print(f"\nbins (frames from the edge): {BINS}")
    for name, *_ in models:
        for cname in ("all", "sup", "bg"):
            rs = [v for (lab, n, c), v in res.items() if n == name and c == cname]
            if not rs:
                continue
            for key in ("cut_after", "cut_before", "full_start", "full_end", "piece_after", "piece_before"):
                vals = [r[key] for r in rs if key in r]
                if vals:
                    m = np.nanmean(np.array(vals, float), 0)
                    sd = np.nanstd(np.array(vals, float), 0) / np.sqrt(len(vals))
                    print(f"{name:16s} {cname:4s} {key:13s} " + " ".join(f"{a:+6.2f} +-{b:4.2f}" for a, b in zip(m, sd))
                          + (f"   (mid {np.nanmean([r['full_mid'] for r in rs if 'full_mid' in r]):+.2f})"
                             if key.startswith("full") else "") + f"  n={len(vals)}")


if __name__ == "__main__":
    main()
