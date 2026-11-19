"""Does a network baseline follow residual motion? Simulated trials with known motion.

usage:
  python motion_test.py simulate SIM_DIR TRIAL.h5 [TRIAL.h5 ...] [--kinds none,fast,medium,slow] [--sd 0.25] [--substeps 0] [--focus_sd 0.15] [--stats DIR]
  slap2-f0 predict SIM_DIR --ckpt name=path ... --out PRED_DIR --no_movie --no_silo
  python motion_test.py score SIM_DIR PRED_DIR [--models a,b]

simulate: each exported trial is the template (slap2_f0.simulate: the
fast-brightness simulator with realistic low counts, as v6-v9 were trained
on) of one simulated trial per kind, all from the same seed. The sample
moves rigidly by d(t), one OU process per axis with SD --sd pixels and time
constants of 1.5-4 frames ("fast", inside the 8 hidden frames, so only the
surround outside the box shows it), 8-30 frames ("medium") or 80-320 frames
("slow", 1-4 s, visible in the pixel's own trace); "jitter" (0.3-1 frames)
needs --substeps. "none" has no motion; its baseline is the other kinds'
static baseline exactly (motion is drawn last). --substeps n draws the
motion at n steps per frame and pools it through the frame-formation
kernel (SimConfig.motion_substeps): frames see the kernel-weighted mean
displacement, smaller than --sd for fast motion, and are blurred by its
spread within the kernel. Use a separate SIM_DIR for it. "focus" has no
motion but focus fluctuations (SimConfig.focus_sd_px2 = --focus_sd px^2):
the frames' blur changes, fast and slow.

score: the baseline's motion part is m = truth (moved) - truth_static. For
each checkpoint, the change of its baseline from the no-motion trial (which
shares everything but the motion and the noise draw) to the moving one is
regressed on m at the kept frames, after removing each pixel's mean over
the trial: "followed" 1 = the baseline follows the motion, 0 = it ignores
it. A 17-frame running mean (the hidden frames) splits both into a slow and
a fast part, each with its own slope. Pixels: all valid, "moving" (the
quarter where m varies most), footprint support and background. Also: the
slope of the error against the static truth (prediction - truth_static) on
m, without the no-motion trial; the same in expectation, but the baseline's
other errors correlate with m by chance, most in the slow part.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

KINDS = {"none": None, "jitter": (0.3, 1.0), "fast": (1.5, 4.0), "medium": (8.0, 30.0), "slow": (80.0, 320.0),
         "focus": "focus"}
RUN = 17  # frames of the running mean that splits slow from fast
SEED = 888_000


def simulate(args):
    from slap2_f0.io import TrialFile
    from slap2_f0.simulate import SimConfig, Template, write_trial
    from slap2_f0.stats import load_stats
    from slap2_f0.train import FAST_SIM
    out = Path(args.out)
    stats = load_stats(args.trials, args.stats or (out / "stats"))
    for fn in args.trials:
        with TrialFile(fn) as tf:
            session = str(tf.attrs.get("label", Path(fn).parent.parent.name))
            tp = Template(tf, stats[str(fn)])
            for kind in args.kinds.split(","):
                tau = KINDS[kind]
                if tau == "focus":
                    mot = dict(focus_sd_px2=(args.focus_sd, args.focus_sd))
                else:
                    mot = {} if tau is None else dict(motion_sd_px=(args.sd, args.sd), motion_tau_frames=tau,
                                                      motion_substeps=args.substeps)
                cfg = SimConfig(**FAST_SIM, photon_zeros=True, **mot)
                dest = out / f"{session}_{kind}" / "Path1" / "trial_001.h5"
                if dest.exists():
                    print(f"{dest} exists, kept", flush=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".tmp")
                r = write_trial(tp, tmp, cfg, seed=SEED)
                with h5py.File(tmp, "a") as f:
                    f.attrs["label"] = f"{session}_{kind}"
                tmp.rename(dest)
                sd = r["motion"].std(1) if "motion" in r else (0, 0)
                spread = (", spread within frames " + " / ".join(f"{v:.2f}" for v in np.sqrt(r["motion_cov"][:2].mean(1)))
                          if "motion_cov" in r else "")
                if "focus" in r:
                    spread += ", focus SD " + " / ".join(f"{v:.3f}" for v in r["focus"].std(1)) + " px^2"
                print(f"{dest}: motion SD {sd[0]:.2f} / {sd[1]:.2f} px (rows / cols){spread}", flush=True)


def _centered(a, keep):
    """a (T, N) with discarded frames NaN -> minus each pixel's mean; NaN at discarded frames."""
    a = np.where(keep[:, None], a, np.nan)
    return a - np.nanmean(a, 0)


def _runmean(a):
    """Centered running mean over RUN frames along axis 0, ignoring NaN."""
    h = RUN // 2
    v = np.isfinite(a)
    cs = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(np.where(v, a, 0), 0, dtype=np.float64)])
    cn = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(v, 0, dtype=np.float64)])
    T = a.shape[0]
    lo, hi = np.clip(np.arange(T) - h, 0, T), np.clip(np.arange(T) + h + 1, 0, T)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (cs[hi] - cs[lo]) / (cn[hi] - cn[lo])
    return np.where(v, out, np.nan).astype(np.float32)


def _sums(e, m):
    ok = np.isfinite(e) & np.isfinite(m)
    e, m = np.where(ok, e, 0), np.where(ok, m, 0)
    return np.array([(e * m).sum(), (m * m).sum(), (e * e).sum(), ((e - m) ** 2).sum()], np.float64)


def score(args):
    sim, pred = Path(args.sim), Path(args.pred)
    want = [m for m in args.models.split(",") if m]
    acc = {}  # (kind, model, px, band) -> list of per-trial sums
    for fn in sorted(sim.rglob("trial_001.h5")):
        label = fn.parent.parent.name
        session, kind = label.rsplit("_", 1)
        if kind == "none" or kind not in KINDS:
            continue
        pf, nf = pred / label / "Path1" / fn.name, pred / f"{session}_none" / "Path1" / fn.name
        if not pf.exists():
            print(f"{label}: no predictions", flush=True)
            continue
        with h5py.File(fn, "r") as f, h5py.File(pf, "r") as p:
            rr, cc = p["rows"][()], p["cols"][()]
            keep = p["frame/discard"][()] == 0
            sup = p["mask/footprint_support"][()][rr, cc].astype(bool)
            bg = ~p["mask/labeled"][()][rr, cc].astype(bool)
            ts = f["truth/baseline_static"][()][:, rr, cc]
            mm = _centered(f["truth/baseline"][()][:, rr, cc] - ts, keep)
            sdm = np.nanstd(mm, 0)
            classes = {"all": np.ones(len(rr), bool), "moving": sdm >= np.nanquantile(sdm, 0.75), "sup": sup, "bg": bg}
            m_parts = {"all": mm}
            m_parts["slow"] = _runmean(mm)
            m_parts["fast"] = mm - m_parts["slow"]
            errs = {}
            for name in (want or list(p["baseline"].keys())):
                if name in p["baseline"]:
                    errs[name + " vs truth"] = p[f"baseline/{name}"][()].astype(np.float32) - ts
            if nf.exists():
                with h5py.File(nf, "r") as q:
                    for name in [n.replace(" vs truth", "") for n in errs]:
                        if name in q["baseline"]:
                            errs[name] = errs[name + " vs truth"] + ts - q[f"baseline/{name}"][()].astype(np.float32)
            for name, e in errs.items():
                ec = _centered(e, keep)
                e_parts = {"all": ec, "slow": _runmean(ec)}
                e_parts["fast"] = ec - e_parts["slow"]
                for band in ("all", "slow", "fast"):
                    for cname, cm in classes.items():
                        if cm.any():
                            acc.setdefault((kind, name, cname, band), []).append(_sums(e_parts[band][:, cm], m_parts[band][:, cm]))
            msd = {b: float(np.sqrt(np.nanmean(m_parts[b][:, classes["moving"]] ** 2) / np.nanmean(ts[:, classes["moving"]]) ** 2))
                   for b in m_parts}
            print(f"{label}: {len(rr)} px; motion part RMS at moving pixels, in % of their baseline: "
                  + ", ".join(f"{b} {100 * v:.1f}" for b, v in msd.items()), flush=True)
    print("\nfollowed: slope of (baseline, moving trial - baseline, no-motion trial) on the motion part; "
          "'vs truth': slope of (baseline - static truth) on it. Mean over trials +- SE")
    print(f"{'kind':7s} {'model':26s} {'px':7s} " + " ".join(f"{b:>14s}" for b in ("all", "slow", "fast")))
    for kind in [k for k in KINDS if k != "none"]:
        names = sorted({n for (k, n, c, b) in acc if k == kind}, key=lambda n: (n.replace(" vs truth", ""), n.endswith("truth")))
        for name in names:
            for cname in ("all", "moving", "sup", "bg"):
                cells = []
                for band in ("all", "slow", "fast"):
                    s = acc.get((kind, name, cname, band))
                    if not s:
                        cells.append(f"{'':>14s}")
                        continue
                    sl = np.array(s)[:, 0] / np.array(s)[:, 1]
                    se = sl.std() / np.sqrt(len(sl)) if len(sl) > 1 else np.nan
                    cells.append(f"{sl.mean():+6.2f} +-{se:4.2f}")
                print(f"{kind:7s} {name:26s} {cname:7s} " + " ".join(f"{c:>14s}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("simulate")
    ps.add_argument("out")
    ps.add_argument("trials", nargs="+")
    ps.add_argument("--kinds", default="none,fast,medium,slow")
    ps.add_argument("--sd", type=float, default=0.25)
    ps.add_argument("--substeps", type=int, default=0)
    ps.add_argument("--focus_sd", type=float, default=0.15)
    ps.add_argument("--stats", default="")
    ps.set_defaults(func=simulate)
    pc = sub.add_parser("score")
    pc.add_argument("sim")
    pc.add_argument("pred")
    pc.add_argument("--models", default="")
    pc.set_defaults(func=score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
