"""What footprint support, background and labeled-outside pixels look like in exported
trials: brightness, photons per frame, zero frames, excess and variance factor
(docs/plan.md, "Low photon counts: the baseline sits low").

usage: python lowcount_charac.py ROOT [--sidecars DIR] [--cache DIR] [--out charac.tsv]

ROOT holds <session>/<path>/trial_*.h5. Stats come through `load_stats`, as in
training: a trial's own sidecar if it is current, else an overlay folder of
sidecars (--sidecars), else the cache folder (--cache), else they are computed.
Stale sidecars (e.g. one-channel exports such as 803496 and 804730) are thus
recomputed, which takes minutes per trial. One tab-separated row per trial; per
class (sup, bg, labns = labeled outside every footprint):
  B_*        the simulator's baseline (mean - mean activity) over the trial scale
  photons_*  mean count / gain (photons per frame)
  zero_*     share of zero frames
  exc_*      residual variance over photon noise (var_res / (gain x mean))
  lvf_*      log variance factor (relative to the trial median)
and the trial's gain, scale, the simulator's photon-event size factor q, and the
median distance of labeled-outside pixels to the nearest support pixel.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from slap2_f0.io import TrialFile  # noqa: E402
from slap2_f0.simulate import Template  # noqa: E402
from slap2_f0.stats import load_stats  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("root", type=Path)
    p.add_argument("--sidecars", default="", help="overlay folder(s) of sidecars, comma-separated")
    p.add_argument("--cache", default="", help="folder to cache computed stats in")
    p.add_argument("--out", default="")
    a = p.parse_args()
    lines = []
    for h5 in sorted(a.root.glob("*/Path*/trial_*.h5")):
        st = load_stats([h5], a.cache or None, sidecar_dirs=a.sidecars)[str(h5)]
        tf = TrialFile(h5)
        tp = Template(tf, st)
        v = tf.valid & np.isfinite(st["mean_movie"])
        cls = {"sup": v & tf.support, "bg": v & ~tf.labeled, "labns": v & tf.labeled & ~tf.support}
        mean = np.nan_to_num(st["mean_movie"])
        gain, s = float(st["gain"]), tf.scale
        B = np.nan_to_num(tp.B) / s
        exc = np.nan_to_num(st["var_res"]) / np.maximum(gain * mean, 1e-6)
        lvf = np.log(np.clip(np.nan_to_num(tf.varfac, nan=1.0), 1e-3, None))
        zero = np.nan_to_num(st["zero_movie"]) if "zero_movie" in st else np.full(mean.shape, np.nan)
        row = {"session": h5.parts[-3], "path": h5.parts[-2], "trial": h5.stem, "gain": gain, "scale": s, "q": tp.q,
               "frac_labns": cls["labns"].sum() / max(v.sum(), 1)}
        for c, m in cls.items():
            if not m.any():
                continue
            row[f"B_{c}"] = np.median(B[m])
            row[f"photons_{c}"] = np.median(mean[m]) / gain
            row[f"zero_{c}"] = np.median(zero[m])
            row[f"exc_{c}"] = np.median(exc[m])
            row[f"lvf_{c}"] = np.median(lvf[m])
        if cls["labns"].any() and cls["sup"].any():
            row["dist_labns"] = np.median(distance_transform_edt(~cls["sup"])[cls["labns"]])
        tf.close()
        keys = ["session", "path", "trial", "gain", "scale", "q", "frac_labns"] + \
            [f"{q}_{c}" for c in cls for q in ("B", "photons", "zero", "exc", "lvf")] + ["dist_labns"]
        if not lines:
            lines.append("\t".join(keys))
        lines.append("\t".join(f"{row.get(k, np.nan):.4g}" if not isinstance(row.get(k), str) else row[k] for k in keys))
        print(lines[-1], flush=True)
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
