"""Fine-tuning trade-off: real-data gain against simulation accuracy (docs/plan.md, 2026-11-21).

usage: python finetune_tradeoff.py RESULTS_DIR [--runs S0,S1,S2,S3] [--test ...] [--val ...] [--drop_sim 804730]
       python finetune_tradeoff.py --averages SOUPS_CSV

RESULTS_DIR holds the held-out evaluation runs (`slap2-f0 evaluate --per_window`), one folder per
run named res_E<run>ft (fine-tuning steps) and res_Ept* (pretraining steps), as for heldout_eval.py.

Printed:
1. per fine-tuning step, % change of the 8-frame MSE (balanced over support and background,
   then mice): real vs the mean image (also per class, and vs SiLo); sim and fast vs the oracle
   (per class, and labeled pixels outside the support from eval.csv); activity-free vs the mean
   image (the true mean there); the support leak;
2. pretraining steps on fast windows;
3. the real-data gain of fine-tuning (mean of steps >= 750 minus step 0) on all held-out mice,
   with 95% intervals over mice and windows, and the share of the gap between the mean image and
   the oracle (the best constant baseline) that the network closes, per class;
4. fine-tuning checkpoints (step 0 included) chosen by the validation mice's real score + w x
   their fast score, read on the test mice;
5. within-run rank correlations between readouts.

--averages: summary of a per-window table of model variants scored on the same windows
(columns kind, session, window, model, sup_/bg_ n8 and se8_<net|mean_image|silo|oracle>), e.g.
weight averages of kept steps made with `slap2-f0 average`, with paired intervals over windows.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PREDS = ("net", "mean_image", "silo", "oracle")
TEST = ("776270", "810196", "829704", "876767")
VAL = ("878030", "794237", "834788", "804730")
STEPS = (0, 250, 500, 750, 1000, 1250, 1500, 1700)
LATE = (750, 1000, 1250, 1500, 1700)


def mouse(s):
    return s.split("_")[0]


def load(d: Path):
    """windows: (run, step, mouse, kind) -> arrays over windows; rows: same keys -> eval.csv row."""
    W, E = {}, {}
    acc = {}
    with open(d / "windows.csv") as fh:
        rd = csv.reader(fh)
        num = next(rd)[4:]
        for row in rd:
            run, step = row[0].split("@")
            acc.setdefault((run, int(step.split("_")[1]), mouse(row[2]), row[3]), []).append([float(x) for x in row[4:]])
    for k, v in acc.items():
        a = np.array(v)
        W[k] = {n: a[:, i] for i, n in enumerate(num)}
    with open(d / "eval.csv") as fh:
        for r in csv.DictReader(fh):
            run, step = r["checkpoint"].split("@")
            E[(run, int(step.split("_")[1]), mouse(r["session"]), r["kind"])] = {
                k: float(v) for k, v in r.items() if k not in ("checkpoint", "session", "kind")}
    return W, E


class Scores:
    def __init__(self, W, E, drop_sim):
        self.W, self.E, self.drop = W, E, set(drop_sim)

    def mice(self, kind, mice):
        return [m for m in mice if kind == "real" or m not in self.drop]

    def sums(self, run, steps, kind, mice, ref):
        """(steps, mice, class, windows) error sums of the net and the reference."""
        ms = self.mice(kind, mice)
        g = lambda s, m, c, p: self.W[(run, s, m, kind)][f"{c}_se8_{p}"]  # noqa: E731
        A = np.array([[[g(s, m, c, "net") for c in ("sup", "bg")] for m in ms] for s in steps])
        B = np.array([[[g(s, m, c, ref) for c in ("sup", "bg")] for m in ms] for s in steps])
        return A, B

    def eval_mean(self, run, steps, kind, mice, key, ref_key=None):
        out = []
        for s in steps:
            v = []
            for m in self.mice(kind, mice):
                d = self.E[(run, s, m, kind)]
                x = d[key] / d[ref_key] - 1 if ref_key else d[key]
                if np.isfinite(x):
                    v.append(x)
            out.append(100 * np.mean(v) if ref_key else np.mean(v))
        return np.array(out)


def pct(A, B, cls="bal", w=None, mm=None):
    """% change per step from error sums (steps, mice, class, windows); w (mice, windows) window weights,
    mm (mice,) mouse multiplicities (bootstrap)."""
    a = A.sum(-1) if w is None else np.einsum("smcw,mw->smc", A, w)
    b = B.sum(-1) if w is None else np.einsum("smcw,mw->smc", B, w)
    r = a / b
    r = {"bal": r.mean(-1), "sup": r[..., 0], "bg": r[..., 1]}[cls]
    mm = np.ones(r.shape[1]) if mm is None else mm
    return 100 * ((r * mm).sum(-1) / mm.sum() - 1)


def trajectories(S, runs, test):
    print(f"1. fine-tuning steps, test mice ({', '.join(test)}; simulated kinds without {', '.join(sorted(S.drop))})")
    hdr = ["step", "real", "r.sup", "r.bg", "r.silo", "sim", "s.sup", "s.bg", "s.lab", "fast", "f.sup", "f.bg", "f.lab",
           "free", "fr.lab", "leak"]
    for run in runs:
        k = f"{run}ft"
        print(f"\n{run}\n" + " ".join(f"{h:>6s}" for h in hdr))
        cols = []
        for kind, ref in (("real", "mean_image"), ("sim", "oracle"), ("fast", "oracle"), ("free", "mean_image")):
            A, B = S.sums(k, STEPS, kind, test, ref)
            if kind == "real":
                cols += [pct(A, B), pct(A, B, "sup"), pct(A, B, "bg"), pct(*S.sums(k, STEPS, kind, test, "silo"))]
            elif kind == "free":
                cols += [pct(A, B), S.eval_mean(k, STEPS, kind, test, "lab_mse8_net", "lab_mse8_mean_image")]
            else:
                cols += [pct(A, B), pct(A, B, "sup"), pct(A, B, "bg"),
                         S.eval_mean(k, STEPS, kind, test, "lab_mse8_net", f"lab_mse8_{ref}")]
        cols.append(S.eval_mean(k, STEPS, "fast", test, "sup_leak_net"))
        for i, s in enumerate(STEPS):
            print(f"{s:6d} " + " ".join(f"{c[i]:6.2f}" if j < len(cols) - 1 else f"{c[i]:6.3f}"
                                        for j, c in enumerate(cols)))


def pretraining(S, runs, test):
    print("\n2. pretraining steps, fast vs oracle, test mice")
    for run in runs:
        k = f"{run}pt"
        steps = sorted({key[1] for key in S.W if key[0] == k})
        if not steps:
            continue
        f = pct(*S.sums(k, steps, "fast", test, "oracle"))
        late = f[np.array(steps) >= 1500]
        print(f"  {run}: " + " ".join(f"{s}:{x:.1f}" for s, x in zip(steps, f))
              + f"  | steps >= 1500: mean {late.mean():.1f}, SD {late.std(ddof=1):.1f}")


def real_gain(S, runs, mice, B=1000):
    rng = np.random.default_rng(0)
    late = [STEPS.index(s) for s in LATE]
    print(f"\n3. real-data gain of fine-tuning (mean of steps {LATE[0]}-{LATE[-1]} minus step 0), percentage points, "
          f"mice {', '.join(mice)}; 95% intervals [mice and windows] [windows]")
    for ref in ("mean_image", "silo", "oracle"):
        for run in runs:
            A, R = S.sums(f"{run}ft", STEPS, "real", mice, ref)
            nm, nw = A.shape[1], A.shape[-1]

            def diff(w=None, mm=None):
                s = pct(A, R, w=w, mm=mm)
                return s[late].mean() - s[0], s[0], s[late].mean()

            d0, s0, sl = diff()
            bm, bw = [], []
            for _ in range(B):
                ww = np.stack([np.bincount(rng.integers(0, nw, nw), minlength=nw) for _ in range(nm)]).astype(float)
                bw.append(diff(ww)[0])
                bm.append(diff(ww, np.bincount(rng.integers(0, nm, nm), minlength=nm))[0])
            print(f"  vs {ref:10s} {run}: step 0 {s0:+.2f}, late {sl:+.2f}, gain {d0:+.2f} "
                  f"[{np.percentile(bm, 2.5):+.2f}, {np.percentile(bm, 97.5):+.2f}] "
                  f"[{np.percentile(bw, 2.5):+.2f}, {np.percentile(bw, 97.5):+.2f}]")
    print("  share of the gap from the mean image to the oracle closed on real windows (support / background), by step:")
    for run in runs:
        A, M = S.sums(f"{run}ft", STEPS, "real", mice, "mean_image")
        _, O = S.sums(f"{run}ft", STEPS, "real", mice, "oracle")
        txt = []
        for c, cn in ((0, "sup"), (1, "bg")):
            a = (A[:, :, c].sum(-1) / M[:, :, c].sum(-1)).mean(-1)
            o = (O[:, :, c].sum(-1) / M[:, :, c].sum(-1)).mean(-1)
            txt.append(f"{cn} (oracle {100 * (o[0] - 1):+.1f}%) " + " ".join(f"{(1 - x) / (1 - y):.2f}" for x, y in zip(a, o)))
        print(f"  {run}: " + " | ".join(txt))


def combined_selection(S, runs, test, val):
    print("\n4. fine-tuning checkpoint chosen by the validation mice: real + w x fast (both balanced vs the mean image);"
          " test real vs mean image / fast vs oracle")
    T = {run: (pct(*S.sums(f"{run}ft", STEPS, "real", test, "mean_image")),
               pct(*S.sums(f"{run}ft", STEPS, "fast", test, "oracle"))) for run in runs}
    V = {run: (pct(*S.sums(f"{run}ft", STEPS, "real", val, "mean_image")),
               pct(*S.sums(f"{run}ft", STEPS, "fast", val, "mean_image"))) for run in runs}
    for w in (0.0, 0.03, 0.1, 0.3, 1.0, np.inf):
        picks = [int(np.argmin(V[r][1] if np.isinf(w) else V[r][0] + w * V[r][1])) for r in runs]
        rr = [T[r][0][i] for r, i in zip(runs, picks)]
        ff = [T[r][1][i] for r, i in zip(runs, picks)]
        print(f"  w {w:<5}: steps " + ",".join(str(STEPS[i]) for i in picks)
              + f"; real {np.mean(rr):+.2f} (" + " ".join(f"{x:+.2f}" for x in rr) + f"), fast {np.mean(ff):+.1f} ("
              + " ".join(f"{x:.1f}" for x in ff) + ")")
    for name, idx in (("step 0", [0]), ("late mean", [STEPS.index(s) for s in LATE]), ("best on test fast", None)):
        rr, ff = [], []
        for r in runs:
            j = [int(np.argmin(T[r][1]))] if idx is None else idx
            rr.append(T[r][0][j].mean())
            ff.append(T[r][1][j].mean())
        print(f"  {name:17s}: real {np.mean(rr):+.2f} (" + " ".join(f"{x:+.2f}" for x in rr) + f"), fast {np.mean(ff):+.1f} ("
              + " ".join(f"{x:.1f}" for x in ff) + ")")
    print("\n5. within-run rank correlations over fine-tuning steps 250-1700")
    for r in runs:
        print(f"  {r}: validation vs test, fast {spearmanr(V[r][1][1:], T[r][1][1:])[0]:+.2f}, real "
              f"{spearmanr(V[r][0][1:], T[r][0][1:])[0]:+.2f}; test real vs test fast "
              f"{spearmanr(T[r][0][1:], T[r][1][1:])[0]:+.2f}")


def averages(path: Path, B=2000):
    """Model variants scored on the same windows: % change vs reference per kind, and paired
    differences against the first model, 95% intervals over windows (mice fixed)."""
    rows = list(csv.DictReader(open(path)))
    for r in rows:  # evaluate's windows.csv names the model "checkpoint"
        r["model"] = r.get("model") or r["checkpoint"]
    models = list(dict.fromkeys(r["model"] for r in rows))
    rng = np.random.default_rng(0)
    for kind, ref in (("fast", "oracle"), ("real", "mean_image"), ("real", "silo")):
        n = defaultdict(int)  # rows per (window, model): keep the windows every model has exactly once
        for r in rows:
            if r["kind"] == kind:
                n[(r["session"], int(r["window"])), r["model"]] += 1
        keys = sorted({k for k, _ in n})
        keys = [k for k in keys if all(n.get((k, m)) == 1 for m in models)]
        if not keys:
            continue
        sess = sorted({k[0] for k in keys})
        idx = {k: i for i, k in enumerate(keys)}
        A = np.zeros((len(models), len(keys), 2))
        R = np.zeros((len(models), len(keys), 2))
        for r in rows:
            k = (r["session"], int(r["window"]))
            if r["kind"] != kind or k not in idx:
                continue
            j = models.index(r["model"])
            for c, cn in enumerate(("sup", "bg")):
                A[j, idx[k], c] = float(r[f"{cn}_se8_net"])
                R[j, idx[k], c] = float(r[f"{cn}_se8_{ref}"])
        smask = np.array([[k[0] == s for k in keys] for s in sess], float)  # (mice, windows)

        def score(w):
            a = np.einsum("mkc,sk->msc", A, smask * w)
            b = np.einsum("mkc,sk->msc", R, smask * w)
            r = a / b
            return 100 * (r.mean(-1).mean(-1) - 1), 100 * (r[..., 0].mean(-1) - 1), 100 * (r[..., 1].mean(-1) - 1)

        point = score(np.ones(len(keys)))
        boots = []
        for _ in range(B):
            w = np.zeros(len(keys))
            for s in range(len(sess)):
                ks = np.flatnonzero(smask[s])
                np.add.at(w, rng.choice(ks, len(ks)), 1.0)
            boots.append(score(w)[0])
        boots = np.array(boots)
        n = int(smask.sum(1).min())
        print(f"\n{kind} vs {ref} ({len(sess)} mice, {n}+ windows each): balanced, support, background; "
              f"difference to {models[0]} [95% over windows]")
        for j, m in enumerate(models):
            d = boots[:, j] - boots[:, 0]
            print(f"  {m:16s} {point[0][j]:+7.2f} {point[1][j]:+7.2f} {point[2][j]:+7.2f}   "
                  f"{point[0][j] - point[0][0]:+6.2f} [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]")


def load_named(path: Path):
    """evaluate --per_window rows of checkpoints named <run>@<variant> (e.g. S3@soupFT):
    (<run>avg, variant, mouse, kind) -> arrays over windows (in window order)."""
    acc = {}
    with open(path) as fh:
        rd = csv.reader(fh)
        num = next(rd)[4:]
        for row in rd:
            run, var = row[0].split("@")
            acc.setdefault((f"{run}avg", var, mouse(row[2]), row[3]), []).append([float(x) for x in row[4:]])
    W = {}
    for k, v in acc.items():
        a = np.array(sorted(v, key=lambda r: r[0]))  # column 0 is the window index
        W[k] = {n: a[:, i] for i, n in enumerate(num)}
    return W


def weight_averages(S, path, runs, test, val, B=1000):
    """6. weight averages of kept steps (`slap2-f0 average`), scored by `evaluate --per_window`
    (checkpoints named <run>@<variant>), against the fine-tuning steps of the same run: per run, then
    the mean over runs with 95% intervals over mice and windows. Each readout uses the mice that every
    model of every run has, and within each mouse the windows (by window index) that all of them share,
    so variants scored on fewer windows or only the test mice are still compared like for like."""
    W = load_named(path)
    S.W.update(W)
    have = {run: list(dict.fromkeys(k[1] for k in W if k[0] == f"{run}avg")) for run in runs}
    late = [STEPS.index(s) for s in LATE]
    print(f"\n6. weight averages ({path}) against the fine-tuning steps; real vs the mean image, fast vs the oracle")

    def keys(run, m, kind):
        return [(f"{run}ft", s, m, kind) for s in STEPS] + [(f"{run}avg", v, m, kind) for v in have[run]]

    def take(key, ids):  # error sums (class, windows) of the net and the references at these window indices
        d = S.W[key]
        pos = {int(x): i for i, x in enumerate(d["window"])}
        j = np.array([pos[x] for x in ids])
        return {p: np.stack([d[f"{c}_se8_{p}"][j] for c in ("sup", "bg")]) for p in PREDS}

    for run in runs:  # copies of kept steps must reproduce the held-out evaluation exactly
        for var, step in (("ft0", 0), ("ft1700", 1700)):
            dev = []
            for k in [k for k in W if k[0] == f"{run}avg" and k[1] == var]:
                o = (f"{run}ft", step, k[2], k[3])
                if o in S.W:
                    ids = sorted(set(W[k]["window"].astype(int)) & set(S.W[o]["window"].astype(int)))
                    a, b = take(k, ids)["net"], take(o, ids)["net"]
                    dev.append(np.abs(a - b).max() / np.abs(b).max())
            if dev:
                print(f"  check {run}@{var} = {run}ft step {step}: largest relative difference {max(dev):.1e}")

    readouts = []
    for name, kind, mice, ref in (("real all", "real", test + val, "mean_image"), ("real test", "real", test, "mean_image"),
                                  ("fast test", "fast", test, "oracle")):
        ms = [m for m in S.mice(kind, mice) if all(k in S.W for r in runs for k in keys(r, m, kind))]
        if not ms:
            print(f"  {name}: no mouse has every model; left out")
            continue
        ids = {m: sorted(set.intersection(*[set(S.W[k]["window"].astype(int)) for r in runs for k in keys(r, m, kind)]))
               for m in ms}
        data = {r: [[take(k, ids[m]) for k in keys(r, m, kind)] for m in ms] for r in runs}
        print(f"  {name}: mice {', '.join(ms)}; {min(len(v) for v in ids.values())}+ shared windows per mouse")
        readouts.append((name, ms, ids, data, ref))

    def scores(data, ref, w=None, mm=None):
        """% change per model (fine-tuning steps, then variants), balanced over classes, mice weighted by mm"""
        per_mouse = []
        for i, models in enumerate(data):
            ww = 1.0 if w is None else w[i]
            per_mouse.append([((d["net"] * ww).sum(-1) / (d[ref] * ww).sum(-1)).mean() for d in models])
        r = np.array(per_mouse)  # (mice, models)
        mm = np.ones(len(r)) if mm is None else mm
        return 100 * ((r * mm[:, None]).sum(0) / mm.sum() - 1)

    def summary(s):  # step 0, late mean, step 1700, variants
        return np.concatenate([[s[0], s[late].mean(), s[len(STEPS) - 1]], s[len(STEPS):]])

    for run in runs:
        rows = ["step 0", "late mean", "step 1700"] + have[run]
        t = np.array([summary(scores(data[run], ref)) for _, _, _, data, ref in readouts])
        print(f"\n  {run:16s}" + " | ".join(f"{name:>11s}" for name, *_ in readouts))
        for j, r in enumerate(rows):
            print(f"  {r:16s}" + " | ".join(f"{t[i, j]:+11.2f}" for i in range(len(readouts))))
    common = ["step 0", "late mean", "step 1700"] + [v for v in have[runs[0]] if all(v in have[r] for r in runs)]

    def mean_over_runs(rd, w=None, mm=None):
        _, _, _, data, ref = rd
        out = []
        for r in runs:
            s = summary(scores(data[r], ref, w, mm))
            names = ["step 0", "late mean", "step 1700"] + have[r]
            out.append(s[[names.index(c) for c in common]])
        return np.mean(out, 0)

    rng = np.random.default_rng(0)
    print(f"\n  mean of {', '.join(runs)}: score; difference to step 0 [95% over mice and windows]; difference to "
          "the late mean")
    for rd in readouts:
        _, ms, ids, _, _ = rd
        pt = mean_over_runs(rd)
        boots = []
        for _ in range(B):  # one draw of this readout's mice and windows, shared by every run and model
            mm = np.bincount(rng.integers(0, len(ms), len(ms)), minlength=len(ms)).astype(float)
            w = [np.bincount(rng.integers(0, len(ids[m]), len(ids[m])), minlength=len(ids[m])).astype(float) for m in ms]
            boots.append(mean_over_runs(rd, w, mm))
        boots = np.array(boots)
        print(f"  {rd[0]}")
        for j, c in enumerate(common):
            d0 = boots[:, j] - boots[:, 0]
            print(f"    {c:16s} {pt[j]:+7.2f}  {pt[j] - pt[0]:+6.2f} [{np.percentile(d0, 2.5):+6.2f}, "
                  f"{np.percentile(d0, 97.5):+6.2f}]  {pt[j] - pt[1]:+6.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="?")
    p.add_argument("--runs", default="S0,S1,S2,S3")
    p.add_argument("--test", default=",".join(TEST))
    p.add_argument("--val", default=",".join(VAL))
    p.add_argument("--drop_sim", default="804730")
    p.add_argument("--averages", default="", help="per-window table of model variants (see the module docstring)")
    a = p.parse_args()
    if a.averages and not a.results:
        averages(Path(a.averages))
        return
    root = Path(a.results)
    W, E = {}, {}
    for d in sorted(q for q in root.iterdir() if (q / "windows.csv").exists()):
        w, e = load(d)
        W.update(w)
        E.update(e)
    S = Scores(W, E, a.drop_sim.split(",") if a.drop_sim else [])
    runs, test, val = a.runs.split(","), tuple(a.test.split(",")), tuple(a.val.split(","))
    if a.averages:  # with RESULTS_DIR: the averages against the run's fine-tuning steps only
        weight_averages(S, Path(a.averages), runs, test, val)
        return
    trajectories(S, runs, test)
    pretraining(S, runs, test)
    real_gain(S, runs, test + val)
    combined_selection(S, runs, test, val)


if __name__ == "__main__":
    main()
