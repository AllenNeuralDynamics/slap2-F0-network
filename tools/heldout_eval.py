"""Experiments 1 and 2 from `slap2-f0 evaluate --per_window` results (docs/plan.md).

usage: python heldout_eval.py exp1|exp2 RESULTS_DIR [--pick S0ft=1500,S1ft=1700,...]
                              [--val 878030,...] [--test 776270,...] [--drop_sim 804730]

RESULTS_DIR holds one folder per evaluation run, each with eval.csv and
windows.csv. Checkpoints are named <run>@step_<N> (`evaluate --ckpt run=.../step_*.pt`);
fine-tuning runs end in "ft", pretraining runs in "pt".

Scores are ratios of summed 8-frame squared errors, net / reference, per mouse
and pixel class (support, background); "balanced" is the mean of the two
classes, then the mean over mice. References: the mean image, SiLo, and the
oracle (the window's true mean baseline, a static baseline; simulations only).

exp1: the picked fine-tuning checkpoints (--pick: the steps training's
validation chose) and the mean of the checkpoints at steps >= 750 ("late") on
the test mice, with 95% intervals from a hierarchical bootstrap (mice, then
windows within mice) or over windows alone; paired differences between runs.
exp2: selection regret (test score of the checkpoint the validation mice pick
minus the best checkpoint's) with 1-4 validation mice and 48 or 192 windows per
mouse, averaged over every subset of mice and, for 48 windows, over the four
disjoint 48-window blocks (block 0 = the windows training's validation used,
with the training seed); rank agreement; and how many mice keep the sign of
the runs' differences.

--drop_sim leaves mice out of the simulated kinds' scores (804730: one window
per kind breaks every checkpoint, see docs/plan.md).
"""
import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PREDS = ("net", "mean_image", "silo", "oracle")
CLASSES = ("sup", "bg")
NW = 192


def mouse(s):
    return s.split("_")[0]


def fnum(x):
    try:
        return float(x)
    except ValueError:
        return x


def ckpt_parts(name):
    run, step = name.split("@")
    return run, int(step.split("_")[1])


def load(root: Path):
    """eval.csv rows of every run, and dict (checkpoint, mouse, kind) -> arrays over windows."""
    rows, W = [], {}
    for d in sorted(p for p in root.iterdir() if (p / "windows.csv").exists()):
        with open(d / "eval.csv") as fh:
            rows += [{k: fnum(v) for k, v in r.items()} for r in csv.DictReader(fh)]
        with open(d / "windows.csv") as fh:
            rd = csv.reader(fh)
            num = next(rd)[4:]
            acc = {}
            for row in rd:
                acc.setdefault((row[0], mouse(row[2]), row[3]), []).append([float(x) for x in row[4:]])
            for key, v in acc.items():
                a = np.array(v)
                W[key] = {k: a[:, i] for i, k in enumerate(num)}
    return rows, W


def build(W, mice):
    """E[kind][pred]: (checkpoint, mouse, class, window) error sums; E[kind]["ck"]: checkpoint names."""
    cks = sorted({k[0] for k in W}, key=ckpt_parts)
    E = {}
    for kind in sorted({k[2] for k in W}):
        ck_k = [c for c in cks if (c, mice[0], kind) in W]
        E[kind] = {"ck": ck_k}
        for p in PREDS + ("n8",):
            a = np.zeros((len(ck_k), len(mice), 2, NW))
            for i, c in enumerate(ck_k):
                for j, m in enumerate(mice):
                    d = W[(c, m, kind)]
                    for q, cl in enumerate(CLASSES):
                        a[i, j, q] = d[f"{cl}_n8"] if p == "n8" else d[f"{cl}_se8_{p}"]
            E[kind][p] = a
    return E


def ratios(E, kind, num, den, wts, sel=None, mice=None):
    """wts (B, mouse, window) window weights for the mice `mice` -> (B, ck, mouse, class)
    ratio of summed errors, for checkpoints `sel` (indices into E[kind]["ck"])."""
    sel = slice(None) if sel is None else sel
    mice = slice(None) if mice is None else mice
    A, D = E[kind][num][sel][:, mice], E[kind][den][sel][:, mice]
    a = np.einsum("bmw,cmkw->bcmk", wts, A)
    b = np.einsum("bmw,cmkw->bcmk", wts, D)
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / b


def boot_weights(B, nm, nw, rng, windows_only=False):
    """Window weights (B, nm, nw), windows resampled within each mouse, and mouse
    multiplicities (B, nm) (all ones if windows_only)."""
    ww = np.zeros((B, nm, nw))
    for m in range(nm):
        idx = rng.integers(0, nw, (B, nw))
        for b in range(B):
            ww[b, m] += np.bincount(idx[b], minlength=nw)
    mm = np.ones((B, nm)) if windows_only else np.stack(
        [np.bincount(rng.integers(0, nm, nm), minlength=nm) for _ in range(B)])
    return ww, mm


def exp1(rows, E, mice, test, val, pick, B=4000):
    rng = np.random.default_rng(1)
    ti = [mice.index(m) for m in test]
    ww, mm = boot_weights(B, len(ti), NW, rng)
    wwo, mmo = boot_weights(B, len(ti), NW, rng, windows_only=True)
    runs = list(pick)

    def score(kind, den, cks, mice_idx=ti, ww=ww, mm=mm, combine="balanced"):
        """(B+1, len(cks)); row 0 is the point estimate."""
        ck = E[kind]["ck"]
        sel = [ck.index(c) for c in cks]
        w = np.concatenate([np.ones((1,) + ww.shape[1:]), ww])
        m = np.concatenate([np.ones((1, mm.shape[1])), mm])
        r = ratios(E, kind, "net", den, w, sel, mice_idx)
        pm = {"balanced": np.nanmean(r, 3), "sup": r[..., 0], "bg": r[..., 1]}[combine]
        mw = m[:, None, :]
        return (pm * mw).sum(2) / mw.sum(2)

    def ci(x):
        return np.percentile(x[1:], [2.5, 97.5])

    def pct(s, i):
        lo, hi = ci(s[:, i])
        return f"{100 * (s[0, i] - 1):+6.1f} [{100 * (lo - 1):+5.1f},{100 * (hi - 1):+5.1f}]"

    picked = [f"{r}@step_{s:06d}" for r, s in pick.items()]
    late = {r: [c for c in E["real"]["ck"] if c.startswith(r + "@") and ckpt_parts(c)[1] >= 750] for r in runs}

    print(f"== Test mice ({len(test)} x {NW} windows): % change of the 8-frame MSE; 95% interval over mice and windows ==")
    print("   'picked' = the checkpoint training's validation chose; 'late' = mean of the checkpoints at steps >= 750")
    for kind, den in (("real", "mean_image"), ("real", "silo"), ("sim", "oracle"), ("fast", "oracle"), ("free", "mean_image")):
        for comb in ("sup", "bg", "balanced"):
            s = score(kind, den, picked, combine=comb)
            print(f"  {kind:4s} vs {den:10s} {comb:8s} picked " + "  ".join(f"{r[:2]} {pct(s, i)}" for i, r in enumerate(runs)))
        sl = np.stack([score(kind, den, late[r]).mean(1) for r in runs], 1)
        print(f"  {kind:4s} vs {den:10s} {'balanced':8s} late   " + "  ".join(f"{r[:2]} {pct(sl, i)}" for i, r in enumerate(runs)))

    print("\n== Paired differences of the balanced score (percentage points of the reference's MSE), test mice ==")
    print("   A: resample mice and windows; B: windows only; negative = first run better")
    pairs = [(a, b) for a, b in (("S1ft", "S0ft"), ("S2ft", "S0ft"), ("S3ft", "S0ft"), ("S2ft", "S1ft"), ("S3ft", "S2ft"))
             if a in runs and b in runs]
    for kind, den in (("real", "mean_image"), ("sim", "oracle"), ("fast", "oracle"), ("free", "mean_image")):
        for lab, get in (("picked", lambda w, m: score(kind, den, picked, ww=w, mm=m)),
                         ("late", lambda w, m: np.stack([score(kind, den, late[r], ww=w, mm=m).mean(1) for r in runs], 1))):
            sA, sB = get(ww, mm), get(wwo, mmo)
            out = []
            for a, b in pairs:
                i, j = runs.index(a), runs.index(b)
                dA, dB = 100 * (sA[:, i] - sA[:, j]), 100 * (sB[:, i] - sB[:, j])
                out.append(f"{a[:2]}-{b[:2]} {dA[0]:+5.1f} A[{ci(dA)[0]:+5.1f},{ci(dA)[1]:+5.1f}] B[{ci(dB)[0]:+5.1f},{ci(dB)[1]:+5.1f}]")
            print(f"  {kind:4s} vs {den:10s} {lab:6s}: " + "; ".join(out))

    print("\n== Per mouse, balanced, picked checkpoints (% change) ==")
    for kind, den in (("real", "mean_image"), ("fast", "oracle")):
        print(f"  {kind} vs {den}")
        for m in test + val:
            j = mice.index(m)
            s = score(kind, den, picked, [j], ww[:1, :1], mm[:1, :1])[0]
            print(f"    {'test' if m in test else 'val '} {m}: " + "  ".join(f"{r[:2]} {100 * (v - 1):+6.1f}" for r, v in zip(runs, s)))

    print("\n== Every fine-tuning checkpoint, test mice, balanced (% change) ==")
    for kind, den in (("real", "mean_image"), ("sim", "oracle"), ("fast", "oracle"), ("free", "mean_image")):
        print(f"  {kind} vs {den}")
        for run in runs:
            cks = [c for c in E[kind]["ck"] if c.startswith(run + "@")]
            s = score(kind, den, cks, ww=ww[:1], mm=mm[:1])[0]
            lt = [k for k, c in enumerate(cks) if ckpt_parts(c)[1] >= 750]
            print(f"    {run}: " + " ".join(f"{ckpt_parts(c)[1]}:{100 * (v - 1):+.1f}" for c, v in zip(cks, s))
                  + f" | >=750 mean {100 * (s[lt].mean() - 1):+.1f} sd {100 * s[lt].std(ddof=1):.1f}")

    idx = {(r["checkpoint"], mouse(r["session"]), r["kind"]): r for r in rows}
    print("\n== Activity-free windows, labeled pixels outside the support: MSE ratio to the mean image (= true mean "
          "baseline) / bias (target - net, scaled units) ==")
    for c in picked:
        print(f"  {c}: " + "  ".join(
            f"{m} {idx[(c, m, 'free')]['lab_mse8_net'] / idx[(c, m, 'free')]['lab_mse8_mean_image']:.2f}"
            f"/{idx[(c, m, 'free')]['lab_bias_net']:+.2f}" for m in test + val))
    print("  leak on simulated support (regression of the baseline's error on the activity; 0 = none, 1 = follows it):")
    for kind in ("sim", "fast"):
        for c in picked:
            print(f"    {kind} {c}: test " + " ".join(f"{idx[(c, m, kind)]['sup_leak_net']:+.3f}" for m in test)
                  + " | val " + " ".join(f"{idx[(c, m, kind)]['sup_leak_net']:+.3f}" for m in val)
                  + f" | mean image test {np.mean([idx[(c, m, kind)]['sup_leak_mean_image'] for m in test]):+.3f}")


def exp2(rows, E, mice, test, val, drop_sim):
    ti = [mice.index(m) for m in test]

    def per_mouse(kind, den, cks, wmask):
        """(ck, mouse) balanced ratio using the windows in wmask (NW,)."""
        ck = E[kind]["ck"]
        sel = [ck.index(c) for c in cks]
        w = np.broadcast_to(wmask, (1, len(mice), NW)).astype(float)
        return np.nanmean(ratios(E, kind, "net", den, w, sel)[0], 2)

    def blocks(nw):
        return [np.ones(NW, bool)] if nw == NW else [np.arange(NW) // 48 == b for b in range(NW // 48)]

    def study(label, kind, pools, val_mice, test_kinds):
        vi = [mice.index(m) for m in val_mice]
        print(f"\n== {label}: selection on {kind} windows (balanced vs mean image), validation mice {', '.join(val_mice)} ==")
        tests = {}
        for tk, td in test_kinds:
            allc = sorted({c for p in pools.values() for c in p})
            tests[(tk, td)] = dict(zip(allc, per_mouse(tk, td, allc, np.ones(NW, bool))[:, ti].mean(1)))
        for pname, cks in pools.items():
            vals = {nw: [per_mouse(kind, "mean_image", cks, b) for b in blocks(nw)] for nw in (48, NW)}
            print(f"  pool {pname} ({len(cks)} checkpoints)")
            for (tk, td), T in tests.items():
                t = np.array([T[c] for c in cks])
                best, rnd, worst = t.min(), t.mean(), t.max()
                print(f"    test {tk} vs {td}: best {100 * (best - 1):+.2f}%, random pick regret {100 * (rnd - best):.2f} pts, "
                      f"worst {100 * (worst - best):.2f}; regret (pts) / share of random's:")
                for k in range(1, len(vi) + 1):
                    cells = []
                    for nw in (48, NW):
                        regs = np.array([t[np.nanargmin(V[:, [vi[j] for j in sub]].mean(1))] - best
                                         for sub in combinations(range(len(vi)), k) for V in vals[nw]])
                        cells.append(f"{nw:3d} win {100 * regs.mean():5.2f} ({regs.mean() / max(rnd - best, 1e-12):4.0%})")
                    print(f"      {k} mouse{'s' if k > 1 else ' '}: " + "   ".join(cells))
            if len(vi) >= 3:  # what training did: first 48 windows of the first three validation mice
                c = cks[np.nanargmin(vals[48][0][:, vi[:3]].mean(1))]
                print(f"    training-like pick (3 mice, block 0): {c}; test " + ", ".join(
                    f"{tk} vs {td} {100 * (T[c] - 1):+.2f}%" for (tk, td), T in tests.items()))
            v192 = vals[NW][0][:, vi].mean(1)
            for (tk, td), T in tests.items():
                print(f"    Spearman(validation, test {tk} vs {td}) over checkpoints: "
                      f"{spearmanr(v192, np.array([T[c] for c in cks])).statistic:+.2f}")

    runs = sorted({ckpt_parts(c)[0][:-2] for c in E["real"]["ck"]})
    ft = {r: [c for c in E["real"]["ck"] if c.startswith(f"{r}ft@")] for r in runs}
    pt = {r: [c for c in E["fast"]["ck"] if c.startswith(f"{r}pt@")] for r in runs}
    pt = {r: v for r, v in pt.items() if v}
    vsim = tuple(m for m in val if m not in drop_sim)
    for r in runs:
        study(f"Fine-tuning, run {r}", "real", {r: ft[r]}, val, [("real", "mean_image"), ("fast", "oracle")])
    study(f"Fine-tuning, all {sum(map(len, ft.values()))} checkpoints", "real", {"all": sum(ft.values(), [])}, val,
          [("real", "mean_image"), ("fast", "oracle")])
    if pt:
        for r in pt:
            study(f"Pretraining, run {r}", "fast", {r: pt[r]}, vsim, [("fast", "mean_image"), ("fast", "oracle")])
        study(f"Pretraining, all {sum(map(len, pt.values()))} checkpoints", "fast", {"all": sum(pt.values(), [])}, vsim,
              [("fast", "mean_image"), ("fast", "oracle")])

    idx = {(r["checkpoint"], mouse(r["session"]), r["kind"]): r for r in rows}

    def bfree(c, m):  # training's fine-tuning score; labeled pixels are only in eval.csv (all windows)
        r, f = idx[(c, m, "real")], idx[(c, m, "free")]
        return np.nanmean([r["sup_mse8_net"] / r["sup_mse8_mean_image"], r["bg_mse8_net"] / r["bg_mse8_mean_image"],
                           f["lab_mse8_net"] / f["lab_mse8_mean_image"]])

    print(f"\n== Fine-tuning selected with balanced_free ({NW} windows); test readout real balanced vs mean image ==")
    allft = sum(ft.values(), [])
    treal = dict(zip(allft, per_mouse("real", "mean_image", allft, np.ones(NW, bool))[:, ti].mean(1)))
    for r in runs:
        cks = ft[r]
        t = np.array([treal[c] for c in cks])
        for k in range(1, len(val) + 1):
            regs = [t[np.argmin([np.mean([bfree(c, val[j]) for j in sub]) for c in cks])] - t.min()
                    for sub in combinations(range(len(val)), k)]
            print(f"  {r} {k} mice: regret {100 * np.mean(regs):.2f} pts (random {100 * (t.mean() - t.min()):.2f})")
        tb = np.array([np.mean([bfree(c, m) for m in test]) for c in cks])
        vb = np.array([np.mean([bfree(c, m) for m in val]) for c in cks])
        print(f"  {r} Spearman(val balanced_free, test balanced_free) {spearmanr(vb, tb).statistic:+.2f}; "
              f"(val balanced_free, test real) {spearmanr(vb, t).statistic:+.2f}")

    print("\n== Resolution: differences between runs (late-checkpoint means) over subsets of k held-out mice ==")
    late = {r: [c for c in ft[r] if ckpt_parts(c)[1] >= 750] for r in runs}
    pairs = [(a, b) for a, b in (("S3", "S0"), ("S2", "S1"), ("S1", "S0"), ("S3", "S2")) if a in runs and b in runs]
    for kind, den, ms in (("real", "mean_image", mice), ("fast", "oracle", tuple(m for m in mice if m not in drop_sim))):
        mi_ = [mice.index(m) for m in ms]
        pm = {r: per_mouse(kind, den, late[r], np.ones(NW, bool)).mean(0) for r in runs}
        d_all = {p: np.mean([pm[p[0]][j] - pm[p[1]][j] for j in mi_]) for p in pairs}
        print(f"  {kind} vs {den} ({len(ms)} mice): all-mice differences (pts) "
              + ", ".join(f"{a}-{b} {100 * v:+.2f}" for (a, b), v in d_all.items()))
        for p, dv in d_all.items():
            out = []
            for k in (1, 2, 3, 4, 6):
                if k > len(mi_):
                    continue
                ds = np.array([np.mean([pm[p[0]][mi_[j]] - pm[p[1]][mi_[j]] for j in sub])
                               for sub in combinations(range(len(mi_)), k)])
                out.append(f"k={k}: same sign {np.mean(np.sign(ds) == np.sign(dv)):.0%}, sd {100 * ds.std():.2f}")
            print(f"    {p[0]}-{p[1]}: " + "; ".join(out))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("what", choices=["exp1", "exp2"])
    p.add_argument("root", type=Path)
    p.add_argument("--pick", default="S0ft=1500,S1ft=1700,S2ft=1700,S3ft=1700",
                   help="run=step of the fine-tuning checkpoint each run's training picked")
    p.add_argument("--val", default="878030,794237,834788,804730", help="validation mice (training's first)")
    p.add_argument("--test", default="776270,810196,829704,876767")
    p.add_argument("--drop_sim", default="804730", help="mice left out of the simulated kinds' scores in exp2")
    a = p.parse_args()
    val, test = tuple(a.val.split(",")), tuple(a.test.split(","))
    mice = val + test
    rows, W = load(a.root)
    E = build(W, mice)
    if a.what == "exp1":
        exp1(rows, E, mice, test, val, {k: int(v) for k, v in (x.split("=") for x in a.pick.split(","))})
    else:
        exp2(rows, E, mice, test, val, tuple(x for x in a.drop_sim.split(",") if x))


if __name__ == "__main__":
    main()
