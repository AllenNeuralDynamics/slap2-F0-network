"""How much detection SNR do SiLo's spatial and temporal filters keep, on real data?

For isolated GIAnT sources in exported trials (movie, activity, mean image), the
event signal at each pixel is estimated without GIAnT's spatial model: the
regression of (movie - baseline) on the source's own activity trace, D_p.
Frames are split into alternating 200-frame blocks (two folds); weights are
built on one fold and scored on the other:

    SNR(w) = (w . D_test)^2 / var_t(w . n_test)      (per unit trace amplitude)

n = movie - activity - baseline (real noise, including correlated noise and
GIAnT's misfit). The spatial schemes:

    silo      DoG(1.33 px, 5x) applied to z = (y - b)/sigma      (SiLo today)
    z_gauss   Gaussian(1.33) on z                                  (no surround)
    a0        Gaussian on y - b (weights G)        optimal if dF/F is uniform over the PSF
    a1        Gaussian on (y - b)/sigma^2           optimal if dF is uniform over the PSF
    fp_pois   D_train / sigma^2   (the source's own footprint, Poisson variance)
    fp_var    D_train / var_p     (own footprint, measured variance)
    fp_white  C_train^-1 D_train  (own footprint, full noise covariance, radius 4)

Temporal: the event kernel k is the event-triggered average of the
fp_pois-projected trace at isolated onsets of the source's activity trace,
the noise autocorrelation that of the projected noise. Filters: SiLo's
anticausal exponential (tau 24 ms), the plain matched filter k, and the
whitened matched filter C^-1 k.

usage: python -I realdata_eff.py <trial.h5> [<trial.h5> ...]
"""

import sys

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d

R_PATCH = 10      # half-width of the patch for the template schemes
R_WHITE = 4       # radius of the disk for the whitened oracle
ISO = 8           # no other source peak within this many px
BLOCK = 200
SIG = 1.33
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def nan_movmean(x, win):
    v = np.isfinite(x)
    s = uniform_filter1d(np.where(v, x, 0).astype(np.float64), win, axis=0, mode="nearest")
    c = uniform_filter1d(v.astype(np.float64), win, axis=0, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s / c).astype(np.float32)


def kernels():
    n = 2 * R_PATCH + 1
    d = np.zeros((n, n)); d[R_PATCH, R_PATCH] = 1
    g = gaussian_filter(d, SIG, mode="constant", truncate=8)
    big = np.zeros((4 * n, 4 * n)); big[2 * n, 2 * n] = 1
    g_big = gaussian_filter(big, SIG, mode="constant", truncate=8)
    dog_big = g_big - gaussian_filter(g_big, 5 * SIG, mode="constant", truncate=8)
    dog = dog_big[2 * n - R_PATCH:2 * n + R_PATCH + 1, 2 * n - R_PATCH:2 * n + R_PATCH + 1]
    return g, dog


def main(files):
    g, dog = kernels()
    yy, xx = np.mgrid[-R_PATCH:R_PATCH + 1, -R_PATCH:R_PATCH + 1]
    disk = (yy ** 2 + xx ** 2) <= R_WHITE ** 2
    rows = []
    ktr, ctr = [], []
    for fn in files:
        with h5py.File(fn, "r") as f:
            M = f["movie"][0]
            A = f["activity"][0]
            pr = f["sources/peak_row"][:] - 1
            pc = f["sources/peak_col"][:] - 1
            valid = f["mask/valid"][:].astype(bool)
            labeled = f["mask/labeled"][:].astype(bool)
        T, H, W = M.shape
        R = M - A
        b = nan_movmean(R, 321)
        n = R - b
        x = M - b
        bg = valid & ~labeled
        bbar = np.nanmean(b, axis=0)
        vres = np.nanvar(n, axis=0)
        sel = bg & (bbar > 0.5) & np.isfinite(vres)
        gain = np.median(vres[sel] / bbar[sel])
        s2 = gain * np.maximum(bbar, 0.05 * gain)          # Poisson variance in movie units
        fold = (np.arange(T) // BLOCK) % 2
        # stationary noise correlation rho(dy, dx) on background pixels (n / sd), offsets within the white disk
        sd = np.sqrt(np.where(np.isfinite(vres) & (vres > 0), vres, np.nan))
        zz = n[::4] / sd
        rho = np.zeros((2 * R_WHITE + 1, 2 * R_WHITE + 1))
        for dy in range(-R_WHITE, R_WHITE + 1):
            for dx in range(-R_WHITE, R_WHITE + 1):
                a1 = zz[:, max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)]
                a2 = zz[:, max(0, -dy):H + min(0, -dy), max(0, -dx):W + min(0, -dx)]
                m1 = bg[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] & bg[max(0, -dy):H + min(0, -dy), max(0, -dx):W + min(0, -dx)]
                pr_ = a1[:, m1] * a2[:, m1]
                rho[dy + R_WHITE, dx + R_WHITE] = np.nanmean(pr_)
        rho /= rho[R_WHITE, R_WHITE]
        if fn == files[0]:
            print("stationary noise correlation (rows dy=-2..2, cols dx=-2..2):", np.round(rho[R_WHITE-2:R_WHITE+3, R_WHITE-2:R_WHITE+3], 3), flush=True)
        F0 = np.nanmean(b, axis=0)
        print(f"{fn}: T={T} gain={gain:.2f} sources={len(pr)} "
              f"baseline photons/frame median {np.nanmedian(bbar[valid])/gain:.2f}", flush=True)
        for j in range(len(pr)):
            r, c = pr[j], pc[j]
            dmin = np.min(np.hypot(pr[np.arange(len(pr)) != j] - r, pc[np.arange(len(pr)) != j] - c))
            if dmin < ISO or r < R_PATCH or c < R_PATCH or r >= H - R_PATCH or c >= W - R_PATCH:
                continue
            sl = (slice(r - R_PATCH, r + R_PATCH + 1), slice(c - R_PATCH, c + R_PATCH + 1))
            if valid[sl].mean() < 0.9:
                continue
            a = A[:, r, c].astype(np.float64)
            xp = x[:, sl[0], sl[1]].reshape(T, -1).astype(np.float64)
            npch = n[:, sl[0], sl[1]].reshape(T, -1).astype(np.float64)
            ok = np.isfinite(a) & np.all(np.isfinite(xp), 1) & np.all(np.isfinite(npch), 1)
            if ok.sum() < 1000 or np.nanstd(a) == 0:
                continue
            s2p = s2[sl].ravel(); vp = vres[sl].ravel()
            vp = np.where(np.isfinite(vp), vp, s2p)
            for k in (0, 1):
                tr = ok & (fold == k); te = ok & (fold == 1 - k)
                at, ae = a[tr], a[te]
                if ae.std() == 0 or at.std() == 0:
                    continue
                Dtr = (at - at.mean()) @ (xp[tr] - xp[tr].mean(0)) / ((at - at.mean()) @ (at - at.mean()))
                Dte = (ae - ae.mean()) @ (xp[te] - xp[te].mean(0)) / ((ae - ae.mean()) @ (ae - ae.mean()))
                ntr, nte = npch[tr], npch[te]
                W_ = {
                    "silo": dog.ravel() / np.sqrt(s2p),
                    "z_gauss": g.ravel() / np.sqrt(s2p),
                    "fp_pois": Dtr / s2p,
                    "fp_var": Dtr / vp,
                }
                for al in ALPHAS:
                    W_[f"a{al:g}"] = g.ravel() / s2p ** al
                m = disk.ravel()
                Ctr = np.cov(ntr[:, m], rowvar=False)
                sh = 0.1
                Ctr = (1 - sh) * Ctr + sh * np.diag(np.diag(Ctr))
                wwh = np.zeros(m.size); wwh[m] = np.linalg.solve(Ctr, Dtr[m])
                W_["fp_white"] = wwh
                # achievable at seeding: stationary correlation x local SDs, isotropic or anatomy template
                idx = np.argwhere(disk)
                dyx = idx[:, None, :] - idx[None, :, :]
                ok_ = (np.abs(dyx) <= 2 * R_WHITE).all(-1)
                Rm = np.zeros((len(idx), len(idx)))
                inn = (np.abs(dyx) <= R_WHITE).all(-1)
                Rm[inn] = rho[dyx[inn][:, 0] + R_WHITE, dyx[inn][:, 1] + R_WHITE]
                sdp = np.sqrt(vp[m])
                Cs = Rm * sdp[:, None] * sdp[None, :] + 1e-6 * np.diag(vp[m])
                Ctr_l = np.cov(ntr[:, m], rowvar=False); Ctr_l = 0.9 * Ctr_l + 0.1 * np.diag(np.diag(Ctr_l))
                f0p = np.clip(F0[sl].ravel(), 0, None)
                for nm, tmpl, CC in (("g_white_loc", g.ravel()[m], Ctr_l), ("g_white", g.ravel()[m], Cs),
                                     ("anat_white", (g.ravel() * f0p)[m], Cs), ("fp_white_st", Dtr[m], Cs)):
                    ww = np.zeros(m.size); ww[m] = np.linalg.solve(CC, tmpl)
                    W_[nm] = ww
                W_["anat"] = g.ravel() * f0p / vp
                row = {"file": fn, "src": j, "fold": k, "bpk": bbar[r, c] / gain,
                       "act": np.nanstd(a) / np.sqrt(s2[r, c])}
                for name, w in W_.items():
                    sig = w @ Dte
                    noi = np.var(nte @ w)
                    row[name] = sig * abs(sig) / noi
                rows.append(row)
                # temporal: projected trace with fp_pois weights built on the training fold
                w = W_["fp_pois"]
                proj = (xp @ w); projn = (npch @ w)
                da = np.diff(a, prepend=a[0])
                thr = np.percentile(da[ok], 99)
                on = np.where((da > thr) & te)[0]
                on = [t for t in on if 10 <= t < T - 20 and np.all(da[t - 10:t] < thr / 2) and np.all(da[t + 1:t + 20] < thr)]
                if len(on) >= 3:
                    seg = np.array([proj[t - 5:t + 20] for t in on if np.all(np.isfinite(proj[t - 5:t + 20]))])
                    if len(seg) >= 3:
                        ktr.append(seg.mean(0) / seg.mean(0)[5:12].max())
                pn = projn[te]
                pn = pn - pn.mean()
                ac = np.array([np.mean(pn[:len(pn) - L] * pn[L:]) for L in range(25)])
                ctr.append(ac / ac[0])
    return rows, np.array(ktr), np.array(ctr)


def report(rows, ktr, ctr):
    keys = ["silo", "z_gauss"] + [f"a{a:g}" for a in ALPHAS] + ["anat", "g_white", "anat_white", "g_white_loc", "fp_pois", "fp_var", "fp_white_st", "fp_white"]
    S = np.array([[r[k] for k in keys] for r in rows])
    bpk = np.array([r["bpk"] for r in rows])
    good = S[:, 0] > 0
    print(f"\n{len(rows)} source-folds ({good.sum()} with positive SiLo SNR)")
    print("detection d' relative to SiLo (sqrt of the SNR ratio), median and IQR over source-folds;")
    print("pooled = ratio of summed SNRs")
    for i, k in enumerate(keys):
        rr = np.sqrt(np.clip(S[good, i], 0, None) / S[good, 0])
        q = np.percentile(rr, [25, 50, 75])
        print(f"  {k:9s} {q[1]:.3f} [{q[0]:.3f} {q[2]:.3f}]  pooled {np.sqrt(S[good, i].sum() / S[good, 0].sum()):.3f}")
    for lo, hi in [(0, 1), (1, 5), (5, 1e9)]:
        m = good & (bpk >= lo) & (bpk < hi)
        if m.sum() < 5:
            continue
        print(f"  peak baseline {lo}-{hi} photons/frame (n={m.sum()}): " + ", ".join(
            f"{k} {np.median(np.sqrt(np.clip(S[m, i], 0, None) / S[m, 0])):.3f}" for i, k in enumerate(keys) if k in ("z_gauss", "a0.5", "anat", "g_white", "anat_white", "fp_pois", "fp_white")))
    if len(ktr):
        k = np.median(ktr, 0)
        ac = np.median(ctr, 0)
        print("\nevent kernel (frames -5..19, median over sources):", np.round(k, 2))
        print("projected-noise autocorrelation lags 0..8:", np.round(ac[:9], 3))
        kk = k[5:]                         # onset at index 0
        L = len(kk)
        C = np.array([[ac[abs(i - j)] if abs(i - j) < len(ac) else 0 for j in range(L)] for i in range(L)])
        gam = np.exp(-1 / (0.024 * 80))
        h_exp = (1 - gam) * gam ** np.arange(L)

        def snr(h, shift_max=4):
            best = -np.inf
            for s in range(-shift_max, shift_max + 1):
                hs = np.roll(h, s)
                if s > 0:
                    hs[:s] = 0
                elif s < 0:
                    hs[s:] = 0
                best = max(best, (hs @ kk) ** 2 / (hs @ C @ hs))
            return best

        kt = kk.copy(); kt[7:] = 0
        optt = kt @ np.linalg.solve(C, kt)
        def snr_k(h, kv):
            best = -np.inf
            for s_ in range(-4, 5):
                hs = np.roll(h, s_)
                if s_ > 0: hs[:s_] = 0
                elif s_ < 0: hs[s_:] = 0
                best = max(best, (hs @ kv) ** 2 / (hs @ C @ hs))
            return best
        print(f"kernel truncated after +6 frames: exp 24 ms d' ratio {np.sqrt(snr_k(h_exp, kt)/optt):.3f}, plain MF {np.sqrt(snr_k(kt, kt)/optt):.3f}")
        opt = kk @ np.linalg.solve(C, kk)
        print(f"temporal SNR relative to the whitened matched filter: exp tau 24 ms {snr(h_exp)/opt:.3f}, "
              f"plain matched filter {snr(kk)/opt:.3f} (d' ratios {np.sqrt(snr(h_exp)/opt):.3f}, {np.sqrt(snr(kk)/opt):.3f})")
        for tau in (0.012, 0.036, 0.048):
            gg = np.exp(-1 / (tau * 80))
            print(f"   exp tau {tau*1000:.0f} ms: d' ratio {np.sqrt(snr((1 - gg) * gg ** np.arange(L))/opt):.3f}")


if __name__ == "__main__":
    rows, ktr, ctr = main(sys.argv[1:])
    report(rows, ktr, ctr)
