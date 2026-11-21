"""Null calibration of SiLo's detection statistic under Poisson noise.

Pure photon noise (no events), baseline known exactly (what an ideal F0
network gives), baseline varying 1000x across columns (0.03-30 photons per
pixel per frame). Two noise models for the Z-score:

    plain     sigma^2 = b                      (Poisson, no floor)
    silo      sigma^2 = b + 1.67 * b10         (SiLo's Vb floor: VIF x the 10th
                                                percentile; here of a field
                                                whose 10th-percentile pixel has
                                                the 780125 value, 0.22 photons)

Each goes through SiLo's filters: anticausal exponential (tau 24 ms at 80 Hz),
DoG (1.33 px, 5x), normalised by the white-noise SD of the filter. Reported
per brightness bin: the frame-level exceedance P(stat > t) vs the Gaussian
value, the mirror P(stat < -t) (what a sign-flip null sees), and the same
after a Cornish-Fisher skew correction computed from convolutions of b.
Then SiLo's activity image (squares of spatio-temporal local maxima, averaged)
and its local-median-subtracted value per brightness bin.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter
from scipy.signal import fftconvolve
from scipy.stats import norm

rng = np.random.default_rng(0)
H, W, T = 96, 400, 2000
SIG, TAU = 1.33, 0.024 * 80
gam = np.exp(-1 / TAU)
logb = np.linspace(np.log(0.003), np.log(300), W)
b = np.exp(logb)[None, :].repeat(H, 0).astype(np.float32)
b10 = 0.22
TH = [3, 4, 5, 6]


def tfilt(z):
    out = np.empty_like(z)
    mem = np.zeros(z.shape[1:], np.float32)
    for t in range(z.shape[0] - 1, -1, -1):
        mem = gam * mem + (1 - gam) * z[t]
        out[t] = mem
    return out


def sfilt(img):
    g = gaussian_filter(img, (0, SIG, SIG), mode="reflect", truncate=4)
    return g - gaussian_filter(g, (0, 5 * SIG, 5 * SIG), mode="reflect", truncate=4)


# filter weights (time x space) for the white-noise SD and the cumulants
L = 40
ht = (1 - gam) * gam ** np.arange(L)
d = np.zeros((1, 41, 41), np.float32); d[0, 20, 20] = 1
ds = sfilt(d)[0]
sd_white = np.sqrt((ht ** 2).sum() * (ds ** 2).sum())

y = rng.poisson(np.broadcast_to(b, (T, H, W))).astype(np.float32)
bins = np.array([0.03, 0.1, 0.3, 1, 3, 10, 30.01])
colbin = np.digitize(np.exp(logb), bins) - 1

for model in ("plain", "silo"):
    s2 = b + (1.67 * b10 if model == "silo" else 0)
    z = (y - b) / np.sqrt(s2)
    u = sfilt(tfilt(z)) / sd_white
    # exact cumulants of u given b (weights w = ht x ds / sigma, Poisson: kappa_n = sum w^n b)
    k2 = np.zeros((H, W)); k3 = np.zeros((H, W))
    for p_, arr in ((2, k2), (3, k3)):
        ker = ds ** p_
        num = fftconvolve(np.pad(b / s2 ** (p_ / 2), 20, mode="reflect"), ker, mode="valid")
        arr[:] = (ht ** p_).sum() * num / sd_white ** p_
    g1 = k3 / k2 ** 1.5
    zc = u / np.sqrt(k2)
    zcf = zc - g1 / 6 * (zc ** 2 - 1) + g1 ** 2 / 36 * (5 * zc ** 3 - 2 * zc) * 0  # first-order Cornish-Fisher
    uu = u[: T - 10]
    print(f"\n== {model} noise model ==")
    print("bin (photons/px/frame) | var(u) | P(u>t)/Gauss for t=3,4,5,6 | P(u<-t)/Gauss | skew-corrected P(>t)/Gauss")
    for i in range(len(bins) - 1):
        cols = colbin == i
        x = uu[:, 10:-10][:, :, cols].ravel()
        k2m = k2[10:-10][:, cols].mean()
        xc = (zcf[: T - 10, 10:-10][:, :, cols]).ravel()
        xn = x / np.sqrt(k2m)
        hi = [np.mean(xn > t) / norm.sf(t) for t in TH]
        lo = [np.mean(xn < -t) / norm.sf(t) for t in TH]
        hc = [np.mean(xc > t) / norm.sf(t) for t in TH]
        f = lambda a: " ".join(f"{v:7.2f}" for v in a)  # noqa: E731
        print(f"  {bins[i]:5.2f}-{bins[i+1]:5.2f} | {x.var():5.2f} | {f(hi)} | {f(lo)} | {f(hc)}")
    # SiLo's activity image
    U = sfilt(tfilt(z))
    loc = (U == maximum_filter(U, size=(1, 3, 3))) & (U > np.roll(U, 1, 0)) & (U >= np.roll(U, -1, 0))
    loc[0] = loc[-1] = False
    nT = T - int(np.ceil(1.5 * TAU))
    sk = (np.where(loc[1:nT], U[1:nT] ** 2, 0)).sum(0) / (300 + nT)
    skm = sk - median_filter(sk, 5, mode="reflect")
    print("activity image by brightness bin: mean | SD of (image - 5x5 median), relative to the brightest bin")
    ref = None
    for i in range(len(bins) - 1)[::-1]:
        cols = colbin == i
        m_, s_ = sk[5:-5][:, cols].mean(), skm[5:-5][:, cols].std()
        ref = ref or (m_, s_)
        print(f"  {bins[i]:5.2f}-{bins[i+1]:5.2f}: mean {m_/ref[0]:.2f}  local SD {s_/ref[1]:.2f}  "
              f"99.9th pct of local excess / local SD {np.percentile(skm[5:-5][:, cols], 99.9)/s_:.2f}")
