"""How should a location's per-frame matched-filter outputs be combined into one
source score? SiLo: the sum of squares of the temporal local maxima, divided by
(300 + frames). Alternatives: the per-frame log-likelihood ratio of a sparse
event model (spike-and-slab), with the event rate and amplitude either matched
to the truth or fixed; plain energy; thresholded energy.

One location, one 28 s trial at 80 Hz, white unit noise per frame (after the
spatial projection), events at Poisson times with the 24 ms exponential
kernel, amplitude a per event (in noise SD units at the peak frame) with
+-50% jitter. The matched filter is SiLo's anticausal exponential, scaled to
unit null variance. Reported: AUC of source vs no source, and the source
score's d' (mean difference over null SD), per event rate and amplitude.
"""

import numpy as np
from scipy.stats import rankdata

rng = np.random.default_rng(1)
T, HZ, N = 2240, 80, 4000
gam = np.exp(-1 / (0.024 * HZ))
L = 30
k = gam ** np.arange(L)                     # event kernel, peak 1 at onset
h = (1 - gam) * gam ** np.arange(L)
h = h / np.sqrt((h ** 2).sum())             # unit null variance


def mf(x):
    # anticausal: u_t = sum_j h_j x_{t+j}
    out = np.zeros_like(x)
    for j in range(L):
        out[:, : T - j] += h[j] * x[:, j:]
    return out


def make(rate, amp, signal=True):
    x = rng.standard_normal((N, T)).astype(np.float32)
    if signal:
        ev = rng.random((N, T)) < rate / HZ
        a = np.where(ev, amp * rng.uniform(0.5, 1.5, (N, T)), 0).astype(np.float32)
        s = np.zeros_like(x)
        for j in range(L):
            s[:, j:] += k[j] * a[:, : T - j]
        x += s
    return mf(x)


def scores(u, rate, amp):
    resp = amp * (h @ k)                     # MF output at an event's onset
    lm = (u[:, 1:-1] > u[:, :-2]) & (u[:, 1:-1] >= u[:, 2:])
    out = {
        "silo": (np.where(lm, u[:, 1:-1] ** 2, 0)).sum(1) / (300 + T),
        "energy": (u ** 2).sum(1),
        "pos_energy_thr2": (np.maximum(u - 2, 0) ** 2).sum(1),
    }
    for name, (r_, a_) in {"bayes_oracle": (rate, resp), "bayes_fixed": (1.0, 3.0)}.items():
        p = r_ / HZ
        z = np.clip(a_ * u - a_ ** 2 / 2, None, 60)
        out[name] = np.log1p(p * np.expm1(z)).sum(1)
    return out


def auc(s1, s0):
    r = rankdata(np.concatenate([s1, s0]))
    return (r[: len(s1)].sum() - len(s1) * (len(s1) + 1) / 2) / (len(s1) * len(s0))


u0 = make(0, 0, False)
print("rate Hz | amp (SD at peak) | AUC: " + " | ".join(["silo", "energy", "pos_energy_thr2", "bayes_oracle", "bayes_fixed"]))
for rate in (0.3, 1, 3, 10):
    for amp in (0.5, 1.0, 2.0):
        u1 = make(rate, amp)
        s1, s0 = scores(u1, rate, amp), scores(u0, rate, amp)
        line = " | ".join(f"{auc(s1[n], s0[n]):.3f}" for n in s1)
        dp = " ".join(f"{(s1[n].mean() - s0[n].mean()) / s0[n].std():5.2f}" for n in s1)
        print(f"{rate:5.1f} | {amp:4.1f} | {line}   d': {dp}", flush=True)
