"""The simulator: noise with the template's statistics, exact truth, consistent whole-trial inputs."""

import h5py
import numpy as np

from slap2_f0.io import TrialFile
from slap2_f0.simulate import SimConfig, Template, _filter, draw, noise_kernels, window
from slap2_f0.stats import trial_stats


def _template_file(fn, T=600, R=30, C=34, seed=0):
    rng = np.random.default_rng(seed)
    base = 4.0 + 4 * rng.random((R, C)).astype(np.float32)
    movie = 9.0 * rng.poisson(np.broadcast_to(base / 9.0, (T, R, C))).astype(np.float32)
    act = np.zeros((T, R, C), np.float32)
    act[100:110, 10:14, 10:14] = 6.0
    act[400:405, 20:23, 5:8] = 3.0
    movie = movie + act
    lab = np.zeros((R, C), np.uint8)
    lab[8:25, 3:16] = 1
    sup = np.zeros((R, C), np.uint8)
    sup[9:15, 9:15] = 1
    sup[19:24, 4:9] = 1
    with h5py.File(fn, "w") as f:
        f["movie"] = movie[None]
        f["activity"] = act[None]
        f["mean_image"] = movie.mean(0)[None]
        f["varfac"] = np.ones((R, C), np.float32)
        f["mask/valid"] = np.ones((R, C), np.uint8)
        f["mask/labeled"] = lab
        f["mask/footprint_support"] = sup
        f["frame/discard"] = np.zeros(T, np.uint8)
        f.attrs["label"] = "s"


def test_kernels_match_correlations():
    st = {"rho_t1": 0.65, "rho_t2": 0.3, "rho_r": 0.25, "rho_c": 0.38}
    kt, kr, kc = noise_kernels(st)
    x = np.random.default_rng(0).standard_normal((4000, 40, 40)).astype(np.float32)
    y = _filter(x, kt, kr, kc)
    c = lambda a, b: float(np.mean(a * b) / np.sqrt(np.mean(a * a) * np.mean(b * b)))  # noqa: E731
    assert abs(c(y[1:], y[:-1]) - 0.65) < 0.03 and abs(c(y[2:], y[:-2]) - 0.3) < 0.05
    assert abs(c(y[:, 1:], y[:, :-1]) - 0.25) < 0.02 and abs(c(y[:, :, 1:], y[:, :, :-1]) - 0.38) < 0.02


def test_window_noise_and_truth(tmp_path):
    fn = tmp_path / "trial_001.h5"
    _template_file(fn)
    st = trial_stats(fn)
    st.update(rho_t1=0.6, rho_t2=0.25, rho_r=0.25, rho_c=0.35)  # pretend the template has correlated noise
    tp = Template(TrialFile(fn), st)
    rng = np.random.default_rng(1)
    r = draw(tp, SimConfig(), rng)
    w = window(tp, r, -20, 640, -2, 34, 0, 34, rng)
    assert w["movie"].shape == (640, 34, 34)
    inside = np.isfinite(w["movie"])
    assert not inside[:20].any() and not inside[620:].any()  # outside the trial
    assert np.isfinite(w["truth"][inside]).all()
    noise = (w["movie"] - w["activity"] - w["truth"])[inside]
    truth = w["truth"][inside]
    assert abs(noise.mean()) < 0.05 * truth.mean()  # unbiased
    assert 0.85 < noise.var() / (tp.gain * truth.mean()) < 1.15  # variance = gain * mean
    assert np.allclose(w["noisy"][inside], (w["movie"] - w["activity"])[inside])
    # the whole-trial mean image is close to the window's own average
    mi = w["mean_image"]
    avg = np.nanmean(w["movie"], 0)
    assert np.nanmean(np.abs(mi - avg)) < 0.1 * np.nanmean(avg)


def test_excess_matches_template(tmp_path):
    """Extra fast noise and slow baseline fluctuations on bright template pixels are
    measured (stats), split into the two parts, and reproduced by the simulation."""
    fn = tmp_path / "trial_001.h5"
    T, R, C = 3200, 24, 24
    rng = np.random.default_rng(3)
    base = np.full((R, C), 4.0, np.float32)
    base[:, 12:] = 40.0  # bright half
    bright = base > 10
    photon = 9.0 * rng.poisson(np.broadcast_to(base / 9.0, (T, R, C))).astype(np.float32) - base
    photon *= np.where(bright, np.sqrt(1.3), 1.0)  # fast excess: 30% more noise variance
    slow = np.zeros((T, 1, 1), np.float32)  # slow excess: a shared OU process (tau 400 frames), SD 3 counts
    a = np.exp(-1 / 400)
    e = rng.standard_normal(T) * np.sqrt(1 - a ** 2)
    for t in range(1, T):
        slow[t] = a * slow[t - 1] + e[t]
    movie = base + photon + 3.0 * slow * bright
    with h5py.File(fn, "w") as f:
        f["movie"] = movie[None]
        f["activity"] = np.zeros((1, T, R, C), np.float32)
        f["mean_image"] = movie.mean(0)[None]
        f["varfac"] = np.ones((R, C), np.float32)
        f["mask/valid"] = np.ones((R, C), np.uint8)
        f["mask/labeled"] = bright.astype(np.uint8)  # the gain is measured on the dim background
        f["mask/footprint_support"] = bright.astype(np.uint8)
        f["frame/discard"] = np.zeros(T, np.uint8)
    st = trial_stats(fn)
    tp = Template(TrialFile(fn), st)
    assert abs(np.median(tp.fast[:, 13:-1]) - 0.3) < 0.12
    assert abs(np.median(tp.slow[:, 13:-1]) - 9.0) < 3.0
    assert np.median(np.abs(tp.fast[:, 1:10])) < 0.08 and np.median(tp.slow[:, 1:10]) < 1.0
    cfg = SimConfig(bleach=0, global_sd=0, field_sd=0)
    sims = [window(tp, draw(tp, cfg, rng), 0, T, 0, R, 0, C, rng) for _ in range(3)]
    noisy = np.stack([w["noisy"] for w in sims])

    def v(x, n):  # variance of n-frame means, per pixel, averaged over virtual trials
        m = x[:, : (T // n) * n].reshape(x.shape[0], -1, n, R, C).mean(2)
        return m.var(1).mean(0)

    for n, key in ((1, "var_res"), (64, "var64_res")):
        for cols in (slice(13, -1), slice(1, 10)):
            assert np.allclose(np.median(v(noisy, n)[:, cols]), np.median(st[key][:, cols]), rtol=0.3), (n, cols)
    # the slow part is baseline (in the truth), the fast part is noise
    assert np.median(np.var(sims[0]["truth"], 0)[:, 13:-1]) > 3.0
    # whole-trial variance input includes both
    assert np.allclose(np.median(sims[0]["var_map"][:, 13:-1]), np.median(st["var_res"][:, 13:-1]), rtol=0.3)
    # without the excess, the bright half has photon noise only
    r0 = dict(draw(tp, SimConfig(bleach=0, global_sd=0, field_sd=0, excess=0.0), rng))
    v0 = np.var(window(tp, r0, 0, T, 0, R, 0, C, rng)["noisy"], 0)
    assert np.median(v0[:, 13:-1]) < 0.85 * np.median(st["var_res"][:, 13:-1])


def test_silent_regions(tmp_path):
    """Silent regions have no activity, in the movie and in the whole-trial
    inputs; the default settings draw exactly what they drew before."""
    fn = tmp_path / "trial_001.h5"
    _template_file(fn)
    st = trial_stats(fn)
    tp = Template(TrialFile(fn), st)
    a, b = draw(tp, SimConfig(), np.random.default_rng(5)), draw(tp, SimConfig(silent_px=(5, 9)), np.random.default_rng(5))
    assert "act_mask" not in a and "act_mask" not in b and a["shift"] == b["shift"] and a["amp"] == b["amp"]
    rng = np.random.default_rng(2)
    r = draw(tp, SimConfig(silent_trial=1.0), rng)
    w = window(tp, r, 0, tp.T, 0, tp.R, 0, tp.C, rng)
    assert not w["activity"].any() and np.nanmax(w["mean_act"]) == 0
    assert np.allclose(w["noisy"], w["movie"], equal_nan=True)
    fr = []
    for _ in range(20):
        r = draw(tp, SimConfig(silent_frac=(0.5, 0.5), silent_px=(5, 9)), rng)
        m = r["act_mask"]
        fr.append(1 - m.mean())
        w = window(tp, r, 0, tp.T, 0, tp.R, 0, tp.C, rng)
        assert not w["activity"][:, m == 0].any() and np.nanmax(w["mean_act"][m == 0], initial=0) == 0
        assert np.allclose(w["mean_act"][m == 1], tp.mean_act[m == 1] * r["amp"], atol=1e-5)
    assert abs(np.mean(fr) - 0.5) < 0.02  # the fraction of valid pixels silenced


def test_fast_shared_brightness(tmp_path):
    """Fast global brightness changes move every pixel's truth together, by
    the drawn fraction, with a short time constant; whole-trial inputs know
    their variance; defaults draw exactly what they drew before."""
    fn = tmp_path / "trial_001.h5"
    _template_file(fn, T=2000)
    st = trial_stats(fn)
    tp = Template(TrialFile(fn), st)
    base = SimConfig(bleach=0, global_sd=0, field_sd=0, excess=0.0)
    a = draw(tp, base, np.random.default_rng(4))
    b = draw(tp, SimConfig(bleach=0, global_sd=0, field_sd=0, excess=0.0, fast_tau_frames=(2, 3)), np.random.default_rng(4))
    assert np.array_equal(a["proc"], b["proc"])
    cfg = SimConfig(bleach=0, global_sd=0, field_sd=0, excess=0.0, fast_global_sd=0.1, fast_field_sd=0.0,
                    fast_tau_frames=(3.0, 3.0))
    rng = np.random.default_rng(8)
    r = draw(tp, cfg, rng)
    w = window(tp, r, 0, tp.T, 0, tp.R, 0, tp.C, rng)
    frac = w["truth"] / tp.B[None] - 1  # the drift factor, the same at every pixel
    assert np.allclose(frac, frac[:, :1, :1], atol=1e-5)
    g = frac[:, 0, 0]
    assert 0.0 < g.std() <= 0.1 * 1.3
    ac = np.corrcoef(g[:-3], g[3:])[0, 1]
    assert abs(ac - np.exp(-1)) < 0.15  # tau 3 frames
    # regional fields add their own processes and are in the whole-trial variance input
    cfg2 = SimConfig(bleach=0, global_sd=0, field_sd=0, excess=0.0, n_fields=0, fast_field_sd=0.1, n_fast_fields=2)
    r2 = draw(tp, cfg2, rng)
    assert r2["fields"].shape[0] == 2 and r2["proc"].shape[0] == 3
    w2 = window(tp, r2, 0, tp.T, 0, tp.R, 0, tp.C, rng)
    tv = np.var(w2["truth"], 0)
    assert np.allclose(np.median(tv / tp.B ** 2), np.median(np.einsum("kl,kij,lij->ij", r2["proc_cov"],
                       np.concatenate([np.ones((1, tp.R, tp.C)), r2["fields"]]),
                       np.concatenate([np.ones((1, tp.R, tp.C)), r2["fields"]]))), rtol=0.05)
