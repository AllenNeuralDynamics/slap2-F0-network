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


def test_dark_sources(tmp_path):
    """Dark sources: chosen support pixels get a lower baseline, in the truth
    and in the mean image, with the same activity; default settings draw what
    they drew before."""
    fn = tmp_path / "trial_001.h5"
    _template_file(fn)
    st = trial_stats(fn)
    tp = Template(TrialFile(fn), st)
    a, b = draw(tp, SimConfig(), np.random.default_rng(5)), draw(tp, SimConfig(dark_px=(1, 2)), np.random.default_rng(5))
    assert "dark" not in a and "dark" not in b and a["shift"] == b["shift"]
    rng = np.random.default_rng(3)
    r = draw(tp, SimConfig(dark_frac=(0.5, 0.5)), rng)
    dark = r["dark"]
    sup = tp.tf.support & tp.tf.valid
    picked = dark < 1
    assert picked.any() and not picked[~sup].any() and abs(picked[sup].mean() - 0.5) < 0.1
    assert (dark[picked] >= 0.02).all() and (dark[picked] <= 0.3).all()
    w0 = window(tp, dict(r, dark=np.ones_like(dark)), 0, tp.T, 0, tp.R, 0, tp.C, np.random.default_rng(1))
    w = window(tp, r, 0, tp.T, 0, tp.R, 0, tp.C, np.random.default_rng(1))
    t0, t1 = np.nanmean(w0["truth"], 0), np.nanmean(w["truth"], 0)
    assert np.allclose(t1[picked], t0[picked] * dark[picked], rtol=0.05, atol=0.05)
    assert np.allclose(t1[~picked], t0[~picked])
    assert np.allclose(w["activity"], w0["activity"])
    assert (np.nanmean(w["mean_image"][picked] - w0["mean_image"][picked]) < -1.0)


def test_photon_zeros(tmp_path):
    """Realistic low counts: never negative, the template's share of zero frames,
    unbiased, variance = gain * mean."""
    fn = tmp_path / "trial_001.h5"
    _template_file(fn, T=1200)
    st = trial_stats(fn)
    tp = Template(TrialFile(fn), st)
    assert tp.zc is not None and 7.0 < float(np.median(tp.zc)) < 11.0  # 9 counts per photon in the template
    plain = window(tp, draw(tp, SimConfig(), np.random.default_rng(2)), 0, 1200, 0, 30, 0, 34, np.random.default_rng(3))
    cfg = SimConfig(photon_zeros=True)
    r = draw(tp, cfg, np.random.default_rng(2))
    w = window(tp, r, 0, 1200, 0, 30, 0, 34, np.random.default_rng(3))
    inside = np.isfinite(w["movie"])
    assert (plain["movie"][inside] < 0).any()  # the filtered Poisson noise goes negative
    m = w["movie"][inside]
    assert (m >= 0).all()
    assert abs((m == 0).mean() - st["zero_frac"]) < 0.06
    noise = (w["movie"] - w["activity"] - w["truth"])[inside]
    truth = w["truth"][inside]
    assert abs(noise.mean()) < 0.05 * truth.mean()
    assert 0.8 < noise.var() / (tp.gain * truth.mean()) < 1.2
    assert np.allclose(w["noisy"][inside], (w["movie"] - w["activity"])[inside])
    # draws before it are unchanged: the truth is the same as without
    assert np.allclose(w["truth"][inside], plain["truth"][inside])


def test_gamma_quantile():
    from scipy import special
    from slap2_f0.simulate import gamma_quantile
    k = np.exp(np.linspace(np.log(0.05), np.log(1500.0), 37))[:, None]
    x = np.linspace(-5.0, 5.0, 41)[None]
    q = gamma_quantile(np.broadcast_to(k, (37, 41)), np.broadcast_to(x, (37, 41)))
    ref = special.gammaincinv(k, special.ndtr(x))
    ok = ref > 1e-20
    assert np.max(np.abs(q[ok] / ref[ok] - 1)) < 0.005
    big = gamma_quantile(np.array([5000.0]), np.array([1.0]))  # Wilson-Hilferty above the table
    assert abs(big[0] / special.gammaincinv(5000.0, special.ndtr(1.0)) - 1) < 1e-3
    kk, xx = np.array([0.3, 2.0, 5000.0]), np.array([-1.0, 0.5, 1.0])
    assert np.allclose(gamma_quantile(kk, xx, log=True), np.log(gamma_quantile(kk, xx)))


def _gg_params(mp, vp, s):
    """k and theta of theta * G^s, G ~ Gamma(k), with mean mp and variance vp (scipy, for checks)."""
    from scipy import optimize, special
    cv2 = lambda k: np.exp(special.gammaln(k) + special.gammaln(k + 2 * s) - 2 * special.gammaln(k + s)) - 1  # noqa: E731
    k = optimize.brentq(lambda k: cv2(k) - vp / mp ** 2, 1e-3, 1e6)
    return k, mp / np.exp(special.gammaln(k + s) - special.gammaln(k))


def test_photon_counts_generalized_gamma():
    """photon_zeros' counts: zeros with probability exp(-mu / c), above 0 a
    generalized gamma theta * G^s (scipy's gengamma with c = 1/s; s = 1 gamma)
    with the positive part's mean and variance, so the counts keep their mean
    and variance for any s while larger s gives more skew."""
    from scipy import special
    from scipy import stats as sst
    from slap2_f0.simulate import _photon_counts
    g = np.random.default_rng(0).standard_normal(1_000_000)
    for mu, var, zc in ((1.9, 23.5, 2.7), (5.0, 60.0, 4.0), (40.0, 480.0, 4.0)):
        z = np.exp(-mu / zc)
        mp = mu / (1 - z)
        vp = (var + mu ** 2) / (1 - z) - mp ** 2
        u = z + (1 - z) * np.array([0.05, 0.25, 0.5, 0.9, 0.99, 0.999])
        skews = []
        for s in (1.0, 1.4, 2.0):
            k, theta = _gg_params(mp, vp, s)
            q = _photon_counts(np.full(6, mu), np.full(6, var), np.full(6, zc), special.ndtri(u), s)
            ref = sst.gengamma.ppf((u - z) / (1 - z), k, 1 / s, scale=theta)
            assert np.max(np.abs(q / ref - 1)) < 0.01, (mu, s, q, ref)
            x = _photon_counts(np.full(g.shape, mu), np.full(g.shape, var), np.full(g.shape, zc), g, s)
            assert abs((x == 0).mean() - z) < 0.003
            assert abs(x.mean() / mu - 1) < 0.01 and abs(x.var() / var - 1) < 0.03, (mu, s, x.mean(), x.var())
            skews.append(float(((x - x.mean()) ** 3).mean() / x.var() ** 1.5))
        assert skews[0] < skews[1] < skews[2]


def _gg_template(fn, s=1.4, T=2000, R=30, C=34, seed=0):
    """Independent voxels, 0 with probability exp(-mean / 2.7), else a
    generalized gamma with power s (scipy), variance 12 x the mean: the shape
    of real dim pixels' counts, without correlations."""
    from scipy import stats as sst
    rng = np.random.default_rng(seed)
    base = 0.8 + 2.2 * rng.random((R, C))
    movie = np.zeros((T, R, C), np.float32)
    for i in range(R):
        for j in range(C):
            mu, var = base[i, j], 12.0 * base[i, j]
            z = np.exp(-mu / 2.7)
            mp = mu / (1 - z)
            k, theta = _gg_params(mp, (var + mu ** 2) / (1 - z) - mp ** 2, s)
            pos = sst.gengamma.rvs(k, 1 / s, scale=theta, size=T, random_state=rng)
            movie[:, i, j] = np.where(rng.random(T) < z, 0.0, pos)
    lab = np.zeros((R, C), np.uint8)
    lab[5:12, 5:12] = 1
    with h5py.File(fn, "w") as f:
        f["movie"] = movie[None]
        f["activity"] = np.zeros((1, T, R, C), np.float32)
        f["mean_image"] = movie.mean(0)[None]
        f["varfac"] = np.ones((R, C), np.float32)
        f["mask/valid"] = np.ones((R, C), np.uint8)
        f["mask/labeled"] = lab
        f["mask/footprint_support"] = lab
        f["frame/discard"] = np.zeros(T, np.uint8)
        f.attrs["label"] = "s"


def test_photon_zeros_skew(tmp_path):
    """Counts whose positive part has the default shape (as real dim pixels):
    photon_zeros reproduces the template's skew, share of zeros and gain; with
    a gamma positive part (pz_shape 1, as up to 2026-11-21) the skew is low."""
    from slap2_f0.simulate import write_trial
    fn = tmp_path / "trial_001.h5"
    _gg_template(fn, s=SimConfig().pz_shape)
    st = trial_stats(fn)
    tp = Template(TrialFile(fn), st)
    out = {}
    for shape in (SimConfig().pz_shape, 1.0):
        sf = tmp_path / f"sim_{shape}.h5"
        write_trial(tp, sf, SimConfig(photon_zeros=True, pz_shape=shape, excess=0.0), seed=1, compress=False)
        out[shape] = trial_stats(sf)
    s = out[SimConfig().pz_shape]
    assert abs(s["skew"] / st["skew"] - 1) < 0.05, (s["skew"], st["skew"])
    assert out[1.0]["skew"] / st["skew"] < 0.95
    assert abs(s["zero_frac"] - st["zero_frac"]) < 0.02 and abs(s["gain"] / st["gain"] - 1) < 0.05


def _counts_template(fn, T=1500, R=40, C=44, seed=0):
    """Photon events spread by a short positive kernel (as the microscope's
    frame formation does): never negative, exact zeros where no event lands
    in the kernel, correlated in time and space, about 30-50% zero frames."""
    rng = np.random.default_rng(seed)
    kt, kr, kc = np.array([0.6, 1.0, 0.6]), np.array([0.25, 1.0, 0.25]), np.array([0.45, 1.0, 0.45])
    base = (0.8 + 1.7 * rng.random((R, C))).astype(np.float32)  # mean counts
    amp = 9.0
    lam = np.pad(base, 1, mode="edge") / (amp * kt.sum() * kr.sum() * kc.sum())
    ev = amp * rng.poisson(np.broadcast_to(lam, (T + 2, R + 2, C + 2))).astype(np.float32)
    movie = _filter(ev, kt, kr, kc)
    act = np.zeros((T, R, C), np.float32)
    act[100:110, 10:14, 10:14] = 3.0
    lab = np.zeros((R, C), np.uint8)
    lab[8:20, 6:16] = 1
    with h5py.File(fn, "w") as f:
        f["movie"] = (movie + act)[None]
        f["activity"] = act[None]
        f["mean_image"] = (movie + act).mean(0)[None]
        f["varfac"] = np.ones((R, C), np.float32)
        f["mask/valid"] = np.ones((R, C), np.uint8)
        f["mask/labeled"] = lab
        f["mask/footprint_support"] = lab
        f["frame/discard"] = np.zeros(T, np.uint8)
        f.attrs["label"] = "s"


def test_photon_zeros_keep_correlations(tmp_path):
    """Realistic low counts reproduce the template's correlations in time and
    space, its share of zeros, gain and (roughly) skew; the legacy draw lost
    about half of the correlations."""
    from slap2_f0.simulate import write_trial
    fn = tmp_path / "trial_001.h5"
    _counts_template(fn)
    st = trial_stats(fn)
    assert 0.25 < st["zero_frac"] < 0.6 and st["rho_c"] > 0.3 and st["rho_t1"] > 0.4
    tp = Template(TrialFile(fn), st)
    out = {}
    for name, cfg in (("new", SimConfig(photon_zeros=True, excess=0.0)),
                      ("legacy", SimConfig(photon_zeros=True, pz_legacy=True, excess=0.0))):
        sf = tmp_path / f"sim_{name}.h5"
        write_trial(tp, sf, cfg, seed=1, compress=False)
        out[name] = trial_stats(sf)
    s = out["new"]
    for key in ("rho_t1", "rho_t2", "rho_r", "rho_c"):
        assert abs(s[key] - st[key]) < 0.04, (key, s[key], st[key])
        assert abs(out["legacy"][key] - st[key]) > abs(s[key] - st[key])
    assert abs(s["zero_frac"] - st["zero_frac"]) < 0.06
    assert abs(s["gain"] / st["gain"] - 1) < 0.1
    assert abs(s["skew"] / st["skew"] - 1) < 0.25


def test_shuffled_timing(tmp_path):
    """Shuffled activity timing keeps every event whole and the total activity."""
    fn = tmp_path / "trial_001.h5"
    _template_file(fn)
    st = trial_stats(fn)
    tp = Template(TrialFile(fn), st)
    plain = window(tp, draw(tp, SimConfig(), np.random.default_rng(4)), 0, 600, 0, 30, 0, 34, np.random.default_rng(5))
    r = draw(tp, SimConfig(shuffle_prob=1.0, shuffle_frames=(20, 40)), np.random.default_rng(4))
    assert sorted(r["tmap"]) == list(range(tp.T))
    w = window(tp, r, 0, 600, 0, 30, 0, 34, np.random.default_rng(5))
    assert np.isclose(w["activity"].sum(), plain["activity"].sum(), rtol=1e-5)
    on = np.flatnonzero(w["activity"][:, 11, 11] > 0)  # the 10-frame event stays in one piece
    assert len(on) == 10 and on[-1] - on[0] == 9
    assert not np.array_equal(w["activity"], plain["activity"])


def test_local_motion(tmp_path):
    """Local motion: without it the draws are unchanged; with it each frame's
    truth is the static truth warped by the drawn displacement field
    (out[x] = static[x - D(x)]), and the field's SD per pixel is about the
    drawn SD."""
    from scipy import ndimage
    fn = tmp_path / "trial_001.h5"
    _template_file(fn, T=800, R=40, C=48)
    st = trial_stats(fn)
    tp = Template(TrialFile(fn), st)
    base = SimConfig(excess=0.0, motion_sd_px=(0.1, 0.1))
    a = draw(tp, base, np.random.default_rng(3))
    b = draw(tp, SimConfig(excess=0.0, motion_sd_px=(0.1, 0.1), local_sd_px=(0.3, 0.3)), np.random.default_rng(3))
    assert np.array_equal(a["motion"], b["motion"]) and "local_fields" not in a
    assert b["local_fields"].shape == (2, 3, tp.R, tp.C) and b["local_proc"].shape == (2, 3, tp.T)
    r = draw(tp, SimConfig(excess=0.0, local_sd_px=(0.3, 0.3), local_px=(15, 15)), np.random.default_rng(5))
    D = np.einsum("ajt,ajrc->tarc", r["local_proc"], r["local_fields"])
    assert 0.15 < D.std(0).mean() < 0.7  # fast and slow parts of 0.3 each: about 0.42
    w = window(tp, r, 0, 50, 0, tp.R, 0, tp.C, np.random.default_rng(6))
    rr, cc = np.indices((tp.R, tp.C), dtype=np.float32)
    for t in (0, 17, 49):
        ref = ndimage.map_coordinates(w["truth_static"][t], [rr - D[t, 0], cc - D[t, 1]], order=3, mode="nearest")
        assert np.allclose(w["truth"][t], ref, rtol=1e-4, atol=1e-4)
