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
