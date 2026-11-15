"""Reference-baseline (ceiling) analysis: helpers match brute force, and it runs on a trial file."""

import h5py
import numpy as np

from slap2_f0.ceiling import _blind_mean, _movmean, trial_ceiling


def test_blind_mean_matches_brute_force():
    rng = np.random.default_rng(0)
    v = rng.random((40, 3, 2)).astype(np.float32)
    v[5:9, 1, 0] = np.nan
    fin = np.isfinite(v)
    got = _blind_mean(np.where(fin, v, 0), fin.astype(np.float32), gap=2, reach=5)
    for t in range(40):
        dt = np.abs(np.arange(40) - t)
        sel = (dt > 2) & (dt <= 7)
        want = np.nanmean(v[sel], axis=0)
        np.testing.assert_allclose(got[t], want, rtol=1e-5)


def test_movmean_matches_brute_force():
    rng = np.random.default_rng(1)
    a = rng.random((30, 4)).astype(np.float32)
    got = _movmean(a, 8)
    for t in range(30):
        np.testing.assert_allclose(got[t], a[max(0, t - 4): t + 4].mean(0), rtol=1e-5)


def test_trial_ceiling_runs(tmp_path):
    rng = np.random.default_rng(2)
    T, R, C = 400, 12, 14
    base = 3.0 + rng.random((R, C)).astype(np.float32)
    movie = rng.poisson(np.broadcast_to(base, (T, R, C))).astype(np.float32)
    act = np.zeros_like(movie)
    act[100:110, 4:7, 4:7] = 5.0
    movie += act
    sup = np.zeros((R, C), np.uint8)
    sup[3:8, 3:8] = 1
    fn = tmp_path / "sess" / "Path1" / "trial_001.h5"
    fn.parent.mkdir(parents=True)
    with h5py.File(fn, "w") as f:
        f["movie"] = movie[None]
        f["activity"] = act[None]
        f["mean_image"] = movie.mean(0)[None]
        f["mask/valid"] = np.ones((R, C), np.uint8)
        f["mask/labeled"] = sup
        f["mask/footprint_support"] = sup
        f["frame/discard"] = np.zeros(T, np.uint8)
        f.attrs["label"] = "sess"
        f.attrs["activityGain"] = np.array([[0.8]])
    for smooth in (1, 8):
        rows = {r["class"]: r for r in trial_ceiling(fn, gap=2, reach=20, silo_pixels=20, smooth=smooth)}
        assert {"support", "unlabeled", "loss"} <= set(rows)
        assert abs(rows["loss"]["bias_target_mean"]) < 1e-3  # the oracle constant is unbiased by construction
        # the mean image includes the activity, so it sits above the target on support pixels
        assert rows["support"]["bias_mean_image"] < 0
