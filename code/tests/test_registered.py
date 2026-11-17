"""Network baselines for registered movies: SiLo's frame discards, and an
in-memory trial predicts exactly as the same trial read from a file."""

import h5py
import numpy as np
import pytest
import torch

from slap2_f0.io import TrialFile
from slap2_f0.predict import predict_trial, valid_index
from slap2_f0.registered import ArrayTrial, _close, _dilate, _medfilt1_zeropad, discard_frames, movie_stats
from test_predict import _model


def test_medfilt1_matches_matlab_convention():
    x = np.arange(1.0, 9.0)
    # MATLAB R2024b: medfilt1(1:8, 4) (zero padding; even window x(k-2:k+1))
    assert np.allclose(_medfilt1_zeropad(x, 4), [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 6.5])


def test_close_matches_matlab_imclose():
    """MATLAB R2024b: imclose(imdilate(x, ones(3,1)), ones(3,1)) and imclose(y, ones(5,1))."""
    b = lambda s: np.array([c == "1" for c in s])  # noqa: E731
    assert (_close(_dilate(b("001100010000"), 3), 3) == b("011111111000")).all()
    assert (_close(_dilate(b("101100010001"), 3), 3) == b("111111111111")).all()
    assert (_close(b("100000000001"), 5) == b("100000000001")).all()


def test_discard_frames_marks_motion_and_closes_gaps():
    rng = np.random.default_rng(0)
    rne = -0.5 + 0.001 * rng.standard_normal(2000)
    rne[1000:1003] += 0.05  # a motion event: reconstruction error jumps
    rne[1006] += 0.05  # a second one a few frames later: the closing joins them
    d = discard_frames(rne, 80.0)
    assert d[1000:1007].all() and d.sum() < 20 and not d[:900].any()
    assert discard_frames(rne, 80.0, discard_initial_s=1.0)[:80].all()


def test_array_trial_predicts_as_the_file(tmp_path):
    rng = np.random.default_rng(1)
    T, R, C = 160, 22, 26
    movie = rng.poisson(5.0 + 5 * rng.random((R, C)), (T, R, C)).astype(np.float32)
    valid = np.ones((R, C), bool)
    valid[:4, :6] = False
    discard = np.zeros(T, bool)
    discard[50:53] = True
    movie[:, ~valid] = np.nan
    movie[discard] = np.nan
    vf = (0.2 + 0.3 * rng.random((R, C))).astype(np.float32)
    tr_a = ArrayTrial(movie.copy(), valid, discard, vf)
    st = movie_stats(movie, valid, discard)
    fn = tmp_path / "trial.h5"
    with h5py.File(fn, "w") as f:
        f["movie"] = movie[None]
        f["activity"] = np.zeros_like(movie)[None]
        f["mean_image"] = tr_a.mean_image[None]
        f["varfac"] = vf
        f["mask/valid"] = valid.astype(np.uint8)
        f["mask/labeled"] = np.zeros((R, C), np.uint8)
        f["mask/footprint_support"] = np.zeros((R, C), np.uint8)
        f["frame/discard"] = discard.astype(np.uint8)
    model, wcfg = _model(box=True)
    cpu = torch.device("cpu")
    with TrialFile(fn) as tr_f:
        assert tr_f.scale == pytest.approx(tr_a.scale) and np.allclose(tr_f.varfac, tr_a.varfac)
        mu_f, var_f = predict_trial(model, wcfg, tr_f, st, cpu, out_frames=64, out_px=8)
    mu_a, var_a = predict_trial(model, wcfg, tr_a, st, cpu, out_frames=64, out_px=8, rows=valid_index(tr_a)[0],
                                cols=valid_index(tr_a)[1])
    assert np.allclose(mu_a, mu_f, rtol=1e-6, atol=1e-6) and np.allclose(var_a, var_f, rtol=1e-6, atol=1e-6)
