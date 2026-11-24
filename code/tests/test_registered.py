"""Network baselines for registered movies: SiLo's frame discards, and an
in-memory trial predicts exactly as the same trial read from a file."""

import h5py
import numpy as np
import pytest
import torch

from slap2_f0.io import TrialFile
from slap2_f0.predict import predict_trial, valid_index
from slap2_f0.registered import (ArrayTrial, _close, _dilate, _medfilt1_zeropad, discard_frames, load_registered,
                                 movie_stats, registered_movies)
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


def _write_registered(moco, stem, act, other, fmt):
    """A registered movie as GIAnT writes it: pages (MATLAB col, row), two
    channels interleaved by frame; TIFF one page at a time (GIAnT-Python) or
    HDF5 /data (pages, col, row) chunked by page."""
    pages = np.empty((2 * act.shape[0], act.shape[2], act.shape[1]), np.float32)
    pages[0::2] = act.transpose(0, 2, 1)
    pages[1::2] = other.transpose(0, 2, 1)
    path = moco / f"{stem}_REGISTERED_DOWNSAMPLED-80Hz.{fmt}"
    if fmt == "tif":
        tifffile = pytest.importorskip("tifffile")
        with tifffile.TiffWriter(path, bigtiff=True) as tw:
            for page in pages:
                tw.write(page, photometric="minisblack")
    else:
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=pages, chunks=(1,) + pages.shape[1:], compression="gzip",
                             compression_opts=1)
    return path


@pytest.mark.parametrize("fmt", ["tif", "h5"])
def test_load_registered_reads_tif_and_h5(tmp_path, fmt):
    rng = np.random.default_rng(2)
    T, R, C = 400, 7, 10  # MATLAB rows x cols; T > the 4 s median of the discards
    act = rng.poisson(5.0, (T, R, C)).astype(np.float32)
    act[:, 0, 0] = np.nan  # a pixel never imaged
    other = np.full((T, R, C), 100, np.float32)
    path = _write_registered(tmp_path, "E1T1DMD1", act, other, fmt)
    rne = -0.5 + 0.001 * rng.standard_normal(T)
    rne[200:202] += 0.05
    vf = rng.uniform(1, 2, (T, C, R)).astype(np.float32)  # h5py order of MATLAB rows x cols x frames
    adata = tmp_path / "E1T1DMD1_ALIGNMENTDATA.h5"
    with h5py.File(adata, "w") as f:
        f["numChannels"] = np.array([[2.0]])
        f["alignHz"] = np.array([[80.0]])
        f["recNegErr"] = rne[None]
        f["slap2/varFacDS"] = vf
    movie, valid, discard, varfac, hz = load_registered(path, adata)
    assert movie.shape == (T, R, C) and hz == 80.0
    assert discard[200:202].all() and not valid[0, 0] and valid.sum() == R * C - 1
    keep = ~discard
    np.testing.assert_array_equal(movie[keep][:, valid], act[keep][:, valid])
    assert np.isnan(movie[discard]).all()
    np.testing.assert_allclose(varfac, np.nanmean(vf, axis=0).T)
    assert np.allclose(load_registered(path, adata, channel=1)[0][keep][:, valid], 100)


def test_registered_movies_takes_one_movie_per_trial(tmp_path):
    import os

    act = np.ones((4, 3, 5), np.float32)
    for stem, fmts in (("E1T1DMD1", ["h5"]), ("E1T2DMD1", ["tif"]), ("E1T3DMD1", ["tif", "h5"]),
                       ("E1T4DMD1", ["h5", "tif"])):
        for age, fmt in zip((200, 100), fmts):  # the second listed is newer
            path = _write_registered(tmp_path, stem, act, act, fmt)
            os.utime(path, (1e9 - age, 1e9 - age))
    names = [p.name for p in registered_movies(tmp_path)]
    assert names == ["E1T1DMD1_REGISTERED_DOWNSAMPLED-80Hz.h5", "E1T2DMD1_REGISTERED_DOWNSAMPLED-80Hz.tif",
                     "E1T3DMD1_REGISTERED_DOWNSAMPLED-80Hz.h5", "E1T4DMD1_REGISTERED_DOWNSAMPLED-80Hz.tif"]
