"""The speed options of predict-registered (slap2-F0-network#33) keep the
baselines: identical inputs (direct HDF5 reads, valid-pixel stats) and the
same files with prefetching, background writes and the streaming engine."""

from dataclasses import asdict

import h5py
import numpy as np
import pytest
import torch

from slap2_f0.cli import main
from slap2_f0.registered import ArrayTrial, load_registered, movie_stats, read_pages_direct
from test_predict import _model
from test_registered import _write_registered


def _alignment(path, T, R, C, seed=0):
    """An alignment file with a motion event (discarded frames)."""
    rng = np.random.default_rng(seed)
    rne = -0.5 + 0.001 * rng.standard_normal(T)
    rne[200:202] += 0.05
    with h5py.File(path, "w") as f:
        f["numChannels"] = np.array([[2.0]])
        f["alignHz"] = np.array([[80.0]])
        f["recNegErr"] = rne[None]
        f["slap2/varFacDS"] = rng.uniform(1, 2, (T, C, R)).astype(np.float32)


@pytest.mark.parametrize("compression", ["gzip", None])
def test_read_pages_direct_matches_h5py(tmp_path, compression):
    """The same bytes as h5py's selection for page-chunked datasets; None for
    other layouts (the caller then reads through h5py)."""
    rng = np.random.default_rng(0)
    pages = rng.normal(size=(9, 5, 4)).astype(np.float32)
    pages[:, 0, 0] = np.nan
    fn = tmp_path / "m.h5"
    with h5py.File(fn, "w") as f:
        f.create_dataset("data", data=pages, chunks=(1, 5, 4), compression=compression)
        f.create_dataset("other", data=pages, chunks=(3, 5, 4), compression="gzip")
        f.create_dataset("shuffled", data=pages, chunks=(1, 5, 4), compression="gzip", shuffle=True)
    with h5py.File(fn, "r") as f:
        got = read_pages_direct(f["data"], range(1, 9, 2))
        assert got.tobytes() == f["data"][1:9:2].tobytes()
        assert read_pages_direct(f["other"], range(0, 9, 2)) is None
        assert read_pages_direct(f["shuffled"], range(0, 9, 2)) is None


def test_fast_read_and_valid_stats_are_identical(tmp_path):
    """load_registered with fast_read, and the mean image, variance and gain
    from the valid pixels only, equal the default path bit for bit."""
    rng = np.random.default_rng(3)
    T, R, C = 400, 9, 12
    act = rng.poisson(5.0 + 5 * rng.random((R, C)), (T, R, C)).astype(np.float32)
    act[:, :2, :3] = np.nan  # never imaged
    act[:150, 5, 5] = np.nan  # imaged in part of the trial
    path = _write_registered(tmp_path, "E1T1DMD1", act, np.full((T, R, C), 100, np.float32), "h5")
    adata = tmp_path / "E1T1DMD1_ALIGNMENTDATA.h5"
    _alignment(adata, T, R, C)
    a = load_registered(path, adata)
    b = load_registered(path, adata, fast_read=True)
    for x, y in zip(a, b):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()
    movie, valid, discard, varfac, _ = a
    assert discard.any() and not valid.all()
    t0, t1 = ArrayTrial(movie, valid, discard, varfac), ArrayTrial(movie, valid, discard, varfac, valid_only=True)
    assert t0.mean_image.tobytes() == t1.mean_image.tobytes() and t0.scale == t1.scale
    s0, s1 = movie_stats(movie, valid, discard), movie_stats(movie, valid, discard, block=1, valid_only=True)
    assert s0["mean_movie"].tobytes() == s1["mean_movie"].tobytes()
    assert s0["var_movie"].tobytes() == s1["var_movie"].tobytes() and s0["gain"] == s1["gain"]


def test_speed_options_write_the_same_baselines(tmp_path):
    """--fast (streaming engine, prefetch in a worker process, background
    writes, valid-pixel stats, direct reads) writes the same files as the
    default path: same datasets and attributes (times aside), baselines
    within float32 rounding of each other before the float16 cast."""
    rng = np.random.default_rng(4)
    T, R, C = 300, 30, 34
    moco = tmp_path / "motion_correction"
    moco.mkdir()
    for k, stem in enumerate(("E1T1DMD1", "E1T2DMD1", "E1T1DMD2")):
        act = rng.poisson(5.0 + 5 * rng.random((R, C)), (T, R, C)).astype(np.float32)
        act[:, :4, :5] = np.nan
        _write_registered(moco, stem, act, np.full((T, R, C), 100, np.float32), "h5")
        _alignment(moco / f"{stem}_ALIGNMENTDATA.h5", T, R, C, seed=k)
    model, wcfg = _model(box=True)
    ck = tmp_path / "ck.pt"
    torch.save({"model": model.state_dict(), "config": asdict(model.cfg), "window": asdict(wcfg), "step": 7}, ck)
    common = ["predict-registered", str(moco), "--ckpt", str(ck), "--device", "cpu"]
    main(common + ["--out", str(tmp_path / "a"), "--out_frames", "64", "--sd"])
    main(common + ["--out", str(tmp_path / "b"), "--fast", "--prefetch", "2", "--stream_frames", "50", "--sd"])
    for stem in ("E1T1DMD1", "E1T2DMD1", "E1T1DMD2"):
        fa, fb = (tmp_path / d / f"{stem}_BASELINE-80Hz.h5" for d in ("a", "b"))
        with h5py.File(fa, "r") as a, h5py.File(fb, "r") as b:
            assert sorted(a) == sorted(b) == ["baseline", "sd"]
            for k in a:
                x, y = a[k][()].astype(np.float32), b[k][()].astype(np.float32)
                assert np.array_equal(np.isnan(x), np.isnan(y))
                ok = np.isfinite(x)
                # float16 storage: values within one float16 step (2^-10 relative)
                np.testing.assert_allclose(y[ok], x[ok], rtol=2 ** -10, atol=1e-6)
            keep = {k: v for k, v in a.attrs.items() if not k.startswith("seconds")}
            assert keep == {k: v for k, v in b.attrs.items() if not k.startswith("seconds")}
