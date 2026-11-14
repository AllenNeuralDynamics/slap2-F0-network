"""Likelihoods are minimized at the true mean and variance; the reader pads windows."""

import h5py
import numpy as np
import torch

from slap2_f0.io import TrialFile
from slap2_f0.losses import beta_nll, student_t_nll


def test_nll_minimum_at_truth():
    torch.manual_seed(0)
    y = 2.0 + 0.5 * torch.randn(200_000)
    w = torch.ones_like(y)
    for f in (lambda m, v: beta_nll(y, m, v, w, beta=0.0), lambda m, v: student_t_nll(y, m, v, w, nu=30.0)):
        best = min(((f(torch.full_like(y, m), torch.full_like(y, v)).item(), m, v)
                    for m in (1.8, 2.0, 2.2) for v in (0.15, 0.25, 0.35)))
        assert best[1:] == (2.0, 0.25)


def _write_trial(fn, T=20, R=6, C=7):
    rng = np.random.default_rng(0)
    movie = rng.poisson(5.0, (1, T, R, C)).astype(np.float32)
    movie[:, :, 0, 0] = np.nan
    with h5py.File(fn, "w") as f:
        f["movie"] = movie
        f["activity"] = np.ones_like(movie)
        f["mean_image"] = np.nanmean(movie, 1)
        f["varfac"] = np.ones((R, C), np.float32)
        valid = np.ones((R, C), np.uint8)
        valid[0, 0] = 0
        f["mask/valid"] = valid
        f["mask/labeled"] = valid
        f["mask/footprint_support"] = np.zeros((R, C), np.uint8)
        disc = np.zeros(T, np.uint8)
        disc[3] = 1
        f["frame/discard"] = disc
        f.attrs["nSources"] = 0


def test_list_trials_skips_other_files(tmp_path):
    from slap2_f0.io import list_trials
    (tmp_path / "s" / "Path1").mkdir(parents=True)
    (tmp_path / "s" / "giant").mkdir(parents=True)
    _write_trial(tmp_path / "s" / "Path1" / "trial_001.h5")
    with h5py.File(tmp_path / "s" / "giant" / "trial_table.h5", "w") as f:  # GIAnT's table, no movie
        f["filename"] = np.zeros(2)
    with h5py.File(tmp_path / "s" / "Path1" / "trial_002.h5", "w") as f:  # a trial file without a movie
        f["x"] = np.zeros(2)
    assert [p.name for p in list_trials(tmp_path)] == ["trial_001.h5"]


def test_read_pads_outside_trial(tmp_path):
    fn = tmp_path / "trial_001.h5"
    _write_trial(fn)
    with TrialFile(fn) as tr:
        p = tr.read(-2, 8, -1, 4, 5, 4)  # starts before the trial, ends past its columns
        assert p.movie.shape == (8, 4, 4)
        assert np.isnan(p.movie[:2]).all()  # before the first frame
        assert np.isnan(p.movie[:, 0]).all()  # row -1
        assert np.isnan(p.movie[:, :, 2:]).all()  # columns 7, 8
        assert np.isnan(p.movie[5]).all()  # frame 3 is discarded
        assert np.isfinite(p.movie[2, 1:, :2]).all()
        assert p.activity[0].sum() == 0 and p.activity[2, 1:, :2].sum() == 6
