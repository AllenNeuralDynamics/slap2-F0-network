"""Where real training windows put loss (data.WindowDataset): labeled pixels
outside every footprint have weight 1 by default (their target is the movie:
GIAnT fits no activity there) and 0 with w_unfitted=0, as in runs up to S4;
GIAnT fit artifacts, the trial-edge frames and discarded frames have none
either way."""
import h5py
import numpy as np

from slap2_f0.data import WindowConfig, WindowDataset
from slap2_f0.stats import load_stats

T, R, C, HZ = 200, 16, 16, 10.0  # 10 Hz: the 2 s edges are 20 frames
DISCARDED = 100


def _trial(fn):
    rng = np.random.default_rng(0)
    movie = rng.poisson(8.0, (1, T, R, C)).astype(np.float32)
    act = np.zeros_like(movie)
    act[0, 60:66, 3:8, 3:8] = 6.0  # an event in the footprint
    act[0, :, 13, 13] = 5000.0  # a GIAnT fit artifact: activity far above the movie at a dim pixel
    movie += np.where(act < 100, act, 0)
    sup = np.zeros((R, C), np.uint8)
    sup[3:8, 3:8] = 1
    lab = sup.copy()
    lab[10:13, 2:7] = 1  # labeled, outside every footprint
    disc = np.zeros(T, np.uint8)
    disc[DISCARDED] = 1
    with h5py.File(fn, "w") as f:
        f["movie"] = movie
        f["activity"] = act
        f["mean_image"] = movie[0].mean(0)[None]
        f["varfac"] = np.ones((R, C), np.float32)
        f["mask/valid"] = np.ones((R, C), np.uint8)
        f["mask/labeled"] = lab
        f["mask/footprint_support"] = sup
        f["frame/discard"] = disc
        f.attrs["nSources"] = np.array([1.0])
        f.attrs["alignHz"] = np.array([HZ])
    return fn


def _window(files, stats, **kw):
    # a loss region twice the trial's size in every axis holds the whole trial wherever it is placed
    cfg = WindowConfig(out_frames=2 * T, out_px=2 * R, t_reach=0, s_reach=0, **kw)
    return WindowDataset(files, cfg, stats, length=1, seed=0, flip="none")[0]


def test_labeled_pixels_outside_footprints_get_full_loss(tmp_path):
    fn = _trial(tmp_path / "trial_001.h5")
    stats = load_stats([fn])
    assert WindowConfig().w_unfitted == 1.0
    new = _window([fn], stats)  # the default
    old = _window([fn], stats, w_unfitted=0.0)  # runs up to S4
    assert np.array_equal(new["movie"].numpy(), old["movie"].numpy(), equal_nan=True)  # the same window
    w1, w0 = new["weight"].numpy(), old["weight"].numpy()
    lab, sup = new["labeled"].numpy().astype(bool), new["support"].numpy().astype(bool)
    outside = lab & ~sup
    assert outside.sum() == 15
    # frames with loss: the whole trial minus 2 s at each end and the discarded frame
    frames = w1.max((1, 2)) > 0
    assert frames.sum() == T - 2 * 20 - 1
    assert (w1[frames][:, outside] == 1).all()  # labeled outside every footprint: full loss by default
    assert (w0[:, outside] == 0).all()  # and none with w_unfitted 0
    others = ~outside
    assert np.array_equal(w1[:, others], w0[:, others])  # support and background pixels as before
    r0, c0 = np.argwhere(sup).min(0) - 3  # the trial's corner in the window (the footprint starts at 3, 3)
    assert (w1[:, r0 + 13, c0 + 13] == 0).all() and (w0[:, r0 + 13, c0 + 13] == 0).all()  # the fit artifact
    assert w1[frames][:, r0 + 14, c0 + 9].min() == 1  # background 4 px away has loss (the artifact mask is dilated)
    # trial edges, the discarded frame and frames outside the trial: no loss at any pixel
    movie = new["movie"].numpy()
    inside = np.isfinite(movie).any((1, 2))
    first = int(np.flatnonzero(inside)[0])
    for t in (first, first + 19, first + DISCARDED, first + T - 20, first + T - 1):
        assert w1[t].max() == 0, t
    assert w1[first + 20].max() > 0 and w1[first + T - 21].max() > 0
    assert (w1[~inside] == 0).all()


def test_trials_without_sources_weigh_every_valid_pixel(tmp_path):
    fn = _trial(tmp_path / "trial_001.h5")
    with h5py.File(fn, "a") as f:
        f["activity"][...] = 0
        f.attrs["nSources"] = np.array([0.0])
    stats = load_stats([fn])
    w0 = _window([fn], stats, w_unfitted=0.0)
    lab, sup = w0["labeled"].numpy().astype(bool), w0["support"].numpy().astype(bool)
    frames = w0["weight"].numpy().max((1, 2)) > 0
    assert (w0["weight"].numpy()[frames][:, lab & ~sup] == 1).all()  # a GFP control: no activity anywhere
