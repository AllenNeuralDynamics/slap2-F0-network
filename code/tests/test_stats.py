"""Trial statistics: GIAnT fit artifacts where pixels miss movie frames, and optional maps."""

import h5py
import numpy as np


def _edge_trial(fn, T=400, R=16, C=20, seed=0):
    """A trial with an artifact at the field's right edge, as on 804730 path 2:
    a core pixel whose activity is far above the movie (the mean rule finds
    it); next to it, edge pixels that leave the imaged field (movie NaN) in 50
    frames, where GIAnT's activity reaches 2000 counts and then decays into the
    first frames with movie; an isolated edge pixel with the same kind of
    artifact (500 counts) far from the core; and a pixel that misses frames
    with ordinary activity there."""
    rng = np.random.default_rng(seed)
    base = np.full((R, C), 8.0, np.float32)
    movie = 9.0 * rng.poisson(np.broadcast_to(base / 9.0, (T, R, C))).astype(np.float32)
    act = np.zeros((T, R, C), np.float32)
    act[:, 8, 17] = 600.0  # the core: mean activity ~75x the movie, not in it
    movie[100:150, 6:11, 19] = np.nan  # edge pixels next to the core leave the field
    act[100:150, 6:11, 19] = 2000.0
    act[150:153, 6:11, 19] = 150.0  # decaying into the frames with movie
    movie[200:220, 2, 19] = np.nan  # an isolated edge pixel with the same artifact
    act[200:220, 2, 19] = 500.0
    movie[200:220, 13, 19] = np.nan  # a pixel that misses frames, with ordinary activity
    act[180:240, 13, 19] = 5.0
    movie[180:200, 13, 19] += 5.0
    movie[220:240, 13, 19] += 5.0
    discard = np.zeros(T, np.uint8)
    discard[390:] = 1
    movie[395, 13, 19] = np.nan  # a discarded frame does not count
    act[395, 13, 19] = 9999.0
    lab = np.zeros((R, C), np.uint8)
    lab[4:12, 14:20] = 1
    with h5py.File(fn, "w") as f:
        f["movie"] = movie[None]
        f["activity"] = act[None]
        f["mean_image"] = np.nanmean(movie, 0)[None]
        f["varfac"] = np.ones((R, C), np.float32)
        f["mask/valid"] = np.ones((R, C), np.uint8)
        f["mask/labeled"] = lab
        f["mask/footprint_support"] = lab
        f["frame/discard"] = discard
        f.attrs["label"] = "s"


def test_artifacts_where_frames_are_missing(tmp_path):
    """Activity in frames without movie: trial_stats measures its maximum
    (act_nan) and the movie's (movie_max). unfit_pixels grows the mean rule's
    mask through neighbouring pixels that miss frames and carry activity
    (older stats too), and with the maps also flags activity in missing
    frames above twice the movie's maximum anywhere; the simulator then puts
    no activity there."""
    from slap2_f0.io import TrialFile
    from slap2_f0.simulate import SimConfig, Template, draw, window
    from slap2_f0.stats import OPTIONAL, load_stats, sanitize, trial_stats, unfit_pixels

    fn = tmp_path / "trial_001.h5"
    _edge_trial(fn)
    raw = trial_stats(fn)
    assert np.allclose(raw["act_nan"][6:11, 19], 2000) and raw["act_nan"][2, 19] == 500 and raw["act_nan"][13, 19] == 5
    others = np.ones(raw["act_nan"].shape, bool)
    others[6:11, 19] = others[2, 19] = others[13, 19] = False
    assert (raw["act_nan"][others] == 0).all()
    assert np.isfinite(raw["movie_max"]).all() and 15 < raw["movie_max"][13, 19] < 80
    # the mean rule cannot see them: the spikes are in frames without movie
    assert (raw["mean_act"][6:11, 19] < 2 * raw["mean_movie"][6:11, 19]).all()
    before = unfit_pixels({k: v for k, v in raw.items() if k not in OPTIONAL + ("n_valid",)})
    assert before.sum() == 9 and before[7:10, 16:19].all()
    old = {k: v for k, v in raw.items() if k not in OPTIONAL}  # stats made before the maps existed
    m_old = unfit_pixels(old)
    assert m_old[6:11, 19].all() and (m_old & ~before).sum() == 5  # grown through the edge pixels
    assert not m_old[2, 19] and not m_old[13, 19]
    m_new = unfit_pixels(raw)
    assert not (m_old & ~m_new).any()
    assert m_new[1:4, 18:20].all() and (m_new[:4] & ~m_old[:4]).sum() == 6  # the isolated one, with its neighbours
    assert not m_new[13, 19]  # ordinary activity in missing frames: kept
    # the simulator puts no activity at any of them
    st = load_stats([fn], tmp_path / "stats", workers=0)[str(fn)]
    assert np.array_equal(st["unfit"], m_new)
    with TrialFile(fn) as tf:
        sims = {}
        for name, s in (("now", st), ("before", sanitize(old) | {"unfit": before})):  # before: the mean rule's mask
            tp = Template(tf, s)
            rng = np.random.default_rng(3)
            sims[name] = window(tp, draw(tp, SimConfig(), rng), 0, tf.n_frames, 0, tf.n_rows, 0, tf.n_cols, rng)
    assert sims["before"]["activity"][:, 6:11, 19].max() >= 1000  # 2000 counts times the amplitude (0.5-1.5)
    assert sims["now"]["activity"][:, m_new].max() == 0
    assert sims["now"]["activity"][:, 13, 19].max() > 0


def test_sidecars_with_optional_maps_come_first(tmp_path, monkeypatch):
    """A sidecar made before the optional maps is still used (no recompute);
    an overlay folder's sidecar that has them is preferred."""
    import slap2_f0.stats as S

    d = tmp_path / "data" / "sessA" / "Path2"
    d.mkdir(parents=True)
    fn = d / "trial_001.h5"
    _edge_trial(fn)
    full = S.trial_stats(fn)
    size = fn.stat().st_size
    np.savez(S.sidecar_path(fn), file_size=size, **{k: v for k, v in full.items() if k not in S.OPTIONAL})
    ov = tmp_path / "overlay"
    S.overlay_path(fn, ov).parent.mkdir(parents=True)
    np.savez(S.overlay_path(fn, ov), file_size=size, **full)

    def boom(*a, **k):
        raise AssertionError("recomputed")

    monkeypatch.setattr(S, "trial_stats", boom)
    plain = S.load_stats([fn], None, workers=0)[str(fn)]
    assert "act_nan" not in plain and plain["unfit"][6:11, 19].all() and not plain["unfit"][2, 19]
    both = S.load_stats([fn], None, workers=0, sidecar_dirs=[str(ov)])[str(fn)]
    assert "act_nan" in both and both["unfit"][2, 19]
