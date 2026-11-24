"""The selection-safe event ratios (slap2_f0.ringqc, the reference for the
exporter's ringEventRatios.m) on a synthetic trial with known answers."""
import h5py
import numpy as np

import pytest

from slap2_f0.ringqc import main, ring_event_ratios, trial_ring_event_ratios

T, R, C = 2000, 40, 60
EVENTS = list(range(220, 1800, 130))  # 13 events, fewer than 1% of frames


def _trial(k_core=1.5, k_near=0.8):
    """A noiseless movie with spatially broad events around three sources,
    and an activity that is k_core x the movie's event at <= 1.5 px and
    k_near x at 1.5-2.5 px (and nothing farther out)."""
    rr, cc = np.mgrid[:R, :C]
    movie = np.full((T, R, C), 10.0)
    activity = np.zeros((T, R, C))
    peaks = [(20, 15), (20, 45), (3, 30)]  # the third is too close to the edge
    kinetics = np.zeros(T)
    for e in EVENTS:
        kinetics[e:e + 6] += 0.6 ** np.arange(6)
    for r0, c0 in peaks:
        d = np.hypot(rr - r0, cc - c0)
        prof = 8 * np.exp(-d ** 2 / (2 * 3.0 ** 2)) * (d <= 8)
        k = np.where(d < 1.5, k_core, np.where(d < 2.5, k_near, 0.0))
        movie += kinetics[:, None, None] * prof
        activity += kinetics[:, None, None] * prof * k
    valid = np.ones((R, C), bool)
    discard = np.zeros(T, bool)
    return movie, activity, valid, np.array([p[0] for p in peaks]), np.array([p[1] for p in peaks]), discard


def test_known_ratios():
    movie, act, valid, pr, pc, disc = _trial()
    q = ring_event_ratios(movie, act, valid, pr, pc, disc)
    assert (q.n_sources, q.n_events, q.n_artifact_sources) == (2, 2 * len(EVENTS), 0)
    assert np.isclose(q.core, 1.5) and np.isclose(q.near, 0.8)
    # halving the activity halves both ratios (the events are picked on the movie)
    q2 =ring_event_ratios(movie, 0.5 * act, valid, pr, pc, disc)
    assert np.isclose(q2.core, 0.75) and np.isclose(q2.near, 0.4)


def test_discarded_frames_artifacts_and_missing_rings():
    movie, act, valid, pr, pc, disc = _trial()
    disc[EVENTS[0] + 3] = True  # inside the first event's window: that event is dropped
    q = ring_event_ratios(movie, act, valid, pr, pc, disc)
    assert q.n_events == 2 * (len(EVENTS) - 1) and np.isclose(q.core, 1.5)
    # a GIAnT fit artifact within 4 px of the second source leaves that source out
    act2 = act.copy()
    act2[:, 22, 46] = 1000.0
    q = ring_event_ratios(movie, act2, valid, pr, pc, disc)
    assert (q.n_sources, q.n_artifact_sources) == (1, 1)
    # a ring that misses one pixel in every frame (in turn) has no usable frame: skipped, no error
    m3 = movie.copy()
    ring = [(20 + dr, 15 + dc) for dr in range(-5, 6) for dc in range(-5, 6) if 3 <= np.hypot(dr, dc) <= 5]
    for t in range(T):
        m3[t, ring[t % len(ring)][0], ring[t % len(ring)][1]] = np.nan
    q = ring_event_ratios(m3, act, valid, pr, pc, disc)
    assert q.n_sources == 1 and np.isclose(q.core, 1.5)
    # no sources at all: NaN ratios
    q = ring_event_ratios(movie, act, valid, pr[2:], pc[2:], disc)
    assert q.n_sources == 0 and np.isnan(q.core) and np.isnan(q.near)


def test_trial_file_divides_by_the_applied_gain(tmp_path):
    movie, act, valid, pr, pc, disc = _trial()
    fn = tmp_path / "trial_001.h5"
    with h5py.File(fn, "w") as f:
        f["movie"] = movie[None].astype(np.float32)
        f["activity"] = (0.6 * act)[None].astype(np.float32)  # an old export: scaled by its gain
        f["mask/valid"] = valid.astype(np.uint8)
        f["frame/discard"] = disc.astype(np.uint8)
        f["sources/peak_row"] = pr.astype(np.int32)
        f["sources/peak_col"] = pc.astype(np.int32)
        f.attrs["activityGain"] = np.array([0.6])
    q = trial_ring_event_ratios(fn)
    assert np.isclose(q.core, 1.5, rtol=1e-5) and np.isclose(q.near, 0.8, rtol=1e-5)


def test_trial_file_uses_the_activity_channel(tmp_path):
    """A two-channel export written with channels [2 1]: the activity channel
    (1) is the file's second array, and the ratios come from it."""
    movie, act, valid, pr, pc, disc = _trial()
    other = np.full_like(movie, 50.0)  # a structural channel with no events
    fn = tmp_path / "trial_001.h5"
    with h5py.File(fn, "w") as f:
        f["movie"] = np.stack([other, movie]).astype(np.float32)
        f["activity"] = np.stack([np.zeros_like(act), act]).astype(np.float32)
        f["mask/valid"] = valid.astype(np.uint8)
        f["frame/discard"] = disc.astype(np.uint8)
        f["sources/peak_row"] = pr.astype(np.int32)
        f["sources/peak_col"] = pc.astype(np.int32)
        f.attrs["channels"] = np.array([2.0, 1.0])
        f.attrs["activityChannel"] = np.array([1.0])
    q = trial_ring_event_ratios(fn)
    assert q.n_sources == 2 and np.isclose(q.core, 1.5) and np.isclose(q.near, 0.8)
    assert trial_ring_event_ratios(fn, channel=0).n_sources == 0  # the other channel, on request
    with h5py.File(fn, "a") as f:
        f.attrs["channels"] = np.array([2.0, 3.0])
    with pytest.raises(ValueError, match="not the activity channel"):
        trial_ring_event_ratios(fn)


def test_folder_scan_skips_other_hdf5_files(tmp_path, capsys):
    """python -m slap2_f0.ringqc <folder> reports the exported trials and skips
    GIAnT's trial_table.h5 next to them."""
    movie, act, valid, pr, pc, disc = _trial()
    d = tmp_path / "S1" / "Path1"
    d.mkdir(parents=True)
    with h5py.File(d / "trial_001.h5", "w") as f:
        f["movie"] = movie[None].astype(np.float32)
        f["activity"] = act[None].astype(np.float32)
        f["mask/valid"] = valid.astype(np.uint8)
        f["frame/discard"] = disc.astype(np.uint8)
        f["sources/peak_row"] = pr.astype(np.int32)
        f["sources/peak_col"] = pc.astype(np.int32)
    with h5py.File(tmp_path / "S1" / "trial_table.h5", "w") as f:  # GIAnT's table: no movie, no masks
        f["filename"] = np.zeros(2)
    main([str(tmp_path)])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and "trial_001.h5" in out[0] and "1.5000 at <= 1.5 px" in out[0]
