"""Training runs end to end on small synthetic trials, on simulated and real windows."""

import csv
import json

import h5py
import numpy as np
import pytest
from dataclasses import replace

from slap2_f0.cli import main


def _trial(fn, label, seed, T=160, R=20, C=24, sources=True):
    rng = np.random.default_rng(seed)
    base = 5.0 + rng.random((R, C)).astype(np.float32) * 5
    movie = rng.poisson(np.broadcast_to(base, (T, R, C))).astype(np.float32)[None]
    act = np.zeros_like(movie)
    lab = np.zeros((R, C), np.uint8)
    lab[5:14, 5:14] = 1
    sup = np.zeros((R, C), np.uint8)
    if sources:  # else a GFP control: no activity, no footprints
        act[0, 50:55, 8:11, 8:11] = 4.0
        sup[6:13, 6:13] = 1
    movie += act
    with h5py.File(fn, "w") as f:
        f["movie"] = movie
        f["activity"] = act
        f["mean_image"] = movie.mean(1)
        f["varfac"] = np.ones((R, C), np.float32)
        f["mask/valid"] = np.ones((R, C), np.uint8)
        f["mask/labeled"] = lab
        f["mask/footprint_support"] = sup
        f["frame/discard"] = np.zeros(T, np.uint8)
        f.attrs["label"] = label
        f.attrs["nSources"] = int(sources)
        f.attrs["alignHz"] = 20.0  # real windows drop real_edge_s (2 s) = 40 frames at each trial edge


@pytest.mark.parametrize("data", ["sim", "real"])
def test_train_smoke(tmp_path, data):
    for k, label in enumerate(["sessA", "sessA", "sessB", "sessC"]):
        d = tmp_path / "data" / label / "Path1"
        d.mkdir(parents=True, exist_ok=True)
        _trial(d / f"trial_{k:03d}.h5", label, k)
    out = tmp_path / "out"
    main(["train", str(tmp_path / "data"), "--out", str(out), "--data", data, "--steps", "3", "--batch", "2",
          "--width", "4", "--temporal_dilations", "1,2", "--spatial_dilations", "1,1", "--gap", "2",
          "--window_reach", "6", "--out_frames", "16", "--out_px", "8", "--workers", "0", "--val_every", "2",
          "--val_windows", "2", "--log_every", "1", "--save_every", "2", "--w_unfitted", "1"])
    assert (out / "best.pt").exists() and (out / "last.pt").exists() and (out / "config.json").exists()
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert [r["step"] for r in rows] == ["0", "2", "3"]  # untrained, then every 2 steps and the last
    for name in ("real", "sim", "tr"):
        for k in ("loss", "mse_net", "mse_silo", "mse_mean_image", "mse8_net", "mse8_oracle", "mse8_oracle_blind",
                  "z_mean", "r8"):
            assert all(np.isfinite(float(r[f"{name}_{k}"])) for r in rows), (name, k)
    # the untrained network predicts the mean image exactly
    r0 = rows[0]
    assert abs(float(r0["real_mse_net"]) - float(r0["real_mse_mean_image"])) < 1e-6
    assert abs(float(r0[f"{data}_balanced"]) - 1.0) < 1e-5  # the checkpoint score starts at 1
    # init from a checkpoint (fine-tuning)
    out2 = tmp_path / "out2"
    main(["train", str(tmp_path / "data"), "--out", str(out2), "--data", "real", "--init", str(out / "best.pt"),
          "--steps", "1", "--batch", "2", "--width", "4", "--temporal_dilations", "1,2", "--spatial_dilations", "1,1",
          "--gap", "2", "--window_reach", "6", "--out_frames", "16", "--out_px", "8", "--workers", "0",
          "--val_windows", "2", "--stats_cache", str(out / "stats")])
    assert (out2 / "best.pt").exists()


def test_evaluate_reproduces_validation(tmp_path):
    for k, label in enumerate(["sessA", "sessA", "sessB", "sessC"]):
        d = tmp_path / "data" / label / "Path1"
        d.mkdir(parents=True, exist_ok=True)
        _trial(d / f"trial_{k:03d}.h5", label, k)
    out = tmp_path / "out"
    main(["train", str(tmp_path / "data"), "--out", str(out), "--data", "sim", "--steps", "2", "--batch", "2",
          "--width", "4", "--temporal_dilations", "1,2", "--spatial_dilations", "1,1", "--gap", "2",
          "--window_reach", "6", "--out_frames", "16", "--out_px", "8", "--workers", "0", "--val_every", "2",
          "--val_windows", "2", "--save_every", "2"])
    ev = tmp_path / "eval"
    main(["evaluate", str(tmp_path / "data"), "--ckpt", f"last={out / 'last.pt'}", "--out", str(ev),
          "--windows", "2", "--workers", "0", "--stats_from", str(out / "stats")])
    rows = list(csv.DictReader(open(ev / "eval.csv")))
    ex = list(csv.DictReader(open(ev / "excess.csv")))  # residual variance over photon noise, per trial
    assert len(ex) == 4 and all(np.isfinite(float(r["all_r1"])) for r in ex)
    assert sorted((r["session"], r["kind"]) for r in rows) == sorted(
        (s, k) for s in ("sessA", "sessB", "sessC") for k in ("real", "sim"))
    # the held-out session's windows are the ones the training validation used
    val = json.loads((out / "config.json").read_text())["split"].split("validation: ")[1].rstrip(")")
    last = list(csv.DictReader(open(out / "metrics.csv")))[-1]
    for kind in ("real", "sim"):
        r = next(r for r in rows if r["session"] == val and r["kind"] == kind)
        for k in ("mse8_net", "mse8_mean_image", "bias_net"):
            assert float(r[k]) == pytest.approx(float(last[f"{kind}_{k}"]), rel=1e-5, abs=1e-7), (kind, k)


def test_evaluate_sourceless_session(tmp_path):
    """A session without sources (a GFP control) trains and scores; its
    labeled pixels are scored (lab_), since its activity is known to be zero,
    while labeled pixels outside footprints elsewhere keep weight 0."""
    for k, label in enumerate(["sessA", "sessB", "sessC", "gfp"]):
        d = tmp_path / "data" / label / "Path1"
        d.mkdir(parents=True, exist_ok=True)
        _trial(d / f"trial_{k:03d}.h5", label, k, sources=label != "gfp")
    out = tmp_path / "out"
    main(["train", str(tmp_path / "data"), "--out", str(out), "--data", "sim", "--steps", "1", "--batch", "2",
          "--width", "4", "--temporal_dilations", "1,2", "--spatial_dilations", "1,1", "--gap", "2",
          "--window_reach", "6", "--out_frames", "16", "--out_px", "8", "--workers", "0", "--val_windows", "2"])
    ev = tmp_path / "eval"
    main(["evaluate", str(tmp_path / "data"), "--ckpt", str(out / "best.pt"), "--out", str(ev),
          "--windows", "4", "--workers", "0", "--stats_from", str(out / "stats")])
    rows = {(r["session"], r["kind"]): r for r in csv.DictReader(open(ev / "eval.csv"))}
    g = rows[("gfp", "real")]
    for k in ("lab_mse8_net", "lab_mse8_mean_image", "lab_bias_net", "bg_mse8_net"):
        assert np.isfinite(float(g[k])), k
    assert not np.isfinite(float(g.get("sup_mse8_net") or "nan"))  # no support pixels
    assert not np.isfinite(float(rows[("sessA", "real")].get("lab_mse8_net") or "nan"))  # w_unfitted 0
    ex = {r["session"]: r for r in csv.DictReader(open(ev / "excess.csv"))}
    assert np.isfinite(float(ex["gfp"]["lab_r1"])) and np.isfinite(float(ex["gfp"]["all_r1"]))


def test_mixed_finetuning_with_activity_free_sims(tmp_path):
    """Mixed real and simulated windows (partly activity-free), a session kept
    out of training, validation on activity-free simulations, and their score
    in evaluate (kind free)."""
    for k, label in enumerate(["sessA", "sessB", "sessC", "sessD", "gfp"]):
        d = tmp_path / "data" / label / "Path1"
        d.mkdir(parents=True, exist_ok=True)
        _trial(d / f"trial_{k:03d}.h5", label, k, sources=label != "gfp")
    out = tmp_path / "out"
    main(["train", str(tmp_path / "data"), "--out", str(out), "--data", "mix", "--sim_frac", "0.5",
          "--silent_trial", "0.3", "--silent_frac", "0,0.8", "--exclude", "gfp", "--val_free",
          "--select", "balanced_free", "--steps", "2", "--batch", "2", "--width", "4", "--temporal_dilations", "1,2",
          "--spatial_dilations", "1,1", "--gap", "2", "--window_reach", "6", "--out_frames", "16", "--out_px", "8",
          "--workers", "0", "--val_windows", "2", "--val_every", "2"])
    cfg = json.loads((out / "config.json").read_text())
    assert not any("gfp" in f for f in cfg["train_files"] + cfg["val_files"])
    assert cfg["sim"]["silent_trial"] == 0.3 and cfg["sim"]["silent_frac"] == [0.0, 0.8]
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    for k in ("free_lab_mse8_net", "free_lab_bias_net", "real_balanced_free"):
        assert all(np.isfinite(float(r[k])) for r in rows), k
    assert abs(float(rows[0]["real_balanced_free"]) - 1.0) < 1e-5  # the untrained network scores 1
    ev = tmp_path / "eval"
    main(["evaluate", str(tmp_path / "data"), "--ckpt", str(out / "best.pt"), "--out", str(ev), "--kinds",
          "real,free", "--windows", "2", "--workers", "0", "--stats_from", str(out / "stats")])
    ev_rows = {(r["session"], r["kind"]): r for r in csv.DictReader(open(ev / "eval.csv"))}
    assert ("gfp", "real") in ev_rows and ("sessA", "free") in ev_rows
    assert np.isfinite(float(ev_rows[("sessA", "free")]["lab_bias_net"]))


def test_activity_subtracted_windows(tmp_path):
    """With subtract_frac 1, a real window's input is movie - activity (the
    target itself) and its whole-trial maps are the residual's."""
    import torch

    from slap2_f0.data import WindowConfig, WindowDataset
    from slap2_f0.stats import load_stats

    d = tmp_path / "data" / "sessA" / "Path1"
    d.mkdir(parents=True)
    fn = d / "trial_000.h5"
    _trial(fn, "sessA", 0)
    stats = load_stats([fn], tmp_path / "stats", workers=0)
    st = stats[str(fn)]
    cfg = WindowConfig(out_frames=60, out_px=12, t_reach=50, s_reach=6, w_unfitted=1.0)
    sub = WindowDataset([fn], cfg, stats, length=4, seed=1, flip=False, subtract_frac=1.0)
    for i in range(4):
        b = sub[i]
        s_ = float(b["scale"])
        ok = torch.isfinite(b["movie"])
        assert torch.allclose(b["movie"][ok] / s_, b["noisy"][ok], atol=1e-5)
        assert torch.equal(b["target"], b["noisy"])
        fin = torch.isfinite(b["mean_image"])
        assert float(b["mean_act"][fin].abs().max()) == 0
        mi = b["mean_image"][fin].numpy()  # values of the residual's mean map
        assert np.isin(np.round(mi, 4), np.round(st["mean_res"].astype(np.float32), 4)).all()


def test_train_box_model_with_accumulation(tmp_path):
    """The box model trains (two micro-batches per step), saves, and evaluate rebuilds it from the checkpoint."""
    for k, label in enumerate(["sessA", "sessB", "sessC"]):
        d = tmp_path / "data" / label / "Path1"
        d.mkdir(parents=True, exist_ok=True)
        _trial(d / f"trial_{k:03d}.h5", label, k)
    out = tmp_path / "out"
    main(["train", str(tmp_path / "data"), "--out", str(out), "--data", "sim", "--steps", "2", "--batch", "1",
          "--accum", "2", "--width", "4", "--temporal_dilations", "1,2", "--spatial_dilations", "1,1", "--gap", "2",
          "--window_reach", "6", "--out_frames", "16", "--out_px", "8", "--workers", "0", "--val_windows", "2",
          "--val_every", "2", "--box_radius", "2", "--hp_width", "4", "--hp_temporal_dilations", "1,2",
          "--hp_spatial_dilations", "1,1", "--hp_full_res_blocks", "1", "--pool_scales", "4,0"])
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model"]["box_radius"] == 2 and cfg["window"]["s_reach"] == 7  # 2 + 1 + 1 + 2 * 1 + 1
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert [r["step"] for r in rows] == ["0", "2"]
    ev = tmp_path / "eval"
    main(["evaluate", str(tmp_path / "data"), "--ckpt", f"box={out / 'best.pt'}", "--out", str(ev), "--windows", "2",
          "--workers", "0", "--stats_from", str(out / "stats")])
    assert len(list(csv.DictReader(open(ev / "eval.csv")))) == 6  # 3 sessions x real, sim


def test_flips_are_up_down_only(tmp_path):
    """The default flip is along rows (up/down, the DMD's short axis) only: an
    item is the unflipped window or its row-flipped copy, never column-flipped."""
    import torch

    from slap2_f0.data import WindowConfig, WindowDataset
    from slap2_f0.stats import load_stats

    d = tmp_path / "data" / "sessA" / "Path1"
    d.mkdir(parents=True)
    fn = d / "trial_000.h5"
    _trial(fn, "sessA", 0)
    stats = load_stats([fn], tmp_path / "stats", workers=0)
    cfg = WindowConfig(out_frames=40, out_px=10, t_reach=20, s_reach=4)
    plain = WindowDataset([fn], cfg, stats, length=40, seed=2, flip="none")
    rows = WindowDataset([fn], cfg, stats, length=40, seed=2, flip="rows")
    n_flipped = 0
    for i in range(40):
        a, b = plain[i]["movie"].nan_to_num(-1), rows[i]["movie"].nan_to_num(-1)
        same, ud = torch.equal(a, b), torch.equal(a.flip(1), b)
        assert same or ud
        n_flipped += ud and not same
        assert not (torch.equal(a.flip(2), b) and not same and not ud)
    assert 5 < n_flipped < 35


def test_event_mask(tmp_path):
    """With event_mask, real-window voxels with activity, and event_margin
    frames around them, weigh 0; everything else is unchanged."""
    from slap2_f0.data import WindowConfig, WindowDataset
    from slap2_f0.stats import load_stats

    d = tmp_path / "data" / "sessA" / "Path1"
    d.mkdir(parents=True)
    fn = d / "trial_000.h5"
    _trial(fn, "sessA", 0)  # activity 4.0 in frames 50-54 at rows/cols 8-10
    stats = load_stats([fn], tmp_path / "stats", workers=0)
    cfg = WindowConfig(out_frames=160, out_px=24, t_reach=0, s_reach=0, w_unfitted=1.0)
    from slap2_f0.data import _dilate_time

    for seed in range(50):  # a window that holds the event
        plain = WindowDataset([fn], cfg, stats, length=1, seed=seed, flip="none")[0]
        act_at = (plain["target"] != plain["movie"] / plain["scale"]) & (plain["weight"] > 0)  # activity subtracted
        if act_at.any():
            break
    assert act_at.any()
    masked = WindowDataset([fn], replace(cfg, event_mask=0.5, event_margin=3), stats, length=1, seed=seed,
                           flip="none")[0]
    zero = _dilate_time(act_at.numpy(), 3)
    assert (masked["weight"].numpy()[zero] == 0).all()
    assert (masked["weight"].numpy()[~zero] == plain["weight"].numpy()[~zero]).all()
    assert (masked["weight"][:, 0, 0] == plain["weight"][:, 0, 0]).all()  # no activity there: unchanged


def test_sim_event_weight(tmp_path):
    """Simulated windows weigh sim_event_weight during events, 1 elsewhere."""
    from slap2_f0.data import WindowConfig, WindowDataset
    from slap2_f0.simulate import SimConfig
    from slap2_f0.stats import load_stats

    d = tmp_path / "data" / "sessA" / "Path1"
    d.mkdir(parents=True)
    fn = d / "trial_000.h5"
    _trial(fn, "sessA", 0)
    stats = load_stats([fn], tmp_path / "stats", workers=0)
    cfg = WindowConfig(out_frames=160, out_px=24, t_reach=0, s_reach=0, sim_event_weight=5.0, event_margin=2)
    seen = False
    for i in range(30):
        b = WindowDataset([fn], cfg, stats, length=30, seed=0, sim=SimConfig(), flip="none")[i]
        vals = set(np.unique(b["weight"].numpy()).tolist())
        assert vals <= {0.0, 1.0, 5.0}
        seen |= 5.0 in vals
    assert seen


def test_edge_windows(tmp_path):
    """With edge_frac 1, every window's loss region starts at the trial's first
    frame or ends at its last, within edge_frames."""
    from slap2_f0.data import WindowConfig, WindowDataset
    from slap2_f0.stats import load_stats

    d = tmp_path / "data" / "sessA" / "Path1"
    d.mkdir(parents=True)
    fn = d / "trial_000.h5"
    T = 400
    _trial(fn, "sessA", 0, T=T)
    with h5py.File(fn, "a") as f:  # the movie holds its frame number, so a window shows which frames it covers
        f["movie"][...] = np.broadcast_to(np.arange(T, dtype=np.float32)[None, :, None, None] + 1.0, f["movie"].shape)
    stats = load_stats([fn], tmp_path / "stats", workers=0)
    cfg = WindowConfig(out_frames=40, out_px=10, t_reach=0, s_reach=0, edge_frac=1.0, edge_frames=5,
                       real_edge_s=0)  # real windows keep their edges, so they are placed there too
    ds = WindowDataset([fn], cfg, stats, length=60, seed=3, flip="none")
    for i in range(60):
        m = ds[i]["movie"].numpy()
        fr = np.nanmax(m.reshape(m.shape[0], -1), 1)  # frame number + 1 per window frame, NaN outside the trial
        first, last = np.nanmin(fr) - 1, np.nanmax(fr) - 1
        assert first <= 5 or last >= T - 6, (first, last)


def test_open_trial_cache_is_bounded(tmp_path):
    """A worker keeps at most max_open trials (and their templates) open,
    closes the ones it drops, and draws the same windows as without a limit."""
    from slap2_f0.data import WindowConfig, WindowDataset
    from slap2_f0.simulate import SimConfig
    from slap2_f0.stats import load_stats

    fns = []
    for k in range(4):
        d = tmp_path / "data" / f"sess{k}" / "Path1"
        d.mkdir(parents=True)
        fns.append(d / "trial_000.h5")
        _trial(fns[-1], f"sess{k}", k)
    stats = load_stats(fns, tmp_path / "stats", workers=0)
    cfg = WindowConfig(out_frames=40, out_px=10, t_reach=0, s_reach=0)
    for sim in (None, SimConfig()):
        lim = WindowDataset(fns, cfg, stats, length=40, seed=5, sim=sim, flip="none", max_open=2)
        unl = WindowDataset(fns, cfg, stats, length=40, seed=5, sim=sim, flip="none", max_open=0)
        handles = {}
        for i in range(40):
            a, b = lim[i], unl[i]
            for k in a:
                assert np.array_equal(a[k].numpy(), b[k].numpy(), equal_nan=True), k
            assert len(lim._open) <= 2 and set(lim._tpl) <= set(lim._open)
            handles.update({fn: tf._f for fn, tf in lim._open.items()})
        assert len(unl._open) == 4
        closed = [fn for fn, f in handles.items() if fn not in lim._open]
        assert closed and all(not handles[fn].id.valid for fn in closed)


def test_one_channel_mean_image(tmp_path):
    """One-channel exports store the mean image as (row, col): it is read
    whole, its stats cover every valid pixel, and a sidecar made before the
    fix (first row only, no mean_image_axes) is redone."""
    from slap2_f0.io import TrialFile
    from slap2_f0.stats import load_stats, sidecar_path, trial_stats

    d = tmp_path / "data" / "sessA" / "Path1"
    d.mkdir(parents=True)
    two, one = d / "trial_000.h5", d / "trial_001.h5"
    _trial(two, "sessA", 0)
    _trial(one, "sessA", 0)
    with h5py.File(one, "a") as f:
        mi = f["mean_image"][0]
        mi[0, :5] = np.nan  # the first row is not all imaged, as in a cropped canvas
        del f["mean_image"]
        f["mean_image"] = mi
    with TrialFile(one) as tf:
        assert tf.mean_image.shape == (20, 24)
    s1, s2 = trial_stats(one), trial_stats(two)
    assert s1["mean_image_axes"] == 2 and s2["mean_image_axes"] == 3
    assert (s1["n_valid"][1:] > 0).all()  # rows below the first are covered
    stale = {k: v for k, v in s1.items() if k != "mean_image_axes"}
    stale["n_valid"] = np.zeros_like(stale["n_valid"])
    np.savez(sidecar_path(one), file_size=one.stat().st_size, **stale)
    np.savez(sidecar_path(two), file_size=two.stat().st_size, **{k: v for k, v in s2.items() if k != "mean_image_axes"})
    st = load_stats([one, two], None, workers=0)
    assert st[str(one)]["mean_image_axes"] == 2 and (st[str(one)]["n_valid"][1:] > 0).all()  # redone
    assert "mean_image_axes" not in st[str(two)]  # a two-channel trial's sidecar stays valid


def test_activity_fit_artifacts_are_left_out(tmp_path):
    """Pixels where GIAnT's mean activity is far above the movie (a fit
    artifact, as on 803121 path 2) and their neighbours: no activity and no
    excess noise in the simulator's maps, no loss weight in real windows."""
    from slap2_f0.data import WindowConfig, WindowDataset
    from slap2_f0.simulate import SimConfig, excess_maps, noise_kernels
    from slap2_f0.stats import load_stats, trial_stats

    d = tmp_path / "data" / "sessA" / "Path1"
    d.mkdir(parents=True)
    fn = d / "trial_000.h5"
    _trial(fn, "sessA", 0, T=800)  # 64-frame variances need at least 8 blocks
    with h5py.File(fn, "a") as f:
        a = f["activity"][...]
        t = np.arange(a.shape[1])
        a[0, :, 2, 2] = 600 + 900 * np.sin(2 * np.pi * t / 300)  # ~100x the movie's 5-10 counts, not in the movie
        f["activity"][...] = a
    raw = trial_stats(fn)
    stats = load_stats([fn], tmp_path / "stats", workers=0)
    st = stats[str(fn)]
    bad = st["unfit"]
    assert bad[2, 2] and bad[1:4, 1:4].all() and bad.sum() == 9
    assert st["mean_act"][2, 2] == 0 and raw["mean_act"][2, 2] > 100
    kt, valid = noise_kernels(raw)[0], raw["n_valid"] > 0
    _, slow_unc = excess_maps(raw, kt, valid, SimConfig(slow_max_sd=np.inf))  # what S0-S3's simulator drew
    _, slow_cap = excess_maps(raw, kt, valid)  # capped at the pixel's mean
    _, slow = excess_maps(st, kt, valid)
    assert slow_unc[2, 2] > 1000 and slow_cap[2, 2] <= raw["mean_movie"][2, 2] ** 2 + 1e-3
    assert slow[1:4, 1:4].max() == 0
    cfg = WindowConfig(out_frames=800, out_px=48, t_reach=0, s_reach=0)  # every window holds the whole 20 x 24 trial
    ds = WindowDataset([fn], cfg, stats, length=6, seed=0, flip="none")
    ds_raw = WindowDataset([fn], cfg, {str(fn): raw}, length=6, seed=0, flip="none")  # the same windows, raw stats
    for i in range(6):
        w, w_raw = ds[i]["weight"].numpy(), ds_raw[i]["weight"].numpy()
        lost = ((w_raw > 0) & (w == 0)).any(0)
        r, c = np.nonzero(lost)
        assert lost.sum() == 9 and np.ptp(r) == 2 and np.ptp(c) == 2  # the 3 x 3 block, nothing else
        assert ((w > 0) == (w_raw > 0))[:, ~lost].all()
    # simulations put no activity there either: the template's per-frame activity at the artifact
    # went into the simulated movie (804730 path 2: up to 3e5 counts) while the inputs said ~10
    from slap2_f0.io import TrialFile
    from slap2_f0.simulate import Template, draw, window
    T, R, C = raw["n_frames"], *bad.shape
    with TrialFile(fn) as tf:
        sims = {}
        for name, s in (("sanitized", st), ("raw", raw)):
            tp = Template(tf, s)
            rng = np.random.default_rng(5)
            sims[name] = window(tp, draw(tp, SimConfig(), rng), 0, int(T), 0, R, 0, C, rng)
    assert sims["raw"]["activity"][:, 2, 2].max() > 100  # what the simulator did before
    assert sims["sanitized"]["activity"][:, bad].max() == 0
    assert sims["sanitized"]["activity"][:, ~bad].max() > 0  # the other sources keep theirs
    assert np.nanmax(sims["sanitized"]["movie"][:, bad]) < 10 * np.nanmax(raw["mean_movie"])
    # real windows with the residual as input (subtract_frac) show the movie there, not the residual
    sub = WindowDataset([fn], cfg, stats, length=3, seed=0, flip="none", sim_frac=0.0, subtract_frac=1.0)
    for i in range(3):  # each window holds the whole trial, artifact included
        b = sub[i]
        assert float(b["target"].min() * b["scale"]) < -100  # the residual there is hugely negative
        assert np.nanmin(b["movie"].numpy()) > -50  # the input is not


def test_real_windows_drop_trial_edges(tmp_path):
    """Real windows weigh 0 within real_edge_s of a trial's first or last frame
    (GIAnT's baseline is biased there) and are not placed at trial edges;
    simulated windows keep both."""
    from slap2_f0.data import WindowConfig, WindowDataset
    from slap2_f0.simulate import SimConfig
    from slap2_f0.stats import load_stats

    d = tmp_path / "data" / "sessA" / "Path1"
    d.mkdir(parents=True)
    fn = d / "trial_000.h5"
    T = 400
    _trial(fn, "sessA", 0, T=T)  # alignHz 20: the 2 s edges are 40 frames
    with h5py.File(fn, "a") as f:  # the movie holds its frame number
        f["movie"][...] = np.broadcast_to(np.arange(T, dtype=np.float32)[None, :, None, None] + 1.0, f["movie"].shape)
    stats = load_stats([fn], tmp_path / "stats", workers=0)
    cfg = WindowConfig(out_frames=60, out_px=10, t_reach=0, s_reach=0, edge_frac=1.0, edge_frames=5)
    real = WindowDataset([fn], cfg, stats, length=40, seed=3, flip="none")
    near_edge = 0
    for i in range(40):
        b = real[i]
        m = b["movie"].numpy()
        fr = np.nanmax(m.reshape(m.shape[0], -1), 1) - 1  # trial frame of each window frame (NaN outside)
        wf = b["weight"].numpy().reshape(m.shape[0], -1).max(1)  # does the frame carry any loss weight?
        inside = np.isfinite(fr)
        edge = inside & ((fr < 40) | (fr >= T - 40))
        assert (wf[edge] == 0).all() and (wf[inside & ~edge] > 0).all()
        near_edge += np.nanmin(fr) <= 5 or np.nanmax(fr) >= T - 6
    assert near_edge < 20  # placed anywhere, not at the edges (with edge_frac 1, every window would be)
    sim = WindowDataset([fn], cfg, stats, length=20, seed=3, sim=SimConfig(), flip="none")
    for i in range(20):  # simulated windows sit at an edge (at most 5 frames outside the trial), edges weighted
        wf = sim[i]["weight"].numpy().reshape(60, -1).max(1)
        assert (wf > 0).sum() >= 55


def test_train_with_motion_simulation_and_features(tmp_path):
    """Simulated residual motion, focus and local motion in training, the motion
    validation set and evaluation kind, and the motion features: it trains,
    selects on motion windows, and evaluate rebuilds the model."""
    for k, label in enumerate(["sessA", "sessB", "sessC"]):
        d = tmp_path / "data" / label / "Path1"
        d.mkdir(parents=True, exist_ok=True)
        _trial(d / f"trial_{k:03d}.h5", label, k)
    out = tmp_path / "out"
    main(["train", str(tmp_path / "data"), "--out", str(out), "--data", "sim", "--steps", "2", "--batch", "1",
          "--width", "4", "--temporal_dilations", "1,2", "--spatial_dilations", "1,1", "--gap", "2",
          "--window_reach", "6", "--out_frames", "16", "--out_px", "8", "--workers", "0", "--val_windows", "2",
          "--val_every", "2", "--box_radius", "2", "--hp_width", "4", "--hp_temporal_dilations", "1,2",
          "--hp_spatial_dilations", "1,1", "--hp_full_res_blocks", "1", "--pool_scales", "4,0",
          "--motion_sd", "0,0.2", "--focus_sd", "0,0.25", "--local_sd", "0,0.2", "--val_motion",
          "--select_on", "motion", "--motion_features"])
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model"]["motion_features"] and cfg["sim"]["local_sd_px"] == [0.0, 0.2]
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert any(k.startswith("motion_") for k in rows[0])
    ev = tmp_path / "eval"
    main(["evaluate", str(tmp_path / "data"), "--ckpt", f"m={out / 'best.pt'}", "--out", str(ev), "--windows", "2",
          "--workers", "0", "--kinds", "motion", "--stats_from", str(out / "stats")])
    assert len(list(csv.DictReader(open(ev / "eval.csv")))) == 3


def test_stats_sidecars(tmp_path):
    """`slap2-f0 stats` stores each trial's stats next to it; training finds them
    there wherever the data are mounted, and ignores them if the file changed."""
    from slap2_f0.stats import load_stats, sidecar_path

    d = tmp_path / "export" / "sessA" / "Path1"
    d.mkdir(parents=True)
    fn = d / "trial_001.h5"
    _trial(fn, "sessA", 0)
    main(["stats", str(tmp_path / "export"), "--workers", "0"])
    side = sidecar_path(fn)
    assert side.exists()
    ref = load_stats([fn])[str(fn)]
    moved = tmp_path / "mounted" / "sessA" / "Path1"  # e.g. a data asset at another path
    moved.mkdir(parents=True)
    for p in (fn, side):
        (moved / p.name).write_bytes(p.read_bytes())
    fn2 = moved / fn.name
    side_mtime = sidecar_path(fn2).stat().st_mtime
    got = load_stats([fn2], tmp_path / "cache")[str(fn2)]
    assert not (tmp_path / "cache").exists()  # read from the sidecar, nothing computed
    assert sidecar_path(fn2).stat().st_mtime == side_mtime
    assert set(got) == set(ref) and all(np.allclose(got[k], ref[k], equal_nan=True) for k in ref)
    with h5py.File(fn2, "a") as f:  # a changed file: the sidecar no longer applies
        f["extra"] = np.zeros(1000)
    load_stats([fn2], tmp_path / "cache")
    assert any((tmp_path / "cache").glob("*.npz"))


def test_named_sessions_caps_and_kept_steps(tmp_path):
    """--val_sessions/--train_sessions pick the sessions, --max_train_s keeps the first
    trials of each session and path, --keep_every keeps the model at every N steps, and
    evaluate scores all kept steps from a glob."""
    for k, (label, path) in enumerate([("sessA", 1), ("sessA", 1), ("sessA", 2), ("sessB", 1), ("sessC", 1)]):
        d = tmp_path / "data" / label / f"Path{path}"
        d.mkdir(parents=True, exist_ok=True)
        _trial(d / f"trial_{k:03d}.h5", label, k)  # 160 frames = 2 s each
    out = tmp_path / "out"
    small = ["--width", "4", "--temporal_dilations", "1,2", "--spatial_dilations", "1,1", "--gap", "2",
             "--window_reach", "6", "--out_frames", "16", "--out_px", "8", "--workers", "0"]
    main(["train", str(tmp_path / "data"), "--out", str(out), "--data", "real", "--steps", "3", "--batch", "2",
          "--val_sessions", "sessB", "--train_sessions", "sessA", "--max_train_s", "1", "--keep_every", "2",
          "--val_every", "2", "--val_windows", "2"] + small)
    cfg = json.load(open(out / "config.json"))
    names = sorted(p.split("data")[-1].replace("\\", "/") for p in cfg["train_files"])
    assert names == ["/sessA/Path1/trial_000.h5", "/sessA/Path2/trial_002.h5"]  # first trial of each path
    assert [p.split("data")[-1].replace("\\", "/") for p in cfg["val_files"]] == ["/sessB/Path1/trial_003.h5"]
    assert abs(cfg["train_seconds"]["sessA"] - 4.0) < 1e-6
    steps = sorted(p.name for p in (out / "steps").glob("*.pt"))
    assert steps == ["step_000000.pt", "step_000002.pt", "step_000003.pt"]
    main(["evaluate", str(tmp_path / "data"), "--out", str(tmp_path / "eval"), "--ckpt",
          f"run={out / 'steps' / 'step_*.pt'}", "--kinds", "real", "--windows", "2", "--workers", "0",
          "--per_window"])
    rows = list(csv.DictReader(open(tmp_path / "eval" / "eval.csv")))
    win = list(csv.DictReader(open(tmp_path / "eval" / "windows.csv")))
    assert len(win) == 3 * 3 * 2  # checkpoints x sessions x windows
    for r in rows:  # the window sums add up to the session's score
        w = [q for q in win if q["checkpoint"] == r["checkpoint"] and q["session"] == r["session"]]
        for c in ("sup_", "bg_"):
            n = sum(float(q[f"{c}n8"]) for q in w)
            if n > 0:
                for b_ in ("net", "mean_image"):
                    se = sum(float(q[f"{c}se8_{b_}"]) for q in w)
                    assert abs(se / n - float(r[f"{c}mse8_{b_}"])) < 1e-4 * max(1.0, se / n)
    assert sorted({r["checkpoint"] for r in rows}) == ["run@step_000000", "run@step_000002", "run@step_000003"]
    with pytest.raises(SystemExit):  # a session in both roles
        main(["train", str(tmp_path / "data"), "--out", str(tmp_path / "bad"), "--steps", "1",
              "--val_sessions", "sessA", "--train_sessions", "sessA"] + small)


def test_selection_per_mouse_and_weighted(tmp_path):
    """Validation scores each validation mouse on its own --val_windows windows
    (the first windows held-out evaluation draws for that session); --select_on
    with weights combines the sets; --val_pooled restores pooled windows."""
    from slap2_f0.train import parse_select_on

    assert parse_select_on("real") == {"real": 1.0}
    assert parse_select_on("real:1, fast:0.5") == {"real": 1.0, "fast": 0.5}
    for k, label in enumerate(["sessA", "sessA", "sessB", "sessC", "sessD"]):
        d = tmp_path / "data" / label / "Path1"
        d.mkdir(parents=True, exist_ok=True)
        _trial(d / f"trial_{k:03d}.h5", label, k)
    small = ["--steps", "2", "--batch", "1", "--width", "4", "--temporal_dilations", "1,2", "--spatial_dilations", "1,1",
             "--gap", "2", "--window_reach", "6", "--out_frames", "16", "--out_px", "8", "--workers", "0",
             "--val_every", "2", "--val_windows", "2", "--save_every", "2", "--data", "sim", "--val_fast",
             "--val_sessions", "sessB,sessC", "--train_sessions", "sessA,sessD", "--select_on", "real:1,fast:2"]
    out = tmp_path / "out"
    main(["train", str(tmp_path / "data"), "--out", str(out)] + small)
    last = list(csv.DictReader(open(out / "metrics.csv")))[-1]
    per = {(k, s): float(last[f"{k}@{s}_balanced"]) for k in ("real", "fast") for s in ("sessB", "sessC")}
    want = (np.mean([per["real", "sessB"], per["real", "sessC"]]) + 2 * np.mean([per["fast", "sessB"], per["fast", "sessC"]])) / 3
    assert float(last["real+fast_balanced"]) == pytest.approx(want, rel=1e-6)
    # each mouse's validation windows are evaluation's first windows of that session
    ev = tmp_path / "eval"
    main(["evaluate", str(tmp_path / "data"), "--ckpt", f"last={out / 'last.pt'}", "--out", str(ev), "--windows", "2",
          "--workers", "0", "--kinds", "real", "--stats_from", str(out / "stats")])
    for r in csv.DictReader(open(ev / "eval.csv")):
        if r["session"] in ("sessB", "sessC"):
            bal = np.mean([float(r["sup_mse8_net"]) / float(r["sup_mse8_mean_image"]),
                           float(r["bg_mse8_net"]) / float(r["bg_mse8_mean_image"])])
            assert bal == pytest.approx(per["real", r["session"]], rel=1e-5)
    out2 = tmp_path / "out2"
    main(["train", str(tmp_path / "data"), "--out", str(out2), "--val_pooled", "--stats_cache", str(out / "stats")] + small)
    cols = next(csv.reader(open(out2 / "metrics.csv")))
    assert "real+fast_balanced" in cols and not any("@" in c for c in cols)


def test_stats_overlay_folder(tmp_path, monkeypatch):
    """`slap2-f0 stats ROOT --out DIR` writes sidecars into an overlay folder
    (<session>/<path>/...) instead of next to the trials; load_stats finds them
    there (sidecar_dirs, train/evaluate --sidecars) without recomputing, and
    ignores them when the trial's size no longer matches."""
    import slap2_f0.stats as st_mod

    d = tmp_path / "data" / "mnt" / "sessA" / "Path2"
    d.mkdir(parents=True)
    fn = d / "trial_003.h5"
    _trial(fn, "sessA", 0)
    ov = tmp_path / "overlay"
    main(["stats", str(tmp_path / "data"), "--out", str(ov), "--workers", "0"])
    assert not st_mod.sidecar_path(fn).exists()
    assert (ov / "sessA" / "Path2" / "trial_003_stats_v5.npz").exists()
    ref = st_mod.load_stats([fn], None, workers=0, sidecar_dirs=str(ov))[str(fn)]

    def boom(*a, **k):
        raise AssertionError("recomputed")

    monkeypatch.setattr(st_mod, "trial_stats", boom)
    again = st_mod.load_stats([fn], None, workers=0, sidecar_dirs=[str(ov)])[str(fn)]
    assert again["gain"] == ref["gain"] and np.array_equal(again["mean_movie"], ref["mean_movie"], equal_nan=True)
    with h5py.File(fn, "a") as f:  # the trial changed: the overlay no longer applies
        f.attrs["note"] = "x" * 10000
    with pytest.raises(AssertionError, match="recomputed"):
        st_mod.load_stats([fn], None, workers=0, sidecar_dirs=[str(ov)])


def test_edge_features_finetune_from_checkpoint(tmp_path):
    """--edge_features: a box checkpoint without them warm-starts the model (its
    step-0 validation is the checkpoint's own), it trains on edge windows, and
    evaluate and predict rebuild the model from the checkpoint."""
    for k, label in enumerate(["sessA", "sessB", "sessC"]):
        d = tmp_path / "data" / label / "Path1"
        d.mkdir(parents=True, exist_ok=True)
        _trial(d / f"trial_{k:03d}.h5", label, k)
    arch = ["--width", "4", "--temporal_dilations", "1,2", "--spatial_dilations", "1,1", "--gap", "2",
            "--window_reach", "6", "--box_radius", "2", "--hp_width", "4", "--hp_temporal_dilations", "1,2",
            "--hp_spatial_dilations", "1,1", "--hp_full_res_blocks", "1", "--pool_scales", "4,0"]
    common = ["--data", "sim", "--batch", "1", "--out_frames", "16", "--out_px", "8", "--workers", "0",
              "--val_windows", "2", "--val_every", "2", "--edge_frac", "1"]
    pre = tmp_path / "pre"
    main(["train", str(tmp_path / "data"), "--out", str(pre), "--steps", "2"] + arch + common)
    out = tmp_path / "edge"
    main(["train", str(tmp_path / "data"), "--out", str(out), "--steps", "2", "--init", str(pre / "last.pt"),
          "--edge_features", "--stats_cache", str(pre / "stats")] + arch + common)
    assert json.loads((out / "config.json").read_text())["model"]["edge_features"]
    last = list(csv.DictReader(open(pre / "metrics.csv")))[-1]
    first = list(csv.DictReader(open(out / "metrics.csv")))[0]
    assert np.isclose(float(first["sim_mse_net"]), float(last["sim_mse_net"]), rtol=1e-5)  # starts as the checkpoint
    ev = tmp_path / "eval"
    main(["evaluate", str(tmp_path / "data"), "--ckpt", f"m={out / 'best.pt'}", "--out", str(ev), "--windows", "2",
          "--workers", "0", "--kinds", "sim", "--stats_from", str(pre / "stats")])
    assert len(list(csv.DictReader(open(ev / "eval.csv")))) == 3


SMALL = ["--width", "4", "--temporal_dilations", "1,2", "--spatial_dilations", "1,1", "--gap", "2",
         "--window_reach", "6", "--out_frames", "16", "--out_px", "8", "--workers", "0"]


def _three_sessions(tmp_path):
    for k, label in enumerate(["sessA", "sessA", "sessB", "sessC"]):
        d = tmp_path / "data" / label / "Path1"
        d.mkdir(parents=True, exist_ok=True)
        _trial(d / f"trial_{k:03d}.h5", label, k)
    return tmp_path / "data"


def _load(p):
    import torch
    return torch.load(p, map_location="cpu", weights_only=False)


def test_ema_and_l2sp(tmp_path):
    """--ema saves the moving average as the model (raw weights as model_raw) and validates it;
    --l2sp pulls the weights back toward --init (with lr x l2sp >= 1 they stay there)."""
    import torch
    data = _three_sessions(tmp_path)
    out = tmp_path / "ema"
    main(["train", str(data), "--out", str(out), "--data", "sim", "--steps", "3", "--batch", "2", "--keep_every", "1",
          "--val_every", "1", "--val_windows", "2", "--ema", "0.5"] + SMALL)
    kept = [_load(out / "steps" / f"step_{s:06d}.pt") for s in range(4)]
    for k in kept[0]["model"]:
        assert torch.equal(kept[0]["model"][k], kept[0]["model_raw"][k])  # the average starts at the initial weights
    moved = False
    for prev, cur in zip(kept, kept[1:]):
        for k, v in cur["model"].items():
            want = 0.5 * prev["model"][k] + 0.5 * cur["model_raw"][k]
            assert torch.allclose(v, want, atol=1e-6), k
            moved |= not torch.allclose(cur["model_raw"][k], prev["model_raw"][k])
    assert moved  # the raw weights trained
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert [r["step"] for r in rows] == ["0", "1", "2", "3"]
    assert abs(float(rows[0]["sim_balanced"]) - 1.0) < 1e-5
    best = _load(out / "best.pt")
    assert "model_raw" in best and "model_raw" in _load(out / "last.pt")
    # --l2sp needs --init; with lr x l2sp >= 1 every step lands back on the initial weights
    for bad in (["--l2sp", "1"], ["--ema", "1"], ["--ema", "nan"], ["--l2sp", "-1", "--init", str(out / "last.pt")]):
        with pytest.raises(SystemExit):  # no --init, or a value outside the regularizer's domain
            main(["train", str(data), "--out", str(tmp_path / "bad"), "--steps", "1"] + bad + SMALL)
    out2 = tmp_path / "l2sp"
    main(["train", str(data), "--out", str(out2), "--data", "real", "--init", str(out / "last.pt"), "--steps", "2",
          "--batch", "2", "--keep_every", "1", "--val_windows", "2", "--l2sp", "1e12",
          "--stats_cache", str(out / "stats")] + SMALL)
    init = _load(out / "last.pt")["model"]
    for s in (1, 2):
        for k, v in _load(out2 / "steps" / f"step_{s:06d}.pt")["model"].items():
            assert torch.allclose(v, init[k], atol=1e-6), (s, k)


def test_average_checkpoints(tmp_path):
    """average: weighted means of kept steps (globs, path=weight), loadable by evaluate;
    configurations must match."""
    import torch
    data = _three_sessions(tmp_path)
    out = tmp_path / "run"
    main(["train", str(data), "--out", str(out), "--data", "sim", "--steps", "2", "--batch", "2", "--keep_every", "1",
          "--val_windows", "2"] + SMALL)
    st = [_load(out / "steps" / f"step_{s:06d}.pt")["model"] for s in range(3)]
    main(["average", str(tmp_path / "soup.pt"), str(out / "steps" / "step_*.pt")])
    soup = _load(tmp_path / "soup.pt")
    assert [a["step"] for a in soup["averaged"]] == [0, 1, 2]
    for k, v in soup["model"].items():
        assert torch.allclose(v, (st[0][k] + st[1][k] + st[2][k]) / 3, atol=1e-6)
    main(["average", str(tmp_path / "wise.pt"), f"{out / 'steps' / 'step_000000.pt'}=1",
          f"{out / 'steps' / 'step_000002.pt'}=3"])
    for k, v in _load(tmp_path / "wise.pt")["model"].items():
        assert torch.allclose(v, 0.25 * st[0][k] + 0.75 * st[2][k], atol=1e-6)
    main(["average", str(tmp_path / "dirglob.pt"), str(tmp_path / "r?n" / "steps" / "step_00000[12].pt")])
    assert [a["step"] for a in _load(tmp_path / "dirglob.pt")["averaged"]] == [1, 2]  # wildcards in folder names
    for w in ("nan", "inf", "0"):
        with pytest.raises(ValueError):  # weights must be finite with a positive sum
            main(["average", str(tmp_path / "bad.pt"), f"{out / 'steps' / 'step_000000.pt'}={w}"])
    main(["evaluate", str(data), "--out", str(tmp_path / "eval"), "--ckpt", f"wise={tmp_path / 'wise.pt'}",
          "--kinds", "real", "--windows", "2", "--workers", "0"])
    assert len(list(csv.DictReader(open(tmp_path / "eval" / "eval.csv")))) == 3  # one row per session
    other = tmp_path / "other"
    main(["train", str(data), "--out", str(other), "--data", "sim", "--steps", "1", "--batch", "2",
          "--val_windows", "2", "--stats_cache", str(out / "stats"), "--width", "6"] + SMALL[2:])
    with pytest.raises(SystemExit):  # another width
        main(["average", str(tmp_path / "bad.pt"), str(out / "best.pt"), str(other / "best.pt")])


def test_loader_wait_accumulates_and_resets():
    """LoaderWait: waits add up over micro-batches; report() returns (wait,
    interval) and starts a new interval; validation between got() and idle()
    is interval time, not wait."""
    from slap2_f0.train import LoaderWait

    t = iter([0.0, 2.0, 5.0, 6.0, 8.0, 20.0, 21.0, 22.0])
    w = LoaderWait(clock=lambda: next(t))  # t0 = 0
    w.got()        # 2: first batch after 2 s of waiting
    w.idle()       # 5: forward and backward took 3 s
    w.got()        # 6: second micro-batch after 1 s
    assert w.report() == (3.0, 8.0)  # 8: step logged
    w.idle()       # 20: validation ran from 8 to 20
    w.got()        # 21
    assert w.report() == (1.0, 14.0)  # 22: only the 1 s fetch is wait


def test_step_profiler_prints_a_summary(capsys):
    """--profile_steps: StepProfiler profiles steps start..start+n and prints
    the wall time, a table and the compile counters once."""
    import torch

    from slap2_f0.train import StepProfiler

    p = StepProfiler(2, start=1, device="cpu")
    x = torch.randn(64, 64)
    for step in range(1, 5):
        p.step(step)
        (x @ x).sum()
    out = capsys.readouterr().out
    assert out.count("profile of 2 steps") == 1 and "torch.compile counters" in out
    assert StepProfiler(0).step(20) is None  # off: nothing happens
