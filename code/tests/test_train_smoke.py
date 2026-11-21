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
    cfg = WindowConfig(out_frames=40, out_px=10, t_reach=0, s_reach=0, edge_frac=1.0, edge_frames=5)
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
