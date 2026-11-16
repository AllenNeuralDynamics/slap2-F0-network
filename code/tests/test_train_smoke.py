"""Training runs end to end on small synthetic trials, on simulated and real windows."""

import csv
import json

import h5py
import numpy as np
import pytest

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
