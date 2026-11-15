"""Training runs end to end on small synthetic trials, on simulated and real windows."""

import csv

import h5py
import numpy as np
import pytest

from slap2_f0.cli import main


def _trial(fn, label, seed, T=160, R=20, C=24):
    rng = np.random.default_rng(seed)
    base = 5.0 + rng.random((R, C)).astype(np.float32) * 5
    movie = rng.poisson(np.broadcast_to(base, (T, R, C))).astype(np.float32)[None]
    act = np.zeros_like(movie)
    act[0, 50:55, 8:11, 8:11] = 4.0
    movie += act
    lab = np.zeros((R, C), np.uint8)
    lab[5:14, 5:14] = 1
    sup = np.zeros((R, C), np.uint8)
    sup[6:13, 6:13] = 1
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
        f.attrs["nSources"] = 1


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
        for k in ("loss", "mse_net", "mse_silo", "mse_mean_image", "mse8_net", "z_mean", "r8"):
            assert all(np.isfinite(float(r[f"{name}_{k}"])) for r in rows), (name, k)
    # the untrained network predicts the mean image exactly
    r0 = rows[0]
    assert abs(float(r0["real_mse_net"]) - float(r0["real_mse_mean_image"])) < 1e-6
    # init from a checkpoint (fine-tuning)
    out2 = tmp_path / "out2"
    main(["train", str(tmp_path / "data"), "--out", str(out2), "--data", "real", "--init", str(out / "best.pt"),
          "--steps", "1", "--batch", "2", "--width", "4", "--temporal_dilations", "1,2", "--spatial_dilations", "1,1",
          "--gap", "2", "--window_reach", "6", "--out_frames", "16", "--out_px", "8", "--workers", "0",
          "--val_windows", "2", "--stats_cache", str(out / "stats")])
    assert (out2 / "best.pt").exists()
