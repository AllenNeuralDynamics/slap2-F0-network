"""Full-trial prediction (tiling does not change the output) and the viewer's data paths."""

import json
import threading
import urllib.request
from dataclasses import asdict

import h5py
import numpy as np
import pytest
import torch

from slap2_f0.cli import main
from slap2_f0.data import WindowConfig
from slap2_f0.io import TrialFile
from slap2_f0.model import ModelConfig, SlabBlindSpotNet
from slap2_f0.predict import predict_trial, read_valid, silo_trial, valid_index
from slap2_f0.stats import load_stats
from slap2_f0.train import silo_baseline
from slap2_f0.view import make_server
from test_blindspot import _activate_box


def _trial(fn, seed, T=160, R=22, C=26, sources=True):
    """A trial with an unimaged corner, a discarded frame and one event
    (trial stats need more than 100 frames)."""
    rng = np.random.default_rng(seed)
    base = 5.0 + rng.random((R, C)).astype(np.float32) * 5
    drift = 1 + 0.1 * np.sin(np.arange(T) / 9.0)[:, None, None]
    movie = rng.poisson(base * drift).astype(np.float32)[None]
    act = np.zeros_like(movie)
    if sources:
        act[0, 30:36, 8:12, 8:12] = 6.0
    movie += act
    valid = np.ones((R, C), np.uint8)
    valid[:5, :7] = 0
    discard = np.zeros(T, np.uint8)
    discard[40] = 1
    sup = np.zeros((R, C), np.uint8)
    sup[7:13, 7:13] = 1
    with h5py.File(fn, "w") as f:
        f["movie"] = movie
        f["activity"] = act
        f["mean_image"] = movie.mean(1)
        vf = (0.2 + 0.4 * rng.random((R, C))).astype(np.float32)
        vf[:5, :7] = 0  # no factor where nothing was imaged
        f["varfac"] = vf
        f["mask/valid"] = valid
        f["mask/labeled"] = sup
        f["mask/footprint_support"] = sup
        f["frame/discard"] = discard
        f.attrs["nSources"] = int(sources)


def _model(box: bool, varfac_norm: str = "trial") -> tuple[SlabBlindSpotNet, WindowConfig]:
    torch.manual_seed(3)
    cfg = ModelConfig(gap=2, width=4, temporal_dilations=[1, 2], spatial_dilations=[1, 1], full_res_blocks=1,
                      window_reach=6, checkpoint=False, varfac_norm=varfac_norm)
    if box:  # pooled scales within the reach: then a box model is tiling-independent too
        cfg = ModelConfig(**{**asdict(cfg), "box_radius": 1, "hp_width": 4, "hp_temporal_dilations": [1, 1],
                             "hp_spatial_dilations": [1, 1], "hp_full_res_blocks": 1, "pool_scales": [2]})
    model = SlabBlindSpotNet(cfg).eval()
    _activate_box(model)
    wcfg = WindowConfig(out_frames=16, out_px=8, t_reach=cfg.gap + 1 + cfg.temporal_reach, s_reach=cfg.spatial_reach)
    return model, wcfg


@pytest.mark.parametrize("box,varfac_norm", [(False, "trial"), (True, "trial"), (True, "window")])
def test_tiling_does_not_change_the_output(tmp_path, box, varfac_norm, monkeypatch):
    """Small tiles (several per axis, partial at the edges) give the same
    baseline and variance as one tile covering the whole trial. With the old
    "window" variance-factor input (checkpoints up to v6) the factor is held
    at 1 here: make_inputs divides it by its median over the window, unimaged
    pixels included, which depends on the window."""
    import slap2_f0.predict as pr
    if varfac_norm == "window":
        orig = pr.make_inputs
        monkeypatch.setattr(pr, "make_inputs",
                            lambda movie, mi, vf, *a, **k: orig(movie, mi, torch.ones_like(vf), *a, **k))
    fn = tmp_path / "s" / "Path1" / "trial_000.h5"
    fn.parent.mkdir(parents=True)
    _trial(fn, 1)
    model, wcfg = _model(box, varfac_norm)
    st = load_stats([fn])[str(fn)]
    cpu = torch.device("cpu")
    with TrialFile(fn) as tr:
        mu_a, var_a = predict_trial(model, wcfg, tr, st, cpu, out_frames=16, out_px=8, batch=3)
        mu_b, var_b = predict_trial(model, wcfg, tr, st, cpu, out_frames=160, out_px=26, batch=1)
        assert mu_a.shape == (tr.n_frames, int(tr.valid.sum()))
    assert np.isfinite(mu_a).all() and np.isfinite(var_a).all()
    assert np.allclose(mu_a, mu_b, rtol=1e-4, atol=1e-4) and np.allclose(var_a, var_b, rtol=1e-4, atol=1e-4)
    assert np.std(mu_a / mu_a.mean(0)) > 1e-3  # the activated network varies in time


def test_tiles_match_training_windows(tmp_path):
    """A tile's output equals the model run on the training window
    (TrialFile.read, as WindowDataset builds it) around that tile."""
    from slap2_f0.data import _crop
    from slap2_f0.model import make_inputs

    fn = tmp_path / "s" / "Path1" / "trial_000.h5"
    fn.parent.mkdir(parents=True)
    _trial(fn, 4)
    model, wcfg = _model(box=True)
    st = load_stats([fn])[str(fn)]
    tR, sR = wcfg.t_reach, wcfg.s_reach
    with TrialFile(fn) as tr:
        mu, _ = predict_trial(model, wcfg, tr, st, torch.device("cpu"), out_frames=16, out_px=8)
        index = np.full((tr.n_rows, tr.n_cols), -1)
        rows, cols = valid_index(tr)
        index[rows, cols] = np.arange(len(rows))
        for t0, r0, c0 in [(32, 8, 8), (0, 0, 8), (144, 16, 24)]:  # inside, at the start, at the far corner
            p = tr.read(t0 - tR, 16 + 2 * tR, r0 - sR, 8 + 2 * sR, c0 - sR, 8 + 2 * sR)
            vm = np.where(np.isfinite(p.mean_image), _crop(st["var_movie"], r0 - sR, 8 + 2 * sR, c0 - sR, 8 + 2 * sR),
                          np.nan).astype(np.float32)
            s = torch.tensor([tr.scale], dtype=torch.float32)
            g = torch.tensor([st["gain"]], dtype=torch.float32)
            x = make_inputs(torch.from_numpy(p.movie)[None], torch.from_numpy(p.mean_image)[None],
                            torch.from_numpy(p.varfac)[None], s, torch.from_numpy(vm)[None], g)
            with torch.no_grad():
                ref = model(x, s, g, (slice(tR, tR + 16), slice(sR, sR + 8), slice(sR, sR + 8)))[0][0].numpy() * tr.scale
            idx = index[r0:r0 + 8, c0:c0 + 8]
            ok = idx >= 0
            t1 = min(t0 + 16, tr.n_frames)
            assert np.allclose(mu[t0:t1, idx[ok]], ref[:t1 - t0, :idx.shape[0], :idx.shape[1]][:, ok], rtol=1e-5, atol=1e-5)


def test_silo_trial_matches_window_baseline(tmp_path):
    fn = tmp_path / "s" / "Path1" / "trial_000.h5"
    fn.parent.mkdir(parents=True)
    _trial(fn, 2)
    with TrialFile(fn) as tr:
        rows, cols = valid_index(tr)
        movie, act = read_valid(tr, rows, cols)
        assert np.isnan(movie[40]).all() and np.isfinite(movie[:40]).all()
        full = tr.read(0, tr.n_frames, 0, tr.n_rows, 0, tr.n_cols)
        assert np.allclose(movie[:40], full.movie[:40][:, rows, cols]) and np.allclose(act, full.activity[:, rows, cols])
        b = silo_trial(movie, torch.device("cpu"), chunk=50)
        ref = silo_baseline(torch.from_numpy(full.movie)[None], torch.ones(1),
                            (slice(0, tr.n_frames), slice(0, tr.n_rows), slice(0, tr.n_cols)))[0].numpy()
    assert np.allclose(b, ref[:, rows, cols], equal_nan=True)


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.read(), r.headers


def test_predict_and_view(tmp_path):
    """predict writes the trial's data, baselines and SiLo; a second run adds
    a model without the movie; the viewer merges them and serves frames and
    traces that match the files."""
    for k, sess in enumerate(["sessA", "sessB"]):
        d = tmp_path / "data" / sess / "Path1"
        d.mkdir(parents=True)
        _trial(d / f"trial_00{k}.h5", 10 + k, sources=sess == "sessA")
    model, wcfg = _model(box=True)
    ck = tmp_path / "ck.pt"
    torch.save({"model": model.state_dict(), "config": asdict(model.cfg), "window": asdict(wcfg), "step": 7}, ck)
    out = tmp_path / "pred"
    main(["predict", str(tmp_path / "data"), "--ckpt", f"boxnet={ck}", "--out", str(out), "--files", "sessA",
          "--out_frames", "32"])
    fa = out / "sessA" / "Path1" / "trial_000.h5"
    assert fa.exists() and not (out / "sessB").exists()
    with h5py.File(fa, "r") as f:
        assert list(f.attrs["models"]) == ["boxnet", "silo"] and f.attrs["session"] == "sessA"
        N = f["rows"].shape[0]
        assert f["movie"].shape == f["baseline/boxnet"].shape == f["baseline/silo"].shape == (160, N)
        assert f["baseline/boxnet"].attrs["step"] == 7
        movie = f["movie"][:].astype(np.float32)
        silo = f["baseline/silo"][:].astype(np.float32)
    with TrialFile(tmp_path / "data" / "sessA" / "Path1" / "trial_000.h5") as tr:
        mu, _ = predict_trial(model, wcfg, tr, load_stats([tr.path])[str(tr.path)], torch.device("cpu"), out_frames=32)
    with h5py.File(fa, "r") as f:
        assert np.allclose(f["baseline/boxnet"][:].astype(np.float32), mu, rtol=2e-3)
    # a later run with another model, no movie: written to its own folder, merged by the viewer
    main(["predict", str(tmp_path / "data"), "--ckpt", f"other={ck}", "--out", str(tmp_path / "pred2" / "x"),
          "--files", "sessA", "--no_movie", "--no_silo"])
    server, store = make_server(tmp_path, port=0)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        meta = json.loads(_get(base + "/api/meta")[0])
        assert [t["key"] for t in meta["trials"]] == ["sessA/Path1/trial_000"]
        assert meta["trials"][0]["models"] == ["boxnet", "silo", "other"]
        key = "sessA/Path1/trial_000"
        st = json.loads(_get(f"{base}/api/static?key={key}")[0])
        assert st["layers"] == ["movie", "activity", "boxnet", "silo", "other"] and len(st["rows"]) == N
        assert st["discard"] == [40]
        body, h = _get(f"{base}/api/frame?key={key}&t=20&avg=1")
        fr = np.frombuffer(body, np.float32).reshape(5, N)
        assert json.loads(h["X-Layers"])[3] == "silo"
        assert np.allclose(fr[0], movie[20]) and np.allclose(fr[3], silo[20])
        body, _ = _get(f"{base}/api/frame?key={key}&t=40&avg=1")  # discarded: no data, baselines blanked
        assert np.isnan(np.frombuffer(body, np.float32).reshape(5, N)[[0, 2, 3, 4]]).all()
        body, _ = _get(f"{base}/api/frame?key={key}&t=10&avg=4")
        assert np.allclose(np.frombuffer(body, np.float32).reshape(5, N)[0], movie[9:13].mean(0), rtol=1e-5)
        r, c = st["rows"][100], st["cols"][100]
        body, h = _get(f"{base}/api/trace?key={key}&r={r}&c={c}&rad=0")
        tr_ = np.frombuffer(body, np.float32).reshape(5, 160)
        assert h["X-Pixels"] == "1" and np.allclose(tr_[0], movie[:, 100], equal_nan=True)
        from slap2_f0.view import Store
        st2 = Store(tmp_path, order=["other", "silo"])
        assert st2.meta[0]["models"] == ["other", "silo", "boxnet"]
        assert st2.trial(key).layers == ["movie", "activity", "other", "silo", "boxnet"]
        page, _ = _get(base + "/")
        assert b"SLAP2 F0 viewer" in page
    finally:
        server.shutdown()
        server.server_close()
