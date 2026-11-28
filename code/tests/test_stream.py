"""Streaming inference (stream.StreamPredictor) gives predict_trial's baselines."""

from dataclasses import asdict

import h5py
import numpy as np
import pytest
import torch

from slap2_f0.data import WindowConfig
from slap2_f0.io import TrialFile
from slap2_f0.model import ModelConfig, SlabBlindSpotNet
from slap2_f0.predict import predict_trial, tile_origins, valid_index
from slap2_f0.stats import load_stats
from slap2_f0.stream import StreamPredictor, plan_rects
from test_blindspot import _activate_box


def _trial(fn, seed, T=150, R=40, C=58):
    """A trial with two imaged regions apart (two rectangles), an unimaged
    corner, discarded frames and one event."""
    rng = np.random.default_rng(seed)
    base = 5.0 + rng.random((R, C)).astype(np.float32) * 5
    drift = 1 + 0.1 * np.sin(np.arange(T) / 9.0)[:, None, None]
    movie = rng.poisson(base * drift).astype(np.float32)[None]
    act = np.zeros_like(movie)
    act[0, 30:36, 8:12, 8:12] = 6.0
    movie += act
    valid = np.zeros((R, C), np.uint8)
    valid[2:20, 1:15] = 1
    valid[5:9, 3:6] = 0
    valid[22:39, 40:57] = 1
    discard = np.zeros(T, np.uint8)
    discard[[40, 41, 99]] = 1
    sup = np.zeros((R, C), np.uint8)
    sup[7:13, 7:13] = 1
    with h5py.File(fn, "w") as f:
        f["movie"] = movie
        f["activity"] = act
        f["mean_image"] = movie.mean(1)
        vf = (0.2 + 0.4 * rng.random((R, C))).astype(np.float32)
        f["varfac"] = vf
        f["mask/valid"] = valid
        f["mask/labeled"] = sup
        f["mask/footprint_support"] = sup
        f["frame/discard"] = discard
        f.attrs["nSources"] = 1


def _model(kind: str) -> tuple[SlabBlindSpotNet, WindowConfig]:
    """Small models of each kind; pooled scales beyond the reach and the whole
    window (0), so the pooled features depend on the tile's window as in the
    released checkpoints."""
    torch.manual_seed(5)
    cfg = ModelConfig(gap=2, width=4, temporal_dilations=[1, 2, 4], spatial_dilations=[1, 1, 2], full_res_blocks=1,
                      window_reach=6, checkpoint=False, branch_mlp=1, fuse_blocks=2, pointwise_init="zero_out")
    if kind != "slab":
        cfg = ModelConfig(**{**asdict(cfg), "box_radius": 1, "hp_width": 4, "hp_temporal_dilations": [1, 2],
                             "hp_spatial_dilations": [1, 1], "hp_full_res_blocks": 1, "pool_scales": [2, 9, 0]})
    if kind == "shared":
        cfg = ModelConfig(**{**asdict(cfg), "hp_shared": True})
    if kind == "edge_motion":
        cfg = ModelConfig(**{**asdict(cfg), "edge_features": True, "motion_features": True})
    model = SlabBlindSpotNet(cfg).eval()
    _activate_box(model)
    for m in model.modules():  # zero_out starts the pointwise blocks at the identity: make them act
        if type(m).__name__ == "PointwiseBlock":
            torch.nn.init.normal_(m.w2.weight, std=0.3)
    for head in [model.head_mu, model.head_var, model.head_mix] + ([model.head_edge] if model.head_edge else []):
        torch.nn.init.normal_(head.weight, std=0.3)
    wcfg = WindowConfig(out_frames=16, out_px=8, t_reach=cfg.gap + 1 + cfg.temporal_reach, s_reach=cfg.spatial_reach)
    return model, wcfg


def test_rectangles_hold_every_tile_window():
    """Each tile is in one rectangle that contains its whole window, corners
    at the windows' parity and sizes even."""
    want = np.zeros((60, 90), bool)
    want[3:20, 2:30] = True
    want[40:58, 70:88] = True
    want[25:30, 5:8] = True
    op, sR = 8, 5
    tiles = {(r0, c0) for r0, cs in tile_origins(want, op) for c0 in cs}
    for max_px in (0, 900):
        rects = plan_rects(want, op, sR, max_px)
        seen = [t for r in rects for t in r.tiles]
        assert sorted(seen) == sorted(tiles)
        for r in rects:
            assert (r.r1 - r.r0) % 2 == 0 and (r.c1 - r.c0) % 2 == 0
            for r0, c0 in r.tiles:
                assert r.r0 <= r0 - sR and r0 + op + sR <= r.r1 and r.c0 <= c0 - sR and c0 + op + sR <= r.c1
                assert (r.r0 - (r0 - sR)) % 2 == 0 and (r.c0 - (c0 - sR)) % 2 == 0
    assert len(plan_rects(want, op, sR, 900)) > len(plan_rects(want, op, sR, 0))


@pytest.mark.parametrize("kind", ["slab", "box", "shared", "edge_motion"])
@pytest.mark.parametrize("chunk,max_px,layout", [(7, 0, "ncdhw"), (40, 600, "cl"), (400, 0, "cl")])
def test_stream_matches_predict_trial(tmp_path, kind, chunk, max_px, layout):
    """Same baselines and variances as the reference tiling (float32, CPU), for
    chunks shorter than the dilations, rectangles split into bands, one chunk
    for the whole trial, and both layouts (the reference's and channels-last)."""
    fn = tmp_path / "s" / "Path1" / "trial_000.h5"
    fn.parent.mkdir(parents=True)
    _trial(fn, 2)
    model, wcfg = _model(kind)
    st = load_stats([fn])[str(fn)]
    cpu = torch.device("cpu")
    with TrialFile(fn) as tr:
        rows, cols = valid_index(tr)
        mu_a, var_a = predict_trial(model, wcfg, tr, st, cpu, out_frames=40, rows=rows, cols=cols)
        sp = StreamPredictor(model, wcfg, cpu, None, chunk=chunk, max_px=max_px, pixel_block=50, layout=layout)
        mu_b, var_b = sp.predict(tr, st, rows, cols)
    assert np.isfinite(mu_b).all() and np.isfinite(var_b).all()
    assert np.std(mu_a / mu_a.mean(0)) > 1e-3  # the activated network varies in time
    np.testing.assert_allclose(mu_b, mu_a, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(var_b, var_a, rtol=2e-5, atol=2e-5)
