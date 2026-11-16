"""The blind spot is exact: no output depends on hidden inputs."""

import pytest
import torch

from slap2_f0.model import CausalBlock, ModelConfig, SlabBlindSpotNet, blind_window_stats, make_inputs


def _activate(model):
    """Non-zero heads and residual scales (both start at 0), so outputs use every block."""
    for head in (model.head_mu, model.head_mix, model.head_var):
        torch.nn.init.normal_(head.weight, std=0.5)
    for m in model.modules():
        if isinstance(m, CausalBlock):
            m.alpha.data.fill_(0.5)


@pytest.mark.parametrize("spatial_kernel", [3, 1])
@pytest.mark.parametrize("which", [0, 1])  # mu, var
def test_no_dependence_inside_gap(spatial_kernel, which):
    torch.manual_seed(0)
    cfg = ModelConfig(gap=3, width=8, temporal_dilations=[1, 2, 4], spatial_dilations=[1, 2, 1],
                      spatial_kernel=spatial_kernel, window_reach=10)
    model = SlabBlindSpotNet(cfg).double()
    _activate(model)
    x = torch.randn(1, cfg.n_inputs, 48, 10, 11, dtype=torch.float64)
    x[:, 2] = x[:, 2].abs() + 1  # positive mean image
    x.requires_grad_(True)
    t0, r0, c0 = 24, 5, 6
    model(x)[which][0, t0, r0, c0].backward()
    # channels 0 and 1 (movie, validity) vary in time; the whole-trial maps
    # are static per trial and broadcast over frames. The blind window
    # statistics are computed inside the model from channels 0 and 1, so this
    # covers them too.
    g = x.grad[0, :2].abs().sum(0)  # (T, H, W)
    gap = cfg.gap
    assert torch.count_nonzero(g[t0 - gap: t0 + gap + 1]) == 0  # every pixel, every hidden frame
    assert g[t0 - gap - 1].sum() > 0 and g[t0 + gap + 1].sum() > 0  # the nearest visible frames are used
    if spatial_kernel == 1:  # own trace only
        others = torch.ones_like(g, dtype=torch.bool)
        others[:, r0, c0] = False
        assert torch.count_nonzero(g[others]) == 0
    else:
        assert g[t0 - gap - 1, r0 + 1, c0].sum() > 0  # neighbours are used


def test_region_output_matches_full_output():
    """Cropping to a region before the pointwise layers gives the same values."""
    torch.manual_seed(1)
    cfg = ModelConfig(gap=2, width=6, temporal_dilations=[1, 2, 4], spatial_dilations=[1, 1, 2], full_res_blocks=1,
                      window_reach=8)
    model = SlabBlindSpotNet(cfg).double().eval()
    _activate(model)
    x = torch.randn(2, cfg.n_inputs, 30, 9, 13, dtype=torch.float64)
    x[:, 2] = x[:, 2].abs() + 1
    s = torch.tensor([2.0, 3.0], dtype=torch.float64)
    region = (slice(10, 20), slice(2, 7), slice(3, 11))
    g = torch.tensor([8.0, 11.0], dtype=torch.float64)
    mu, var = model(x, s, g)
    mu_r, var_r = model(x, s, g, region)
    assert torch.allclose(mu[(slice(None),) + region], mu_r) and torch.allclose(var[(slice(None),) + region], var_r)


def test_make_inputs_masks_missing():
    movie = torch.tensor([[[[1.0, float("nan")], [2.0, 3.0]]]]).repeat(1, 5, 1, 1)
    x = make_inputs(movie, torch.ones(1, 2, 2), torch.ones(1, 2, 2), torch.tensor([2.0]),
                    torch.full((1, 2, 2), 10.0), torch.tensor([10.0]))
    assert x.shape == (1, 5, 5, 2, 2)
    assert torch.allclose(x[0, 4], torch.zeros(5, 2, 2))  # variance = gain * mean: ratio 1
    assert x[0, 0, 0, 0, 1] == 0 and x[0, 1, 0, 0, 1] == 0
    assert x[0, 0, 0, 1, 1] == 1.5 and x[0, 1, 0, 1, 1] == 1


def test_untrained_network_predicts_mean_image():
    torch.manual_seed(2)
    cfg = ModelConfig(gap=2, width=4, temporal_dilations=[1, 2], spatial_dilations=[1, 1], window_reach=6)
    model = SlabBlindSpotNet(cfg).double()
    x = torch.randn(1, cfg.n_inputs, 20, 6, 7, dtype=torch.float64)
    x[:, 2] = x[:, 2].abs() + 1
    mu, _ = model(x)
    assert torch.allclose(mu.double(), x[:, 2], atol=1e-5)


def test_outputs_are_bounded():
    """Saturated heads move the baseline by at most the correction cap and mix at most fully toward the window."""
    torch.manual_seed(4)
    cfg = ModelConfig(gap=2, width=4, temporal_dilations=[1, 2], spatial_dilations=[1, 1], window_reach=6)
    model = SlabBlindSpotNet(cfg).double()
    x = torch.randn(1, cfg.n_inputs, 20, 6, 7, dtype=torch.float64)
    x[:, 0] = x[:, 0].abs()
    x[:, 1] = 1.0
    x[:, 2] = x[:, 2].abs() + 1
    for head in (model.head_mu, model.head_mix, model.head_var):
        head.bias.data.fill_(100.0)
    mu, var, aux = model(x, aux=True)
    assert aux["mix"].max() <= 1.0
    lo, hi = model.cfg.max_log_correction, model.cfg.max_log_var
    assert torch.all(mu <= aux["base"] * torch.exp(torch.tensor(lo)) * (1 + 1e-6))
    assert torch.all(var <= (10.0 * mu + model.log_floor.exp()) * torch.exp(torch.tensor(hi)) * (1 + 1e-6))


def test_blind_window_stats_match_brute_force():
    torch.manual_seed(3)
    m = torch.rand(2, 30, 3, 4, dtype=torch.float64)
    v = (torch.rand_like(m) > 0.2).double()
    gap, reach = 2, 5
    mean, var, n = blind_window_stats(m * v, v, gap, reach)
    for t in range(30):
        dt = (torch.arange(30) - t).abs()
        sel = (dt > gap) & (dt <= gap + reach)
        ww = v[:, sel]
        mm = (m[:, sel] * ww).sum(1) / ww.sum(1)
        vv = (m[:, sel] ** 2 * ww).sum(1) / ww.sum(1) - mm ** 2
        assert torch.allclose(n[:, t], ww.sum(1).float())
        assert torch.allclose(mean[:, t].double(), mm, atol=1e-5) and torch.allclose(var[:, t].double(), vv, atol=1e-5)


def _activate_box(model):
    _activate(model)
    for m in model.modules():
        if type(m).__name__ == "HalfPlaneBlock":
            m.alpha.data.fill_(0.5)


@pytest.mark.parametrize("which", [0, 1])  # mu, var
@pytest.mark.parametrize("radius", [2, 4])
def test_box_blind_spot(which, radius):
    """Box model: nothing inside |dt| <= gap, |dr|, |dc| <= radius reaches the
    output; the nearest pixels outside the box are used in the output frame."""
    torch.manual_seed(0)
    cfg = ModelConfig(gap=3, width=6, temporal_dilations=[1, 2, 4], spatial_dilations=[1, 2, 1], window_reach=10,
                      full_res_blocks=1, box_radius=radius, hp_width=6, hp_temporal_dilations=[1, 2, 1],
                      hp_spatial_dilations=[1, 1, 2], hp_full_res_blocks=1, pool_scales=[3, 0])
    model = SlabBlindSpotNet(cfg).double()
    _activate_box(model)
    x = torch.randn(1, cfg.n_inputs, 40, 23, 25, dtype=torch.float64)
    x[:, 1] = (torch.rand(1, 40, 23, 25) > 0.1).double()
    x[:, 0] = x[:, 0] * x[:, 1]
    x[:, 2] = x[:, 2].abs() + 1
    x.requires_grad_(True)
    t0, r0, c0 = 20, 11, 12
    model(x)[which][0, t0, r0, c0].backward()
    g = x.grad[0, :2].abs().sum(0)  # (T, H, W)
    gap, rho = cfg.gap, radius
    assert torch.count_nonzero(g[t0 - gap: t0 + gap + 1, r0 - rho: r0 + rho + 1, c0 - rho: c0 + rho + 1]) == 0
    for dr, dc in ((-rho - 1, 0), (rho + 1, 0), (0, -rho - 1), (0, rho + 1), (rho + 1, rho + 1)):
        assert g[t0, r0 + dr, c0 + dc] > 0, (dr, dc)  # the same frame, just outside the box
    assert g[t0 - gap - 1, r0, c0] > 0 and g[t0 + gap + 1, r0, c0] > 0  # the pixel's own trace beyond the gap


def test_box_blind_spot_every_position():
    """The box holds at every output position of a small window, edges included
    (each output checked through the gradient of a random projection would mix
    positions, so check a grid of single outputs)."""
    torch.manual_seed(5)
    cfg = ModelConfig(gap=2, width=4, temporal_dilations=[1, 2], spatial_dilations=[1, 1], window_reach=5,
                      full_res_blocks=1, box_radius=2, hp_width=4, hp_temporal_dilations=[1, 2],
                      hp_spatial_dilations=[1, 2], hp_full_res_blocks=1, pool_scales=[2, 0])
    model = SlabBlindSpotNet(cfg).double()
    _activate_box(model)
    x = torch.randn(1, cfg.n_inputs, 16, 11, 13, dtype=torch.float64)
    x[:, 1] = 1.0
    x[:, 2] = x[:, 2].abs() + 1
    for r0 in range(11):
        for c0 in (0, 1, 6, 11, 12):
            xx = x.clone().requires_grad_(True)
            model(xx)[0][0, 8, r0, c0].backward()
            g = xx.grad[0, :2].abs().sum(0)
            box = g[6:11, max(0, r0 - 2): r0 + 3, max(0, c0 - 2): c0 + 3]
            assert torch.count_nonzero(box) == 0, (r0, c0)


def test_box_region_output_matches_full_output():
    torch.manual_seed(6)
    cfg = ModelConfig(gap=2, width=6, temporal_dilations=[1, 2, 4], spatial_dilations=[1, 1, 2], full_res_blocks=1,
                      window_reach=8, box_radius=2, hp_width=4, hp_temporal_dilations=[1, 2, 2],
                      hp_spatial_dilations=[1, 1, 2], hp_full_res_blocks=1, pool_scales=[3, 0])
    model = SlabBlindSpotNet(cfg).double().eval()
    _activate_box(model)
    x = torch.randn(2, cfg.n_inputs, 30, 14, 14, dtype=torch.float64)
    x[:, 2] = x[:, 2].abs() + 1
    s = torch.tensor([2.0, 3.0], dtype=torch.float64)
    g = torch.tensor([8.0, 11.0], dtype=torch.float64)
    region = (slice(10, 20), slice(3, 11), slice(3, 11))
    mu, var = model(x, s, g)
    mu_r, var_r = model(x, s, g, region)
    assert torch.allclose(mu[(slice(None),) + region], mu_r) and torch.allclose(var[(slice(None),) + region], var_r)
    # untrained (zero heads): still the mean image
    model2 = SlabBlindSpotNet(cfg).double()
    assert torch.allclose(model2(x)[0].double(), x[:, 2], atol=1e-5)


def test_pooled_features_match_brute_force():
    from slap2_f0.model import pooled_features

    torch.manual_seed(7)
    x = torch.rand(1, 5, 12, 9, 10, dtype=torch.float64)
    x[:, 1] = (torch.rand(1, 12, 9, 10) > 0.2).double()
    x[:, 0] = x[:, 0] * x[:, 1]
    x[:, 2] = x[:, 2] + 0.5
    pf = pooled_features(x, 4, 9, 1, [2])
    m, v, mi = x[0, 0], x[0, 1], x[0, 2]
    for t in range(4, 9):
        for r, c in ((0, 0), (4, 5), (8, 9)):
            sq = torch.zeros(9, 10, dtype=torch.bool)
            sq[max(0, r - 2): r + 3, max(0, c - 2): c + 3] = True
            sq[max(0, r - 1): r + 2, max(0, c - 1): c + 2] = False
            ref1 = (v[t] * (m[t] - mi[t]))[sq].sum() / (v[t] * mi[t])[sq].sum()
            ts = slice(max(0, t - 3), t + 5)
            ref8 = (v[ts] * (m[ts] - mi[ts]))[:, sq].sum() / (v[ts] * mi[ts])[:, sq].sum()
            assert torch.allclose(pf[0, 0, t - 4, r, c].double(), ref1.clamp(-1, 1), atol=1e-5)
            assert torch.allclose(pf[0, 1, t - 4, r, c].double(), ref8.clamp(-1, 1), atol=1e-5)


def test_box_model_warm_starts_from_slab():
    """A slab model's weights in a box model: the same outputs at the start."""
    from slap2_f0.train import load_compatible

    torch.manual_seed(9)
    kw = dict(gap=2, width=6, temporal_dilations=[1, 2], spatial_dilations=[1, 1], window_reach=6, full_res_blocks=1)
    slab = SlabBlindSpotNet(ModelConfig(**kw)).double()
    _activate(slab)
    box = SlabBlindSpotNet(ModelConfig(**kw, box_radius=2, hp_width=4, hp_temporal_dilations=[1, 2],
                                       hp_spatial_dilations=[1, 1], hp_full_res_blocks=1, pool_scales=[3, 0])).double()
    msg = load_compatible(box, slab.state_dict())
    assert "fuse.0.weight" in msg
    _activate_box(box)  # half-plane branches active: their outputs still enter with zero weights
    # _activate_box re-drew the heads: give them the slab's again
    box.head_mu.load_state_dict(slab.head_mu.state_dict())
    box.head_mix.load_state_dict(slab.head_mix.state_dict())
    box.head_var.load_state_dict(slab.head_var.state_dict())
    x = torch.randn(1, 5, 20, 12, 12, dtype=torch.float64)
    x[:, 2] = x[:, 2].abs() + 1
    a, b = slab(x), box(x)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])
