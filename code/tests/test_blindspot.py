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
