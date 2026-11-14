"""The blind spot is exact: no output depends on hidden inputs."""

import pytest
import torch

from slap2_f0.model import ModelConfig, SlabBlindSpotNet, make_inputs


@pytest.mark.parametrize("spatial_kernel", [3, 1])
@pytest.mark.parametrize("which", [0, 1])  # mu, var
def test_no_dependence_inside_gap(spatial_kernel, which):
    torch.manual_seed(0)
    cfg = ModelConfig(gap=3, width=8, temporal_dilations=[1, 2, 4], spatial_dilations=[1, 2, 1],
                      spatial_kernel=spatial_kernel)
    model = SlabBlindSpotNet(cfg).double()
    for head in (model.head_mu, model.head_var):  # non-zero heads, so outputs use the features
        torch.nn.init.normal_(head.weight, std=0.5)
    x = torch.randn(1, cfg.n_inputs, 48, 10, 11, dtype=torch.float64)
    x[:, 2] = x[:, 2].abs() + 1  # positive mean image
    x.requires_grad_(True)
    t0, r0, c0 = 24, 5, 6
    model(x)[which][0, t0, r0, c0].backward()
    # channels 0 and 1 (movie, validity) vary in time; the mean image and
    # variance factor are static per trial and broadcast over frames
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


def test_make_inputs_masks_missing():
    movie = torch.tensor([[[[1.0, float("nan")], [2.0, 3.0]]]]).repeat(1, 5, 1, 1)
    x = make_inputs(movie, torch.ones(1, 2, 2), torch.ones(1, 2, 2), torch.tensor([2.0]))
    assert x.shape == (1, 4, 5, 2, 2)
    assert x[0, 0, 0, 0, 1] == 0 and x[0, 1, 0, 0, 1] == 0
    assert x[0, 0, 0, 1, 1] == 1.5 and x[0, 1, 0, 1, 1] == 1
