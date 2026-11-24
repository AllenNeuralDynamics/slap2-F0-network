"""The systematic-error term (train.bias_loss, --bias_weight)."""

import types

import torch

from slap2_f0.train import _losses, bias_loss


def _batch(B=2, T=8, H=10, W=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    scale = torch.tensor([20.0, 35.0])[:B]
    mi = (torch.rand(B, H, W, generator=g) * 3 + 0.5) * scale.view(-1, 1, 1)  # counts
    sup = torch.zeros(B, H, W, dtype=torch.uint8)
    sup[:, 2:5, 3:7] = 1
    lab = sup.clone()
    lab[:, 1:7, 2:9] = 1
    target = (mi / scale.view(-1, 1, 1))[:, None].expand(B, T, H, W).clone()
    return {"scale": scale, "gain": torch.tensor([3.0, 8.0])[:B], "mean_image": mi, "support": sup, "labeled": lab,
            "target": target, "noisy": target.clone(), "weight": torch.ones(B, T, H, W), "sim": torch.ones(B)}


def _region(b):
    _, T, H, W = b["target"].shape
    return slice(0, T), slice(0, H), slice(0, W)


def _mu_loss_part(b, mu, region, mask):
    from slap2_f0.train import _ref_var
    w = b["weight"] * mask
    return (w * (b["target"] - mu) ** 2 / _ref_var(b, region)).sum() / b["weight"].sum()


def test_constant_offset_equals_its_share_of_the_mean_loss():
    b = _batch()
    region = _region(b)
    bg = (b["labeled"] == 0)[:, None].float().expand_as(b["target"]).clone()
    bg[1] = 0  # only window 0's background is off
    mu = b["target"] - 0.07 * bg
    assert torch.isclose(bias_loss(b, mu, region), _mu_loss_part(b, mu, region, bg), rtol=1e-5)


def test_zero_without_error_and_for_real_windows():
    b = _batch()
    region = _region(b)
    assert bias_loss(b, b["target"].clone(), region) == 0
    mu = b["target"] * 0.9  # every class biased
    assert bias_loss(b, mu, region) > 0
    b["sim"] = torch.zeros(2)  # real windows: left out
    assert bias_loss(b, mu, region) == 0
    del b["sim"]  # batches without the flag (older datasets): no term
    assert bias_loss(b, mu, region) == 0


def test_random_errors_cost_almost_nothing():
    b = _batch(T=32, H=16, W=16)
    region = _region(b)
    g = torch.Generator().manual_seed(1)
    mu = b["target"] + 0.05 * torch.sign(torch.randn(b["target"].shape, generator=g))
    from slap2_f0.train import _ref_var
    full = (b["weight"] * (b["target"] - mu) ** 2 / _ref_var(b, region)).sum() / b["weight"].sum()
    assert bias_loss(b, mu, region) < 0.02 * full


def test_losses_add_the_weighted_term_on_simulated_windows_only():
    b = _batch()
    region = _region(b)
    mu = (b["target"] * 0.95).requires_grad_(True)
    var = torch.ones_like(mu)
    args0 = types.SimpleNamespace(w_mu=1.0, w_var=0.1, bias_weight=0.0)
    args3 = types.SimpleNamespace(w_mu=1.0, w_var=0.1, bias_weight=3.0)
    l0, m0, v0 = _losses(args0, b, mu, var, region)
    l3, m3, v3 = _losses(args3, b, mu, var, region)
    assert torch.equal(m0, m3) and torch.equal(v0, v3)
    assert torch.isclose(l3 - l0, 3.0 * bias_loss(b, mu, region))
    b["sim"] = torch.tensor([1.0, 0.0])
    (g,) = torch.autograd.grad(bias_loss(b, mu, region), mu)
    assert g[0].abs().sum() > 0 and g[1].abs().sum() == 0
