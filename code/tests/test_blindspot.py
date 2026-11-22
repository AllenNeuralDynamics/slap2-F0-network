"""The blind spot is exact: no output depends on hidden inputs."""

import math

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
@pytest.mark.parametrize("shared", [False, True])  # one half-plane trunk per direction, or one for all
def test_box_blind_spot(which, radius, shared):
    """Box model: nothing inside |dt| <= gap, |dr|, |dc| <= radius reaches the
    output; the nearest pixels outside the box are used in the output frame."""
    torch.manual_seed(0)
    cfg = ModelConfig(gap=3, width=6, temporal_dilations=[1, 2, 4], spatial_dilations=[1, 2, 1], window_reach=10,
                      full_res_blocks=1, box_radius=radius, hp_width=6, hp_temporal_dilations=[1, 2, 1],
                      hp_spatial_dilations=[1, 1, 2], hp_full_res_blocks=1, pool_scales=[3, 0], hp_shared=shared)
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
    msg = load_compatible(box, slab.state_dict(), slab.cfg)
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


def test_varfac_input_does_not_depend_on_the_window():
    """The "trial" variance-factor input is log(factor) at each pixel, whatever
    else is in the window; the old "window" input divided by the window's
    median, unimaged pixels (factor 0) included, and saturated at +3 once more
    than half of the window was unimaged."""
    from slap2_f0.model import config_from_dict
    from dataclasses import asdict
    movie = torch.ones(1, 3, 4, 4)
    vf = torch.full((1, 4, 4), 1.5)
    vf_sparse = vf.clone()
    vf_sparse[0, :3] = 0  # 12 of 16 pixels unimaged
    one, s = torch.ones(1, 4, 4), torch.tensor([1.0])
    a = make_inputs(movie, one, vf, s)[0, 3, 0]
    b = make_inputs(movie, one, vf_sparse, s)[0, 3, 0]
    assert torch.allclose(a[3], b[3]) and torch.allclose(a[3], torch.log(torch.tensor(1.5)))
    assert (b[:3] == 0).all()
    w = make_inputs(movie, one, vf_sparse, s, varfac_norm="window")[0, 3, 0]
    assert torch.allclose(w[3], torch.tensor(3.0))
    old = asdict(ModelConfig())
    del old["varfac_norm"]
    assert config_from_dict(old).varfac_norm == "window" and ModelConfig().varfac_norm == "trial"


def test_adapted_varfac_weights_keep_the_output():
    """A "window" checkpoint folded into a "trial" model (train --init) gives
    the same output on a window where the old input was saturated (more than
    half unimaged, no discarded frames), box and slab alike."""
    from slap2_f0.train import adapt_varfac_input, load_compatible
    torch.manual_seed(0)
    kw = dict(gap=2, width=4, temporal_dilations=[1, 2], spatial_dilations=[1, 1], full_res_blocks=1,
              window_reach=6, checkpoint=False, box_radius=1, hp_width=4, hp_temporal_dilations=[1, 1],
              hp_spatial_dilations=[1, 1], hp_full_res_blocks=1, pool_scales=[2])
    old = SlabBlindSpotNet(ModelConfig(**kw, varfac_norm="window")).eval()
    _activate_box(old)
    new = SlabBlindSpotNet(ModelConfig(**kw, varfac_norm="trial")).eval()
    load_compatible(new, adapt_varfac_input(old.state_dict(), 4), old.cfg)
    T, H, W = 24, 12, 12
    movie = torch.full((1, T, H, W), float("nan"))
    movie[:, :, 3:9, 4:10] = 5 + torch.rand(1, T, 6, 6)  # 36 of 144 pixels imaged
    mean = torch.where(torch.isfinite(movie[0, 0]), movie.nanmean(1)[0], torch.tensor(float("nan")))[None]
    vf = torch.where(torch.isfinite(mean), 0.8 + 0.4 * torch.rand(1, H, W), torch.zeros(1, H, W))
    s = torch.tensor([5.0])
    region = (slice(8, 16), slice(3, 9), slice(4, 10))
    with torch.no_grad():
        a = old(make_inputs(movie, mean, vf, s, varfac_norm="window"), s, region=region)[0]
        b = new(make_inputs(movie, mean, vf, s, varfac_norm="trial"), s, region=region)[0]
        c = old(make_inputs(movie, mean, vf, s, varfac_norm="trial"), s, region=region)[0]
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)
    assert not torch.allclose(a, c, rtol=1e-3, atol=1e-4)  # unfolded, the new input changes the output


def test_shared_halfplane_trunk_starts_per_direction_trunks():
    """A v6 checkpoint (one half-plane trunk for all directions) in a model with
    one trunk per direction: the same outputs at the start; then the
    directions can differ (they no longer share weights)."""
    from slap2_f0.model import config_from_dict
    from slap2_f0.train import load_compatible

    torch.manual_seed(3)
    kw = dict(gap=2, width=4, temporal_dilations=[1, 2], spatial_dilations=[1, 1], full_res_blocks=1,
              window_reach=6, checkpoint=False, box_radius=1, hp_width=4, hp_temporal_dilations=[1, 2],
              hp_spatial_dilations=[1, 1], hp_full_res_blocks=1, pool_scales=[2])
    old = SlabBlindSpotNet(ModelConfig(**kw, hp_shared=True)).double().eval()
    _activate_box(old)
    new = SlabBlindSpotNet(ModelConfig(**kw)).double().eval()
    assert new.hp is None and len(new.hp_views) == 4
    load_compatible(new, old.state_dict(), old.cfg)
    x = torch.randn(1, 5, 20, 12, 14, dtype=torch.float64)
    x[:, 2] = x[:, 2].abs() + 1
    with torch.no_grad():
        a, b = old(x), new(x)
        assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])
        new.hp_views[1].inp.weight.add_(0.1)  # one direction changes on its own
        assert not torch.allclose(new(x)[0], a[0])
    # checkpoints saved before the option load as shared
    d = {k: v for k, v in vars(ModelConfig(**kw)).items() if k != "hp_shared"}
    assert config_from_dict(d).hp_shared


@pytest.mark.parametrize("which", [0, 1])  # mu, var
def test_box_blind_spot_with_motion_features(which):
    """The surround's displacement and blur features keep the box blind spot."""
    torch.manual_seed(1)
    cfg = ModelConfig(gap=3, width=6, temporal_dilations=[1, 2, 4], spatial_dilations=[1, 2, 1], window_reach=10,
                      full_res_blocks=1, box_radius=2, hp_width=6, hp_temporal_dilations=[1, 2, 1],
                      hp_spatial_dilations=[1, 1, 2], hp_full_res_blocks=1, pool_scales=[3, 0], motion_features=True)
    model = SlabBlindSpotNet(cfg).double()
    _activate_box(model)
    torch.nn.init.normal_(model.fuse[0].weight, std=0.5)  # every fusion input counts, the new features included
    x = torch.randn(1, cfg.n_inputs, 40, 23, 25, dtype=torch.float64)
    x[:, 1] = (torch.rand(1, 40, 23, 25) > 0.1).double()
    x[:, 0] = x[:, 0] * x[:, 1]
    x[:, 2] = x[:, 2].abs() + 1
    x.requires_grad_(True)
    t0, r0, c0 = 20, 11, 12
    model(x)[which][0, t0, r0, c0].backward()
    g = x.grad[0, :2].abs().sum(0)
    assert torch.count_nonzero(g[t0 - 3: t0 + 4, r0 - 2: r0 + 3, c0 - 2: c0 + 3]) == 0
    assert g[t0, r0 + 3, c0] > 0 and g[t0, r0, c0 - 3] > 0


def test_motion_features_warm_start_and_read_a_shift():
    """A box model without motion features warm-starts one with them exactly;
    the feature reads a small shift of the scene as the change it makes at the pixel."""
    from slap2_f0.model import motion_features
    from slap2_f0.train import load_compatible

    torch.manual_seed(2)
    kw = dict(gap=2, width=6, temporal_dilations=[1, 2], spatial_dilations=[1, 1], window_reach=6, full_res_blocks=1,
              box_radius=2, hp_width=4, hp_temporal_dilations=[1, 2], hp_spatial_dilations=[1, 1], hp_full_res_blocks=1,
              pool_scales=[3, 0])
    old = SlabBlindSpotNet(ModelConfig(**kw)).double()
    _activate_box(old)
    new = SlabBlindSpotNet(ModelConfig(**kw, motion_features=True)).double()
    load_compatible(new, old.state_dict(), old.cfg)
    x = torch.randn(1, 5, 20, 12, 12, dtype=torch.float64)
    x[:, 2] = x[:, 2].abs() + 1
    a, b = old(x), new(x)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])
    # a smooth scene of Gaussian blobs, shifted by d in frame 1 (exact), not in frame 0
    H = W = 40
    rr, cc = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    blobs = [(8, 9, 2.0, 6), (20, 25, 2.5, 4), (30, 12, 1.8, 5), (14, 32, 2.2, 7), (31, 31, 3.0, 3)]

    def scene(dr, dc):
        s = torch.full((H, W), 1.0, dtype=torch.float64)
        for r0, c0, sg, amp in blobs:
            s = s + amp * torch.exp(-((rr - r0 - dr) ** 2 + (cc - c0 - dc) ** 2) / (2 * sg ** 2))
        return s

    d = (0.25, -0.15)
    mi, moved = scene(0, 0), scene(*d)
    x = torch.zeros(1, 5, 9, H, W, dtype=torch.float64)
    x[:, 0] = mi
    x[:, 0, 4] = moved
    x[:, 1] = 1.0
    x[:, 2] = mi
    f = motion_features(x, 4, 5, 2, [0])  # (1, 2 + 4, 1, H, W): whole-window scale, frame and 8-frame mean
    true = ((moved - mi) / mi)[4:-4, 4:-4]
    est = f[0, 0, 0, 4:-4, 4:-4].double()
    slope = (est * true).sum() / (true * true).sum()
    assert abs(slope - 1) < 0.15 and torch.corrcoef(torch.stack([est.flatten(), true.flatten()]))[0, 1] > 0.95


def test_edge_features_match_brute_force():
    """Per-side counts and means over the blind window, and the gates."""
    from slap2_f0.model import edge_features

    torch.manual_seed(5)
    T, gap, reach, near = 40, 2, 7, 3
    m = torch.rand(2, T, 3, 4, dtype=torch.float64) + 0.5
    v = (torch.rand_like(m) > 0.2).double()
    v[:, 30:] = 0  # the trial ends at frame 30
    mi = torch.rand(2, 3, 4, dtype=torch.float64) + 0.5
    f, g = edge_features(m * v, v, mi, gap, reach, near)
    f5, g5 = edge_features(m * v, v, mi, gap, reach, near, (5, 25))
    assert torch.allclose(f5, f[:, :, 5:25]) and torch.allclose(g5, g[:, :, 5:25])
    t_ = torch.arange(T)
    for t in range(T):
        for side, k in ((t_ < t - gap) & (t_ >= t - gap - reach), 0), ((t_ > t + gap) & (t_ <= t + gap + reach), 1):
            n = v[:, side].sum(1)
            kn = v[:, side & ((t_ - t).abs() <= gap + near)].sum(1)
            mean = (m[:, side] * v[:, side]).sum(1) / n.clamp_min(1)
            rel = torch.where(n > 0, torch.log(mean.clamp_min(1e-3) / mi).clamp(-3, 3), torch.zeros_like(n))
            assert torch.allclose(f[:, k, t].double(), torch.log1p(n) / math.log1p(reach), atol=1e-6)
            assert torch.allclose(f[:, 2 + k, t].double(), kn / near, atol=1e-6)
            assert torch.allclose(f[:, 4 + k, t].double(), rel, atol=1e-5)
            assert torch.allclose(g[:, k, t].double(), 1 - n / reach, atol=1e-6)
    last = slice(30 - gap - 1, 30)  # frames whose future window lies past the trial end
    assert torch.all(g[:, 1, last] == 1) and torch.all(g[:, 0, last] < 1)


def test_edge_near_capped_at_reach():
    """With edge_near > window_reach the near counts span reach frames, so a
    frame with complete window statistics gets the same features in a tile cut
    to its halo (gap + reach frames each side) as in the whole movie."""
    from slap2_f0.model import edge_features

    torch.manual_seed(8)
    T, gap, reach, near = 40, 2, 5, 16
    m = torch.rand(1, T, 3, 4, dtype=torch.float64) + 0.5
    v = torch.ones_like(m)
    mi = torch.rand(1, 3, 4, dtype=torch.float64) + 0.5
    f, g = edge_features(m, v, mi, gap, reach, near)
    t, h = 20, gap + reach
    fc, gc = edge_features(m[:, t - h: t + h + 1], v[:, t - h: t + h + 1], mi, gap, reach, near)
    # counts and gates exact; the side means differ by float32 rounding of the cumulative sums (~4e-7)
    assert torch.equal(fc[:, :4, h], f[:, :4, t]) and torch.equal(gc[:, :, h], g[:, :, t])
    assert torch.allclose(fc[:, 4:, h], f[:, 4:, t], atol=1e-5)
    assert torch.all(f[:, 2:4, t] == 1)
    assert torch.allclose(f, edge_features(m, v, mi, gap, reach, reach)[0])


@pytest.mark.parametrize("box", [False, True])
@pytest.mark.parametrize("which", [0, 1])  # mu, var
def test_blind_spot_with_edge_features(box, which):
    """The per-side statistics and the edge-gated heads keep the blind spot,
    at a trial edge too."""
    torch.manual_seed(6)
    kw = dict(gap=3, width=6, temporal_dilations=[1, 2, 4], spatial_dilations=[1, 2, 1], window_reach=10,
              full_res_blocks=1, edge_features=True, edge_near=4)
    if box:
        kw.update(box_radius=2, hp_width=6, hp_temporal_dilations=[1, 2, 1], hp_spatial_dilations=[1, 1, 2],
                  hp_full_res_blocks=1, pool_scales=[3, 0])
    model = SlabBlindSpotNet(ModelConfig(**kw)).double()
    _activate_box(model) if box else _activate(model)
    torch.nn.init.normal_(model.fuse[0].weight, std=0.5)  # every fusion input counts, the new ones included
    torch.nn.init.normal_(model.head_edge.weight, std=0.5)
    for t0 in (20, 33):  # mid-window, and 3 frames before the trial's end (frame 36)
        x = torch.randn(1, 5, 40, 13, 14, dtype=torch.float64)
        x[:, 1] = (torch.rand(1, 40, 13, 14) > 0.1).double()
        x[:, 1, 36:] = 0
        x[:, 0] = x[:, 0] * x[:, 1]
        x[:, 2] = x[:, 2].abs() + 1
        x.requires_grad_(True)
        r0, c0 = 6, 7
        model(x)[which][0, t0, r0, c0].backward()
        g = x.grad[0, :2].abs().sum(0)
        if box:
            assert torch.count_nonzero(g[t0 - 3: t0 + 4, r0 - 2: r0 + 3, c0 - 2: c0 + 3]) == 0
        else:
            assert torch.count_nonzero(g[t0 - 3: t0 + 4]) == 0
        assert g[t0 - 4, r0, c0] > 0


def test_edge_features_warm_start_and_region():
    """A box model without edge features warm-starts one with them exactly
    (the new fusion columns and the gated head start at 0); the region
    output matches the full output."""
    from slap2_f0.train import load_compatible

    torch.manual_seed(7)
    kw = dict(gap=2, width=6, temporal_dilations=[1, 2], spatial_dilations=[1, 1], window_reach=6, full_res_blocks=1,
              box_radius=2, hp_width=4, hp_temporal_dilations=[1, 2], hp_spatial_dilations=[1, 1], hp_full_res_blocks=1,
              pool_scales=[3, 0])
    old = SlabBlindSpotNet(ModelConfig(**kw)).double()
    _activate_box(old)
    new = SlabBlindSpotNet(ModelConfig(**kw, edge_features=True)).double().eval()
    msg = load_compatible(new, old.state_dict(), old.cfg)
    assert "fuse.0.weight" in msg and "new" in msg
    x = torch.randn(1, 5, 24, 12, 12, dtype=torch.float64)
    x[:, 1] = 1.0
    x[:, 1, 18:] = 0  # a trial edge inside the window
    x[:, 0] = x[:, 0].abs() * x[:, 1]
    x[:, 2] = x[:, 2].abs() + 1
    a, b = old(x), new(x)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])
    torch.nn.init.normal_(new.fuse[0].weight, std=0.5)
    torch.nn.init.normal_(new.head_edge.weight, std=0.5)
    region = (slice(6, 20), slice(2, 9), slice(3, 11))
    mu, var = new(x)
    mu_r, var_r = new(x, region=region)
    assert torch.allclose(mu[(slice(None),) + region], mu_r) and torch.allclose(var[(slice(None),) + region], var_r)
    # the gated head acts only where a side is missing: mid-window frames with full windows are unchanged
    ref = SlabBlindSpotNet(ModelConfig(**kw, edge_features=True)).double().eval()
    ref.load_state_dict(new.state_dict())
    torch.nn.init.zeros_(ref.head_edge.weight)
    torch.nn.init.zeros_(ref.head_edge.bias)
    xm = x.clone()
    xm[:, 1] = 1.0
    xm[:, 0] = x[:, 0].abs() + 0.1
    big = dict(kw, window_reach=2)  # windows that fit inside 24 frames: full on both sides at frames 5-18
    m1 = SlabBlindSpotNet(ModelConfig(**big, edge_features=True)).double().eval()
    m1.load_state_dict(new.state_dict())
    m2 = SlabBlindSpotNet(ModelConfig(**big, edge_features=True)).double().eval()
    m2.load_state_dict(ref.state_dict())
    assert torch.allclose(m1(xm)[0][:, 5:19], m2(xm)[0][:, 5:19])
    assert not torch.allclose(m1(xm)[0][:, -2:], m2(xm)[0][:, -2:])


@pytest.mark.parametrize("grow", ["motion", "box"])
def test_edge_features_warm_start_into_more_inputs(grow):
    """A checkpoint with edge features warm-starts a model with more fusion
    inputs exactly: the new ones (motion features, or the box's half-planes and
    pooled features) come before the edge features, so load_compatible matches
    the columns by input part (``ModelConfig.fusion_inputs``)."""
    from slap2_f0.train import load_compatible

    torch.manual_seed(11)
    kw = dict(gap=2, width=6, temporal_dilations=[1, 2], spatial_dilations=[1, 1], window_reach=6, full_res_blocks=1,
              edge_features=True, edge_near=4)
    box = dict(box_radius=2, hp_width=4, hp_temporal_dilations=[1, 2], hp_spatial_dilations=[1, 1],
               hp_full_res_blocks=1, pool_scales=[3, 0])
    src, dst = (dict(kw, **box), dict(kw, **box, motion_features=True)) if grow == "motion" else (kw, dict(kw, **box))
    old = SlabBlindSpotNet(ModelConfig(**src)).double().eval()
    _activate_box(old) if grow == "motion" else _activate(old)
    torch.nn.init.normal_(old.fuse[0].weight, std=0.5)  # the edge columns count too
    torch.nn.init.normal_(old.head_edge.weight, std=0.5)
    new = SlabBlindSpotNet(ModelConfig(**dst)).double().eval()
    msg = load_compatible(new, old.state_dict(), old.cfg)
    assert "fuse.0.weight" in msg
    x = torch.randn(1, 5, 24, 12, 12, dtype=torch.float64)
    x[:, 1] = 1.0
    x[:, 1, 18:] = 0  # a trial edge inside the window
    x[:, 0] = x[:, 0].abs() * x[:, 1]
    x[:, 2] = x[:, 2].abs() + 1
    a, b = old(x), new(x)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])


def test_load_compatible_refuses_other_layout_of_equal_width():
    """A checkpoint whose fusion inputs differ from the model's is refused even
    when the first fusion layer has the same width: 6 motion channels
    (pool_scales [0]) are not the 6 edge channels."""
    from slap2_f0.train import load_compatible

    kw = dict(gap=2, width=6, temporal_dilations=[1, 2], spatial_dilations=[1, 1], window_reach=6, full_res_blocks=1,
              box_radius=2, hp_width=4, hp_temporal_dilations=[1, 2], hp_spatial_dilations=[1, 1],
              hp_full_res_blocks=1, pool_scales=[0])
    old = SlabBlindSpotNet(ModelConfig(**kw, motion_features=True))
    new = SlabBlindSpotNet(ModelConfig(**kw, edge_features=True))
    assert old.fuse[0].weight.shape == new.fuse[0].weight.shape
    with pytest.raises(SystemExit, match="motion"):
        load_compatible(new, old.state_dict(), old.cfg)


def _activate_pointwise(model):
    """Non-zero residual scales of the pointwise blocks (they start at 0)."""
    from slap2_f0.model import PointwiseBlock

    for m in model.modules():
        if isinstance(m, PointwiseBlock):
            m.alpha.data.fill_(0.5)
            torch.nn.init.normal_(m.w2.weight, std=0.5)


@pytest.mark.parametrize("which", [0, 1])  # mu, var
@pytest.mark.parametrize("box", [False, True])
def test_pointwise_depth_keeps_the_blind_spot(which, box):
    """branch_mlp and fuse_blocks add depth without reach: the blind spot (slab
    or box) is unchanged and the inputs used are the same as without them."""
    torch.manual_seed(3)
    kw = dict(gap=3, width=6, temporal_dilations=[1, 2, 4], spatial_dilations=[1, 2, 1], window_reach=10,
              full_res_blocks=1)
    if box:
        kw.update(box_radius=2, hp_width=6, hp_temporal_dilations=[1, 2, 1], hp_spatial_dilations=[1, 1, 2],
                  hp_full_res_blocks=1, pool_scales=[3, 0])
    deep_cfg = ModelConfig(**kw, branch_mlp=2, fuse_blocks=3)
    assert deep_cfg.temporal_reach == ModelConfig(**kw).temporal_reach
    model = SlabBlindSpotNet(deep_cfg).double()
    (_activate_box if box else _activate)(model)
    _activate_pointwise(model)
    x = torch.randn(1, deep_cfg.n_inputs, 40, 23, 25, dtype=torch.float64)
    x[:, 1] = (torch.rand(1, 40, 23, 25) > 0.1).double()
    x[:, 0] = x[:, 0] * x[:, 1]
    x[:, 2] = x[:, 2].abs() + 1
    x.requires_grad_(True)
    t0, r0, c0 = 20, 11, 12
    model(x)[which][0, t0, r0, c0].backward()
    g = x.grad[0, :2].abs().sum(0)
    gap, rho = deep_cfg.gap, (2 if box else None)
    if box:
        assert torch.count_nonzero(g[t0 - gap: t0 + gap + 1, r0 - rho: r0 + rho + 1, c0 - rho: c0 + rho + 1]) == 0
        assert g[t0, r0 + rho + 1, c0] > 0
    else:
        assert torch.count_nonzero(g[t0 - gap: t0 + gap + 1]) == 0
    assert g[t0 - gap - 1, r0, c0] > 0 and g[t0 + gap + 1, r0, c0] > 0
    reach = deep_cfg.gap + 1 + deep_cfg.temporal_reach  # nothing beyond the reach either
    assert torch.count_nonzero(g[: max(0, t0 - reach)]) == 0 and torch.count_nonzero(g[t0 + reach + 1:]) == 0


def test_pointwise_depth_warm_starts_exactly():
    """A checkpoint without pointwise blocks in a model with them: the same
    outputs at the start, every old weight copied and every new block counted."""
    from slap2_f0.train import load_compatible

    torch.manual_seed(5)
    kw = dict(gap=2, width=6, temporal_dilations=[1, 2], spatial_dilations=[1, 1], window_reach=6, full_res_blocks=1,
              box_radius=2, hp_width=4, hp_temporal_dilations=[1, 2], hp_spatial_dilations=[1, 1],
              hp_full_res_blocks=1, pool_scales=[3, 0])
    old = SlabBlindSpotNet(ModelConfig(**kw)).double()
    _activate_box(old)
    deep = SlabBlindSpotNet(ModelConfig(**kw, branch_mlp=1, fuse_blocks=2)).double()
    load_compatible(deep, old.state_dict(), old.cfg)
    own = deep.state_dict()
    assert all(torch.equal(own[k], v) for k, v in old.state_dict().items())  # old keys unchanged
    assert any(k.startswith("fuse_more.1.") for k in own) and any(".mlp.0.w1." in k for k in own)
    x = torch.randn(1, 5, 20, 12, 12, dtype=torch.float64)
    x[:, 2] = x[:, 2].abs() + 1
    a, b = old(x), deep(x)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_pointwise_depth_region_matches_full():
    torch.manual_seed(6)
    cfg = ModelConfig(gap=2, width=6, temporal_dilations=[1, 2, 4], spatial_dilations=[1, 1, 2], full_res_blocks=1,
                      window_reach=8, branch_mlp=1, fuse_blocks=2)
    model = SlabBlindSpotNet(cfg).double().eval()
    _activate(model)
    _activate_pointwise(model)
    x = torch.randn(1, cfg.n_inputs, 30, 14, 15, dtype=torch.float64)
    x[:, 2] = x[:, 2].abs() + 1
    region = (slice(10, 20), slice(3, 11), slice(4, 12))
    full, part = model(x), model(x, region=region)
    for f, p in zip(full, part):
        assert torch.allclose(f[(slice(None),) + region], p)
