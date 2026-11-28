"""Multiply-accumulates per output voxel of the F0 network (s6_ft config), for
the tiles engine (predict_trial), the streaming engine and low-rate designs
(design note, slap2-F0-network#33).

usage: PYTHONPATH=code python tools/inference_speed/macs.py CKPT VALID_NPY [VALID_NPY ...]
(VALID_NPY: a trial's valid mask, (row, col) bool, in the network's orientation)
Counts every convolution's MACs per voxel of the resolution it runs at
(full or half in space, full or 1/k in time); element-wise work and the
pooled features are not counted (memory traffic, see the profile).
"""

import json
import sys

import numpy as np
import torch

from slap2_f0.model import config_from_dict  # noqa: E402
from slap2_f0.stream import plan_rects  # noqa: E402
from slap2_f0.predict import tile_origins  # noqa: E402


def conv_macs(cin, cout, k):
    """MACs per output voxel of a convolution with kernel volume k."""
    return cin * cout * k


def branch_macs(cfg):
    """Per input voxel: one causal branch (full-res voxels) and one half-plane
    direction, and the fusion per output voxel."""
    w, mlp = cfg.width, cfg.mlp_hidden
    f = cfg.spatial_factor
    k2 = cfg.spatial_kernel ** 2
    blk = conv_macs(w, w, k2) + conv_macs(w, w, 3) + cfg.branch_mlp * 2 * conv_macs(w, mlp * w, 1)
    nfull = min(cfg.full_res_blocks, len(cfg.temporal_dilations))
    nlow = len(cfg.temporal_dilations) - nfull
    causal = conv_macs(cfg.n_inputs, w, 1) + nfull * blk + (nlow * blk + conv_macs(w, w, f * f)) / f ** 2
    hw = cfg.hp_width
    hblk = conv_macs(hw, hw, 6) + conv_macs(hw, hw, 3) + cfg.branch_mlp * 2 * conv_macs(hw, mlp * hw, 1)
    hn = min(cfg.hp_full_res_blocks, len(cfg.hp_temporal_dilations))
    hl = len(cfg.hp_temporal_dilations) - hn
    hp = conv_macs(cfg.n_inputs, hw, 1) + hn * hblk + (hl * hblk + conv_macs(hw, hw, 4)) / 4
    n_in = sum(n for _, n in cfg.fusion_inputs())
    fuse = conv_macs(n_in, 2 * w, 1) + conv_macs(2 * w, w, 1) + cfg.fuse_blocks * 2 * conv_macs(w, mlp * w, 1) + 3 * w
    return {"causal_per_voxel": causal, "hp_per_voxel": hp, "fusion_per_output_voxel": fuse,
            "causal_block": blk, "hp_block": hblk}


def main():
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    cfg = config_from_dict(ck["config"])
    w = ck["window"]
    tR, sR, op = w["t_reach"], w["s_reach"], w["out_px"]
    b = branch_macs(cfg)
    R = cfg.hp_temporal_reach
    out = {"per_voxel": b, "fields": {}}
    T = 2254
    for f in sys.argv[2:]:
        v = np.load(f)
        N = int(v.sum())
        ntile = sum(len(c) for _, c in tile_origins(v, op))
        ns = op + 2 * sR
        rects = plan_rects(v, op, sR, 0)
        A = sum(r.area for r in rects)
        rec = {"valid_px": N, "tiles": ntile, "rect_px": A, "tile_window_px": ntile * ns * ns}
        for name, nf in (("tiles_512", 512), ("tiles_whole", T)):
            nt = -(-T // nf)
            frames = nt * (nf + 2 * tR)  # every branch runs on the whole window
            hpf = nt * (nf + 2 * R)
            m = ntile * ns * ns * (2 * frames * b["causal_per_voxel"] + 4 * hpf * b["hp_per_voxel"]) \
                + ntile * op * op * T * b["fusion_per_output_voxel"]
            rec[name] = m / (N * T)
        frames = T + tR - (cfg.gap + 1)  # each causal branch from its trial edge to the far output
        for L in (128, 256, 512):  # --stream_layout ncdhw: half-planes per chunk with their reach as halo
            hpf = -(-T // L) * (L + 2 * max(R, 4))
            m = A * (2 * frames * b["causal_per_voxel"] + 4 * hpf * b["hp_per_voxel"]) \
                + N * T * b["fusion_per_output_voxel"]
            rec[f"stream_ncdhw_{L}"] = m / (N * T)
        # --stream_layout cl (default): half-planes streamed once over frames -R .. T + R - 1, any chunk
        m = A * (2 * frames * b["causal_per_voxel"] + 4 * (T + 2 * R) * b["hp_per_voxel"]) \
            + N * T * b["fusion_per_output_voxel"]
        rec["stream_cl"] = m / (N * T)
        # low-rate design, factor k in time after the first full-rate block of each branch, half-planes
        # and fusion at the low rate (output interpolated), streaming geometry, chunk 256
        for k in (2, 4):
            nfull = 1
            blk = b["causal_block"]
            f2 = cfg.spatial_factor ** 2
            causal_low = conv_macs(cfg.n_inputs, cfg.width, 1) + nfull * blk + \
                ((len(cfg.temporal_dilations) - nfull - 1) * blk / f2 + blk) / k
            frames = T + tR
            m = (A * (2 * frames * causal_low + 4 * (T / k + 64) * b["hp_per_voxel"])
                 + N * T / k * b["fusion_per_output_voxel"])
            rec[f"lowrate_k{k}_stream"] = m / (N * T)
        out["fields"][f.split("/")[-1]] = rec
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
