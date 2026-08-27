# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
"""
Multi-Scale Deformable Attention Module
"""

from __future__ import absolute_import, division, print_function

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from rfdetr.models.ops.functions import ms_deform_attn_core_pytorch


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module"""

    def __init__(
        self,
        d_model=256,
        n_levels=4,
        n_heads=8,
        n_points=4,
        scale_routing=False,
        scale_routing_mode="legacy",
        last_level_initial_bias=0.0,
    ):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads, but got {} and {}".format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the "
                "dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.last_level_initial_bias = float(last_level_initial_bias)
        self.scale_routing = scale_routing
        if scale_routing_mode not in {"legacy", "cell"}:
            raise ValueError(f"Unsupported scale_routing_mode: {scale_routing_mode}")
        self.scale_routing_mode = scale_routing_mode

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        if scale_routing:
            routing_dim = max(d_model // 4, 32)
            routing_outputs = n_heads * n_levels if scale_routing_mode == "legacy" else n_heads
            # Keep all pre-existing model parameters bitwise aligned with a no-router
            # run under the same seed, so routing ablations change only the router.
            with torch.random.fork_rng(devices=[]):
                self.scale_router = nn.Sequential(
                    nn.Linear(d_model + 4, routing_dim),
                    nn.GELU(),
                    nn.Linear(routing_dim, routing_outputs),
                )
        else:
            self.scale_router = None

        self._reset_parameters()

        self._export = False

    def export(self):
        """export mode"""
        self._export = True

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        if self.last_level_initial_bias != 0.0:
            with torch.no_grad():
                attention_bias = self.attention_weights.bias.view(self.n_heads, self.n_levels, self.n_points)
                attention_bias[:, -1].fill_(self.last_level_initial_bias)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)
        if self.scale_router is not None:
            with torch.random.fork_rng(devices=[]):
                xavier_uniform_(self.scale_router[0].weight.data)
                constant_(self.scale_router[0].bias.data, 0.0)
                constant_(self.scale_router[-1].weight.data, 0.0)
                constant_(self.scale_router[-1].bias.data, 0.0)

    def disable_scale_routing(self):
        self.scale_routing = False
        self.scale_router = None

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        r"""
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        if self.scale_router is not None:
            if reference_points.shape[-1] != 4:
                raise ValueError("Scale routing requires 4D reference boxes.")
            if self.scale_routing_mode == "legacy":
                log_wh = reference_points[..., 2:].clamp_min(1e-6).log().mean(dim=2)
                box_geometry = torch.cat(
                    [
                        log_wh,
                        log_wh.sum(dim=-1, keepdim=True),
                        log_wh[..., :1] - log_wh[..., 1:],
                    ],
                    dim=-1,
                ).to(dtype=query.dtype)
                level_bias = self.scale_router(torch.cat([query, box_geometry], dim=-1))
                level_bias = level_bias.view(N, Len_q, self.n_heads, self.n_levels, 1)
            else:
                level_wh = torch.stack(
                    [input_spatial_shapes[:, 1], input_spatial_shapes[:, 0]],
                    dim=-1,
                ).to(device=reference_points.device, dtype=reference_points.dtype)
                log_cell_wh = (reference_points[..., 2:] * level_wh[None, None]).clamp_min(1e-6).log()
                box_geometry = torch.cat(
                    [
                        log_cell_wh,
                        log_cell_wh.sum(dim=-1, keepdim=True),
                        log_cell_wh[..., :1] - log_cell_wh[..., 1:],
                    ],
                    dim=-1,
                ).to(dtype=query.dtype)
                routing_query = query[:, :, None].expand(-1, -1, self.n_levels, -1)
                level_bias = self.scale_router(torch.cat([routing_query, box_geometry], dim=-1))
                level_bias = level_bias.permute(0, 1, 3, 2).unsqueeze(-1)
            level_bias = level_bias.expand(-1, -1, -1, -1, self.n_points)
            attention_weights = attention_weights + level_bias.flatten(-2)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
            )
        attention_weights = F.softmax(attention_weights, -1)

        value = value.transpose(1, 2).contiguous().view(N, self.n_heads, self.d_model // self.n_heads, Len_in)
        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output
