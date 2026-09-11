# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from ViTDet (https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Projector
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """
        LayerNorm forward
        TODO: this is a hack to avoid overflow when using fp16
        """
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.size(3),), self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


def get_activation(name, inplace=False):
    """get activation"""
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name in ["LeakyReLU", "leakyrelu", "lrelu"]:
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name is None:
        module = nn.Identity()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class ConvX(nn.Module):
    """Conv-bn module"""

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel=3,
        stride=1,
        groups=1,
        dilation=1,
        act="relu",
        layer_norm=False,
        rms_norm=False,
    ):
        super(ConvX, self).__init__()
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel)
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        if rms_norm:
            self.bn = nn.RMSNorm(out_planes)
        else:
            self.bn = get_norm("LN", out_planes) if layer_norm else nn.BatchNorm2d(out_planes)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        """forward"""
        out = self.act(self.bn(self.conv(x.contiguous())))
        return out


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, act="silu", layer_norm=False, rms_norm=False):
        """ch_in, ch_out, shortcut, groups, kernels, expand"""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, c_, k[0], 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX(c_, c2, k[1], 1, groups=g, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act="silu", layer_norm=False, rms_norm=False):
        """ch_in, ch_out, number, shortcut, groups, expansion"""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, 2 * self.c, 1, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX(
            (2 + n) * self.c, c2, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm
        )  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class DirectFusionP5(nn.Module):
    """Build a low-cost P5 directly from all ViT hidden states."""

    def __init__(
        self,
        in_channels,
        out_channels,
        rank_channels=None,
        layer_norm=False,
        rms_norm=False,
    ):
        super().__init__()
        if rank_channels is None:
            rank_channels = max(out_channels // len(in_channels), 32)
        self.projections = nn.ModuleList(
            [
                ConvX(
                    in_dim,
                    rank_channels,
                    kernel=1,
                    act="silu",
                    layer_norm=layer_norm,
                    rms_norm=rms_norm,
                )
                for in_dim in in_channels
            ]
        )
        fused_channels = rank_channels * len(in_channels)
        self.downsample = ConvX(
            fused_channels,
            fused_channels,
            kernel=3,
            stride=2,
            groups=fused_channels,
            act=None,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )
        self.mix = ConvX(
            fused_channels,
            out_channels,
            kernel=1,
            act=None,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.output_norm = get_norm("LN", out_channels) if layer_norm else nn.Identity()

    def forward(self, features):
        if len(features) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} features, got {len(features)}")
        projected = [projection(feature) for projection, feature in zip(self.projections, features)]
        fused = torch.cat(projected, dim=1)
        low_pass = F.avg_pool2d(fused, kernel_size=2, stride=2, ceil_mode=True)
        learned = self.downsample(fused)
        fused = low_pass + torch.tanh(self.residual_scale) * learned
        return self.output_norm(self.mix(fused))


class LowRankResidualPreMix(nn.Module):
    """Expose grouped spatial filters to a cheap, learnable cross-group signal."""

    def __init__(self, channels, rank_channels=64):
        super().__init__()
        rank_channels = min(rank_channels, channels)
        self.down = nn.Conv2d(channels, rank_channels, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.up = nn.Conv2d(rank_channels, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class FullResidualPreMix(nn.Module):
    """Learn unrestricted cross-group mixing from an exact identity start."""

    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.mix.weight)

    def forward(self, x):
        return x + self.mix(x)


class ResidualFusionP5(nn.Module):
    """Add learnable post-fusion capacity without perturbing its initial output."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks,
        layer_norm=False,
        rms_norm=False,
    ):
        super().__init__()
        self.fusion = DirectFusionP5(
            in_channels,
            out_channels,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )
        self.refine = C2f(
            out_channels,
            out_channels,
            num_blocks,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )
        self.refine_gate = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, features):
        fused = self.fusion(features)
        refined = self.refine(fused)
        return fused + torch.tanh(self.refine_gate) * refined


class MultiScaleProjector(nn.Module):
    """
    This module implements MultiScaleProjector in :paper:`lwdetr`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factors,
        p5_mode="full",
        num_blocks=3,
        layer_norm=False,
        rms_norm=False,
        survival_prob=1.0,
        force_drop_last_n_features=0,
        source_indices_by_scale=None,
        source_selection_mode="mask",
        c2f_blocks_by_scale=None,
        resample_share_mode="none",
        p4_depth_prior=None,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
        """
        super(MultiScaleProjector, self).__init__()

        self.scale_factors = scale_factors
        if p5_mode not in {
            "full",
            "pool",
            "dwconv",
            "group2",
            "group2_mix",
            "group2_mix128",
            "group2_fullmix",
            "group2_first_full",
            "group2_first_full_dw",
            "group2_first2_full",
            "group2_last_full",
            "group4",
            "group8",
            "fusion",
            "fusion_wide",
            "fusion_refine",
            "fusion_residual",
        }:
            raise ValueError(f"Unsupported P5 projector mode: {p5_mode}")
        if p5_mode != "full":
            if 0.5 not in scale_factors:
                raise ValueError("A lightweight P5 mode requires P5 in projector_scale.")
            p5_index = scale_factors.index(0.5)
            if p5_index == 0 or scale_factors[p5_index - 1] != 1.0:
                raise ValueError("A lightweight P5 mode requires P4 immediately before P5.")
        self.p5_mode = p5_mode
        self.survival_prob = survival_prob
        self.force_drop_last_n_features = force_drop_last_n_features
        if source_selection_mode not in {"mask", "prune"}:
            raise ValueError("source_selection_mode must be 'mask' or 'prune'.")
        self.source_selection_mode = source_selection_mode
        if resample_share_mode not in {"none", "p3", "p5", "p3_p5"}:
            raise ValueError(
                "resample_share_mode must be one of: none, p3, p5, p3_p5."
            )
        self.resample_share_mode = resample_share_mode
        num_features = len(in_channels)
        if source_indices_by_scale is None:
            source_indices_by_scale = [list(range(num_features)) for _ in scale_factors]
        if len(source_indices_by_scale) != len(scale_factors):
            raise ValueError(
                "source_indices_by_scale must provide one index list per scale factor."
            )
        normalized_source_indices = []
        for indices in source_indices_by_scale:
            indices = list(indices)
            if not indices:
                raise ValueError("Every projector scale must retain at least one source.")
            if len(indices) != len(set(indices)):
                raise ValueError("Projector source indices must not contain duplicates.")
            if any(index < 0 or index >= num_features for index in indices):
                raise ValueError(
                    f"Projector source indices must be in [0, {num_features - 1}]."
                )
            normalized_source_indices.append(indices)
        self.source_indices_by_scale = normalized_source_indices
        if c2f_blocks_by_scale is None:
            c2f_blocks_by_scale = [num_blocks] * len(scale_factors)
        if len(c2f_blocks_by_scale) != len(scale_factors):
            raise ValueError(
                "c2f_blocks_by_scale must provide one block count per scale factor."
            )
        if any(blocks < 0 for blocks in c2f_blocks_by_scale):
            raise ValueError("C2f block counts must be non-negative.")
        self.c2f_blocks_by_scale = list(c2f_blocks_by_scale)
        if p4_depth_prior is not None:
            if 1.0 not in scale_factors:
                raise ValueError("A P4 depth prior requires P4 in projector_scale.")
            if len(p4_depth_prior) != num_features:
                raise ValueError(
                    "p4_depth_prior must provide one fixed weight per backbone feature."
                )
            if any(not np.isfinite(weight) for weight in p4_depth_prior):
                raise ValueError("p4_depth_prior weights must be finite.")
            if any(weight <= 0 for weight in p4_depth_prior):
                raise ValueError("p4_depth_prior weights must be positive.")
            # Deliberately keep the prior as immutable Python data: it is neither a
            # Parameter nor a buffer and therefore cannot enter the optimizer or
            # alter checkpoint state. Multiplication below preserves input dtype.
            p4_depth_prior = tuple(float(weight) for weight in p4_depth_prior)
        self.p4_depth_prior = p4_depth_prior

        stages_sampling = []
        stages = []
        stage_sources = []
        stage_input_indices = []
        # use_bias = norm == ""
        self.use_extra_pool = False
        for scale_index, scale in enumerate(scale_factors):
            selected_sources = self.source_indices_by_scale[scale_index]
            stage_num_blocks = self.c2f_blocks_by_scale[scale_index]
            if (
                scale == 0.5
                and self.p5_mode
                in {"pool", "dwconv", "fusion", "fusion_wide", "fusion_refine", "fusion_residual"}
                and len(selected_sources) != num_features
            ):
                raise ValueError(
                    "Per-scale source masking is not supported by replacement-style P5 modes."
                )
            stage_sampling = []
            for in_dim in in_channels:
                layers = []

                # if in_dim > 512:
                #     layers.append(ConvX(in_dim, in_dim // 2, kernel=1))
                #     in_dim = in_dim // 2

                if scale == 4.0:
                    layers.extend(
                        [
                            nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                            get_norm("LN", in_dim // 2),
                            nn.GELU(),
                            nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                        ]
                    )
                    # in_dim // 4
                elif scale == 2.0:
                    # a hack to reduce the FLOPs and Params when the dimension of output feature is too large
                    # if in_dim > 512:
                    #     layers = [
                    #         ConvX(in_dim, in_dim // 2, kernel=1),
                    #         nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                    #     ]
                    #     out_dim = in_dim // 4
                    # else:
                    layers.extend(
                        [
                            nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                        ]
                    )
                    # in_dim // 2
                elif scale == 1.0:
                    pass
                elif scale == 0.5:
                    layers.extend(
                        [
                            ConvX(in_dim, in_dim, 3, 2, layer_norm=layer_norm),
                        ]
                    )
                elif scale == 0.25:
                    self.use_extra_pool = True
                    continue
                else:
                    raise NotImplementedError("Unsupported scale_factor:{}".format(scale))
                layers = nn.Sequential(*layers)
                stage_sampling.append(layers)
            if scale == 0.25:
                continue

            in_dim = int(sum(in_channel // max(1, scale) for in_channel in in_channels))
            layers = [
                C2f(in_dim, out_channels, num_blocks, layer_norm=layer_norm),
                get_norm("LN", out_channels),
            ]
            full_stage = nn.Sequential(*layers)
            active_stage = full_stage
            if (
                self.source_selection_mode == "prune"
                and len(selected_sources) != num_features
            ) or stage_num_blocks != num_blocks:
                selected_in_dim = int(
                    sum(
                        in_channels[index] // max(1, scale)
                        for index in (
                            selected_sources
                            if self.source_selection_mode == "prune"
                            else range(num_features)
                        )
                    )
                )
                with torch.random.fork_rng(devices=[]):
                    active_stage = nn.Sequential(
                        C2f(
                            selected_in_dim,
                            out_channels,
                            stage_num_blocks,
                            layer_norm=layer_norm,
                        ),
                        get_norm("LN", out_channels),
                    )

            if scale == 0.5 and self.p5_mode in {
                "group2",
                "group2_mix",
                "group2_mix128",
                "group2_fullmix",
                "group2_first_full",
                "group2_first_full_dw",
                "group2_first2_full",
                "group2_last_full",
                "group4",
                "group8",
            }:
                group_count = {
                    "group2": 2,
                    "group2_mix": 2,
                    "group2_mix128": 2,
                    "group2_fullmix": 2,
                    "group2_first_full": 2,
                    "group2_first_full_dw": 2,
                    "group2_first2_full": 2,
                    "group2_last_full": 2,
                    "group4": 4,
                    "group8": 8,
                }[self.p5_mode]
                if any(in_dim % group_count != 0 for in_dim in in_channels):
                    raise ValueError(
                        f"{self.p5_mode} requires every input channel count to be "
                        f"divisible by {group_count}."
                    )
                with torch.random.fork_rng(devices=[]):
                    if self.p5_mode == "group2_first_full_dw":
                        grouped_convs = [
                            nn.Sequential(
                                nn.Conv2d(
                                    in_dim,
                                    in_dim,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=in_dim,
                                    bias=False,
                                ),
                                ConvX(
                                    in_dim,
                                    in_dim,
                                    kernel=1,
                                    layer_norm=layer_norm,
                                    rms_norm=rms_norm,
                                ),
                            )
                            for in_dim in in_channels
                        ]
                    else:
                        grouped_convs = [
                            ConvX(
                                in_dim,
                                in_dim,
                                kernel=3,
                                stride=2,
                                groups=group_count,
                                layer_norm=layer_norm,
                                rms_norm=rms_norm,
                            )
                            for in_dim in in_channels
                        ]
                    if self.p5_mode in {"group2_mix", "group2_mix128"}:
                        rank_channels = 128 if self.p5_mode == "group2_mix128" else 64
                        pre_mixers = [
                            LowRankResidualPreMix(in_dim, rank_channels=rank_channels)
                            for in_dim in in_channels
                        ]
                        grouped_sampling = nn.ModuleList(
                            [
                                nn.Sequential(pre_mixer, grouped_conv)
                                for pre_mixer, grouped_conv in zip(pre_mixers, grouped_convs)
                            ]
                        )
                    elif self.p5_mode == "group2_fullmix":
                        grouped_sampling = nn.ModuleList(
                            [
                                nn.Sequential(FullResidualPreMix(in_dim), grouped_conv)
                                for in_dim, grouped_conv in zip(in_channels, grouped_convs)
                            ]
                        )
                    elif self.p5_mode in {
                        "group2_first_full",
                        "group2_first_full_dw",
                        "group2_first2_full",
                        "group2_last_full",
                    }:
                        full_indexes = {
                            "group2_first_full": {0},
                            "group2_first_full_dw": {0},
                            "group2_first2_full": {0, 1},
                            "group2_last_full": {len(grouped_convs) - 1},
                        }[self.p5_mode]
                        grouped_sampling = nn.ModuleList(
                            [
                                full_sampler
                                if index in full_indexes
                                else (
                                    grouped_conv
                                    if isinstance(grouped_conv, nn.Sequential)
                                    else nn.Sequential(grouped_conv)
                                )
                                for index, (full_sampler, grouped_conv) in enumerate(
                                    zip(stage_sampling, grouped_convs)
                                )
                            ]
                        )
                    else:
                        grouped_sampling = nn.ModuleList(
                            [
                                nn.Sequential(grouped_conv)
                                for grouped_conv in grouped_convs
                            ]
                        )
                if self.source_selection_mode == "prune":
                    grouped_sampling = nn.ModuleList(
                        [grouped_sampling[index] for index in selected_sources]
                    )
                    input_indices = selected_sources
                else:
                    input_indices = list(range(num_features))
                stages_sampling.append(grouped_sampling)
                stages.append(active_stage)
                stage_sources.append("backbone")
                stage_input_indices.append(input_indices)
            elif scale == 0.5 and self.p5_mode != "full":
                # Constructing the full branch above intentionally advances the RNG
                # exactly as the baseline does. The lightweight branch is initialized
                # in a fork so later shared modules keep identical initial weights.
                with torch.random.fork_rng(devices=[]):
                    if self.p5_mode == "pool":
                        lightweight_stage = nn.Sequential(
                            nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
                            get_norm("LN", out_channels) if layer_norm else nn.Identity(),
                        )
                        stage_source = "previous"
                    elif self.p5_mode == "dwconv":
                        lightweight_stage = nn.Sequential(
                            ConvX(
                                out_channels,
                                out_channels,
                                kernel=3,
                                stride=2,
                                groups=out_channels,
                                act="silu",
                                layer_norm=layer_norm,
                                rms_norm=rms_norm,
                            ),
                            ConvX(
                                out_channels,
                                out_channels,
                                kernel=1,
                                act=None,
                                layer_norm=layer_norm,
                                rms_norm=rms_norm,
                            ),
                        )
                        stage_source = "previous"
                    elif self.p5_mode == "fusion_residual":
                        lightweight_stage = ResidualFusionP5(
                            in_channels,
                            out_channels,
                            stage_num_blocks,
                            layer_norm=layer_norm,
                            rms_norm=rms_norm,
                        )
                        stage_source = "features"
                    else:
                        direct_fusion = DirectFusionP5(
                            in_channels,
                            out_channels,
                            rank_channels=out_channels // 2 if self.p5_mode == "fusion_wide" else None,
                            layer_norm=layer_norm,
                            rms_norm=rms_norm,
                        )
                        if self.p5_mode == "fusion_refine":
                            lightweight_stage = nn.Sequential(
                                direct_fusion,
                                C2f(
                                    out_channels,
                                    out_channels,
                                    stage_num_blocks,
                                    layer_norm=layer_norm,
                                    rms_norm=rms_norm,
                                ),
                                get_norm("LN", out_channels),
                            )
                        else:
                            lightweight_stage = direct_fusion
                        stage_source = "features"
                stages_sampling.append(nn.ModuleList())
                stages.append(lightweight_stage)
                stage_sources.append(stage_source)
                stage_input_indices.append(list(range(num_features)))
            else:
                if self.source_selection_mode == "prune":
                    stage_sampling = [stage_sampling[index] for index in selected_sources]
                    input_indices = selected_sources
                else:
                    input_indices = list(range(num_features))
                stages_sampling.append(nn.ModuleList(stage_sampling))
                stages.append(active_stage)
                stage_sources.append("backbone")
                stage_input_indices.append(input_indices)

        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)
        self.stage_sources = stage_sources
        self.stage_input_indices = stage_input_indices
        self._share_resampling_kernels()

    @staticmethod
    def _conv_kernels(module):
        kernels = []

        def visit(operation):
            if isinstance(operation, ConvX):
                kernels.append(operation.conv)
            elif isinstance(operation, (nn.Conv2d, nn.ConvTranspose2d)):
                kernels.append(operation)
            else:
                for child in operation.children():
                    visit(child)

        visit(module)
        return kernels

    def _share_scale_kernels(self, scale):
        if scale not in self.scale_factors:
            raise ValueError(
                f"Requested resampling weight sharing for missing scale factor {scale}."
            )
        stage_index = self.scale_factors.index(scale)
        branches = self.stages_sampling[stage_index]
        groups = {}
        for branch in branches:
            for kernel in self._conv_kernels(branch):
                signature = (
                    type(kernel),
                    tuple(kernel.weight.shape),
                    kernel.groups,
                    kernel.stride,
                )
                groups.setdefault(signature, []).append(kernel)

        shared_groups = 0
        for kernels in groups.values():
            if len(kernels) < 2:
                continue
            shared_weight = kernels[0].weight
            for kernel in kernels[1:]:
                # Keep source-specific bias and normalization while sharing geometry.
                kernel.weight = shared_weight
            shared_groups += 1
        if shared_groups == 0:
            raise ValueError(
                f"No compatible resampling kernels are available to share at scale {scale}."
            )

    def _share_resampling_kernels(self):
        if self.resample_share_mode in {"p3", "p3_p5"}:
            self._share_scale_kernels(2.0)
        if self.resample_share_mode in {"p5", "p3_p5"}:
            self._share_scale_kernels(0.5)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        num_features = len(x)
        if self.survival_prob < 1.0 and self.training:
            final_drop_prob = 1 - self.survival_prob
            drop_p = np.random.uniform()
            for i in range(1, num_features):
                critical_drop_prob = i * (final_drop_prob / (num_features - 1))
                if drop_p < critical_drop_prob:
                    x[i][:] = 0
        elif self.force_drop_last_n_features > 0:
            for i in range(self.force_drop_last_n_features):
                # don't do it inplace to ensure the compiler can optimize out the backbone layers
                x[-(i + 1)] = torch.zeros_like(x[-(i + 1)])

        results = []
        # x list of len(out_features_indexes)
        for i, stage in enumerate(self.stages):
            if self.stage_sources[i] == "previous":
                results.append(stage(results[-1]))
                continue
            if self.stage_sources[i] == "features":
                results.append(stage(x))
                continue
            feat_fuse = []
            selected_sources = set(self.source_indices_by_scale[i])
            for source_index, stage_sampling in zip(
                self.stage_input_indices[i], self.stages_sampling[i]
            ):
                sampled = stage_sampling(x[source_index])
                # Fixed P4 depth prior: apply to each independently sampled P4
                # contribution before the original concatenation/C2f fusion.
                if self.scale_factors[i] == 1.0 and self.p4_depth_prior is not None:
                    sampled = sampled * self.p4_depth_prior[source_index]
                if (
                    self.source_selection_mode == "mask"
                    and source_index not in selected_sources
                ):
                    sampled = sampled * 0
                feat_fuse.append(sampled)
            if len(feat_fuse) > 1:
                feat_fuse = torch.cat(feat_fuse, dim=1)
            else:
                feat_fuse = feat_fuse[0]
            results.append(stage(feat_fuse))
        if self.use_extra_pool:
            results.append(F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0))
        return results


class SimpleProjector(nn.Module):
    def __init__(self, in_dim, out_dim, factor_kernel=False):
        super(SimpleProjector, self).__init__()
        if not factor_kernel:
            self.convx1 = ConvX(in_dim, in_dim * 2, layer_norm=True, act="silu")
            self.convx2 = ConvX(in_dim * 2, out_dim, layer_norm=True, act="silu")
        else:
            self.convx1 = ConvX(in_dim, out_dim, kernel=(3, 1), layer_norm=True, act="silu")
            self.convx2 = ConvX(out_dim, out_dim, kernel=(1, 3), layer_norm=True, act="silu")
        self.ln = get_norm("LN", out_dim)

    def forward(self, x):
        """forward"""
        out = self.ln(self.convx2(self.convx1(x[0])))
        return [out]
