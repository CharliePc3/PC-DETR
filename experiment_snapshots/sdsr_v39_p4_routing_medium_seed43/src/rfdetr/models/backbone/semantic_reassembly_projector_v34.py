# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Kernel-packed two-basis P5 reassembly with separable semantic detail."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.backbone.projector import C2f, LayerNorm
from rfdetr.models.backbone.semantic_reassembly_projector_v7 import (
    BidirectionalScaleCalibration,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v10 import (
    ExactScaleFusion,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v22 import (
    _largest_group_divisor,
)


class PackedSeparableTwoBasisP5Fusion(nn.Module):
    """Pack layerwise sampling while preserving the v23 two-basis layout."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_blocks: int,
        anti_alias: bool,
        maximum_groups: int,
        grouped_feature_index: int = -1,
    ):
        super().__init__()
        if not in_channels:
            raise ValueError("in_channels must contain at least one feature")
        if len(set(in_channels)) != 1:
            raise ValueError("Packed P5 fusion requires equal hidden-state widths")

        self.num_features = len(in_channels)
        self.channels = in_channels[0]
        self.total_channels = self.num_features * self.channels
        self.anti_alias = anti_alias
        self.grouped_feature_index = grouped_feature_index % self.num_features

        blur_kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
        )
        self.register_buffer(
            "blur_kernel",
            (blur_kernel / blur_kernel.sum())[None, None],
            persistent=False,
        )
        self.primary = (
            None
            if anti_alias
            else nn.Conv2d(
                self.total_channels,
                self.total_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=self.total_channels,
                bias=False,
            )
        )
        self.detail = nn.Conv2d(
            self.total_channels,
            self.total_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=self.total_channels,
            bias=False,
        )

        groups = _largest_group_divisor(self.channels, maximum_groups)
        with torch.random.fork_rng(devices=[]):
            self.semantic_mixer = nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=1,
                groups=groups,
                bias=False,
            )
        channels_per_group = self.channels // groups
        with torch.no_grad():
            self.semantic_mixer.weight.zero_()
            for channel in range(self.channels):
                self.semantic_mixer.weight[
                    channel, channel % channels_per_group, 0, 0
                ] = 1.0

        norm_shape = (self.num_features, self.channels)
        self.primary_weight = nn.Parameter(torch.ones(norm_shape))
        self.primary_bias = nn.Parameter(torch.zeros(norm_shape))
        self.detail_weight = nn.Parameter(torch.ones(norm_shape))
        self.detail_bias = nn.Parameter(torch.zeros(norm_shape))
        self.detail_scale = nn.Parameter(
            torch.full((1, self.num_features, self.channels, 1, 1), 0.1)
        )

        self.fusion = C2f(
            2 * sum(in_channels),
            out_channels,
            n=num_blocks,
            layer_norm=True,
        )
        self.output_norm = LayerNorm(out_channels)

    def _featurewise_norm(
        self,
        feature: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        feature = feature.permute(0, 1, 3, 4, 2)
        feature = F.layer_norm(feature, (self.channels,), eps=1.0e-6)
        feature = feature * weight[None, :, None, None]
        feature = feature + bias[None, :, None, None]
        return F.silu(feature).permute(0, 1, 4, 2, 3)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {len(features)}"
            )
        packed = torch.cat(features, dim=1)
        if self.primary is None:
            primary = F.conv2d(
                packed,
                self.blur_kernel.to(dtype=packed.dtype).expand(
                    self.total_channels, 1, -1, -1
                ),
                stride=2,
                padding=1,
                groups=self.total_channels,
            )
        else:
            primary = self.primary(packed)
        detail = self.detail(packed)

        batch_size, _, height, width = primary.shape
        primary = primary.reshape(
            batch_size, self.num_features, self.channels, height, width
        )
        detail = detail.reshape(
            batch_size, self.num_features, self.channels, height, width
        )
        deep_detail = self.semantic_mixer(
            detail[:, self.grouped_feature_index].contiguous()
        )
        detail_parts = list(detail.unbind(dim=1))
        detail_parts[self.grouped_feature_index] = deep_detail
        detail = torch.stack(detail_parts, dim=1)

        primary = self._featurewise_norm(
            primary, self.primary_weight, self.primary_bias
        )
        detail = self._featurewise_norm(
            detail, self.detail_weight, self.detail_bias
        )
        detail = torch.tanh(self.detail_scale).to(detail.dtype) * detail
        sampled = torch.stack((primary, detail), dim=2).reshape(
            batch_size,
            2 * self.total_channels,
            height,
            width,
        )
        return self.output_norm(self.fusion(sampled))


class ScaleDecoupledReassemblyProjectorV34(nn.Module):
    """Retain v23 semantics with a packed and separable P5 implementation."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        levels: Sequence[str],
        rank_channels: int = 64,
        cross_scale_mode: str = "none",
        cross_scale_rank: int = 32,
        use_phase_downsample: bool = True,
        num_blocks: int = 3,
        detail_groups: int = 16,
    ):
        super().__init__()
        del rank_channels
        unsupported = set(levels) - {"P3", "P4", "P5"}
        if unsupported:
            raise ValueError(
                f"SDSR-v34 Projector does not support levels: {sorted(unsupported)}"
            )
        if not levels:
            raise ValueError("levels must contain at least one pyramid level")
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if detail_groups < 1:
            raise ValueError("detail_groups must be positive")

        self.levels = list(levels)
        self.branches = nn.ModuleDict()
        if "P3" in self.levels:
            self.branches["P3"] = ExactScaleFusion(
                in_channels, out_channels, scale=2, num_blocks=num_blocks
            )
        if "P4" in self.levels:
            self.branches["P4"] = ExactScaleFusion(
                in_channels, out_channels, scale=1, num_blocks=num_blocks
            )
        if "P5" in self.levels:
            self.branches["P5"] = PackedSeparableTwoBasisP5Fusion(
                in_channels,
                out_channels,
                num_blocks=num_blocks,
                anti_alias=use_phase_downsample,
                maximum_groups=detail_groups,
                grouped_feature_index=-1,
            )

        self.scale_calibration = BidirectionalScaleCalibration(
            channels=out_channels,
            levels=self.levels,
            rank_channels=cross_scale_rank,
            mode=cross_scale_mode,
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        del image
        outputs = {
            level: self.branches[level](features)
            for level in self.levels
        }
        outputs = self.scale_calibration(outputs)

        results = []
        for level in self.levels:
            feature = outputs[level]
            if mask is not None:
                output_mask = F.interpolate(
                    mask[:, None].float(),
                    size=feature.shape[-2:],
                    mode="nearest",
                ).to(torch.bool)
                feature = feature.masked_fill(output_mask, 0)
            results.append(feature)
        return results
