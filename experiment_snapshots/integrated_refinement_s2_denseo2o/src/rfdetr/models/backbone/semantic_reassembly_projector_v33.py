# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Residual-parameterized grouped detail sampling for stable P5 reassembly."""

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
from rfdetr.models.backbone.semantic_reassembly_projector_v11 import (
    TwoBasisDownsample,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v22 import (
    _largest_group_divisor,
)


class GatedOffDiagonalGroupedConv(nn.Module):
    """Keep depthwise sampling explicit and learn only gated cross-channel terms."""

    def __init__(
        self,
        depthwise: nn.Conv2d,
        channels: int,
        maximum_groups: int,
        initial_gate: float = 0.1,
    ):
        super().__init__()
        groups = _largest_group_divisor(channels, maximum_groups)
        channels_per_group = channels // groups
        self.depthwise = depthwise
        with torch.random.fork_rng(devices=[]):
            self.cross_channel = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=groups,
                bias=False,
            )
        nn.init.zeros_(self.cross_channel.weight)

        off_diagonal = torch.ones_like(self.cross_channel.weight)
        for output_channel in range(channels):
            input_within_group = output_channel % channels_per_group
            off_diagonal[output_channel, input_within_group].zero_()
        self.register_buffer("off_diagonal", off_diagonal, persistent=False)
        gate_value = torch.atanh(torch.tensor(float(initial_gate)))
        self.cross_channel_gate = nn.Parameter(
            gate_value.expand(1, channels, 1, 1).clone()
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        residual = F.conv2d(
            feature,
            self.cross_channel.weight * self.off_diagonal,
            stride=2,
            padding=1,
            groups=self.cross_channel.groups,
        )
        return self.depthwise(feature) + torch.tanh(
            self.cross_channel_gate
        ).to(dtype=residual.dtype) * residual


class ResidualParameterizedP5Fusion(nn.Module):
    """Use conservative cross-channel detail learning only at ViT depth 12."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_blocks: int,
        anti_alias: bool,
        maximum_groups: int,
    ):
        super().__init__()
        if not in_channels:
            raise ValueError("in_channels must contain at least one feature")
        self.num_features = len(in_channels)
        self.downsampling = nn.ModuleList()
        for index, channels in enumerate(in_channels):
            sampler = TwoBasisDownsample(channels, anti_alias=anti_alias)
            if index == self.num_features - 1:
                sampler.detail = GatedOffDiagonalGroupedConv(
                    sampler.detail,
                    channels=channels,
                    maximum_groups=maximum_groups,
                )
            self.downsampling.append(sampler)

        self.fusion = C2f(
            2 * sum(in_channels),
            out_channels,
            n=num_blocks,
            layer_norm=True,
        )
        self.output_norm = LayerNorm(out_channels)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {len(features)}"
            )
        sampled = [
            downsample(feature)
            for downsample, feature in zip(self.downsampling, features)
        ]
        return self.output_norm(self.fusion(torch.cat(sampled, dim=1)))


class ScaleDecoupledReassemblyProjectorV33(nn.Module):
    """Regularize v23 cross-channel detail as a gated off-diagonal residual."""

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
                f"SDSR-v33 Projector does not support levels: {sorted(unsupported)}"
            )
        if not levels:
            raise ValueError("levels must contain at least one pyramid level")

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
            self.branches["P5"] = ResidualParameterizedP5Fusion(
                in_channels,
                out_channels,
                num_blocks=num_blocks,
                anti_alias=use_phase_downsample,
                maximum_groups=detail_groups,
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
