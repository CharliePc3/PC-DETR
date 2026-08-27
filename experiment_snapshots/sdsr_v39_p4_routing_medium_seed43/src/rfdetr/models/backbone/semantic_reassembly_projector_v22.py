# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Two-basis P5 reassembly with expandable grouped detail sampling."""

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


def _largest_group_divisor(channels: int, maximum_groups: int) -> int:
    for groups in range(min(channels, maximum_groups), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _expand_detail_sampler(
    sampler: TwoBasisDownsample,
    channels: int,
    maximum_groups: int,
) -> None:
    """Embed a depthwise kernel on the diagonal of a wider grouped kernel."""

    groups = _largest_group_divisor(channels, maximum_groups)
    channels_per_group = channels // groups
    depthwise_weight = sampler.detail.weight.detach().clone()
    with torch.random.fork_rng(devices=[]):
        grouped_detail = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=groups,
            bias=False,
        )
    with torch.no_grad():
        grouped_detail.weight.zero_()
        for output_channel in range(channels):
            input_within_group = output_channel % channels_per_group
            grouped_detail.weight[output_channel, input_within_group].copy_(
                depthwise_weight[output_channel, 0]
            )
    sampler.detail = grouped_detail


class ExpandableGroupedDetailP5Fusion(nn.Module):
    """Start from v11 exactly and learn grouped spatial-channel detail."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_blocks: int,
        anti_alias: bool,
        maximum_groups: int,
    ):
        super().__init__()
        self.num_features = len(in_channels)
        self.downsampling = nn.ModuleList()
        for channels in in_channels:
            sampler = TwoBasisDownsample(channels, anti_alias=anti_alias)
            _expand_detail_sampler(
                sampler,
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


class ScaleDecoupledReassemblyProjectorV22(nn.Module):
    """Preserve v11 initialization while making P5 detail semantically expandable."""

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
                f"SDSR-v22 Projector does not support levels: {sorted(unsupported)}"
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
            self.branches["P5"] = ExpandableGroupedDetailP5Fusion(
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
