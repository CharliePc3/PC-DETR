# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Endpoint-adaptive grouped detail sampling for P5 reassembly."""

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
    _expand_detail_sampler,
)


class EndpointAdaptiveP5Fusion(nn.Module):
    """Use stronger shallow and lighter deep semantic detail coupling."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_blocks: int,
        anti_alias: bool,
        early_groups: int,
        deep_groups: int,
    ):
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError("Endpoint-adaptive fusion needs at least two features")
        self.num_features = len(in_channels)
        self.downsampling = nn.ModuleList()
        for index, channels in enumerate(in_channels):
            sampler = TwoBasisDownsample(channels, anti_alias=anti_alias)
            if index == 0:
                _expand_detail_sampler(sampler, channels, early_groups)
            elif index == self.num_features - 1:
                _expand_detail_sampler(sampler, channels, deep_groups)
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


class ScaleDecoupledReassemblyProjectorV26(nn.Module):
    """Decouple shallow spatial and deep semantic P5 reassembly capacity."""

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
        early_groups: int = 4,
        deep_groups: int = 16,
    ):
        super().__init__()
        del rank_channels
        unsupported = set(levels) - {"P3", "P4", "P5"}
        if unsupported:
            raise ValueError(
                f"SDSR-v26 Projector does not support levels: {sorted(unsupported)}"
            )
        if not levels:
            raise ValueError("levels must contain at least one pyramid level")
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if early_groups < 1 or deep_groups < 1:
            raise ValueError("detail groups must be positive")

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
            self.branches["P5"] = EndpointAdaptiveP5Fusion(
                in_channels,
                out_channels,
                num_blocks=num_blocks,
                anti_alias=use_phase_downsample,
                early_groups=early_groups,
                deep_groups=deep_groups,
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
