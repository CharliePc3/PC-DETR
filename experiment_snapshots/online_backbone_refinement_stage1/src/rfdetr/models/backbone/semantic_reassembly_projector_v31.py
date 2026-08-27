# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Complementary grouped-detail and semantic-residual P5 reassembly."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from rfdetr.models.backbone.semantic_reassembly_projector_v23 import (
    DepthAdaptiveGroupedDetailP5Fusion,
    ScaleDecoupledReassemblyProjectorV23,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v29 import (
    DeepSemanticBasis,
    _expand_fusion_input_exactly,
)


class ComplementarySemanticP5Fusion(nn.Module):
    """Append a deep semantic basis to an already initialized v23 P5 branch."""

    def __init__(
        self,
        base: DepthAdaptiveGroupedDetailP5Fusion,
        in_channels: Sequence[int],
        out_channels: int,
        num_blocks: int,
        semantic_groups: int,
    ):
        super().__init__()
        self.num_features = base.num_features
        self.downsampling = base.downsampling
        self.output_norm = base.output_norm
        base_channels = 2 * sum(in_channels)
        self.fusion = _expand_fusion_input_exactly(
            base.fusion,
            base_channels=base_channels,
            expanded_channels=base_channels + in_channels[-1],
            out_channels=out_channels,
            num_blocks=num_blocks,
        )
        with torch.random.fork_rng(devices=[]):
            self.semantic_basis = DeepSemanticBasis(
                in_channels[-1], maximum_groups=semantic_groups
            )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {len(features)}"
            )
        sampled = [
            downsample(feature)
            for downsample, feature in zip(self.downsampling, features)
        ]
        sampled.append(self.semantic_basis(features[-1]))
        return self.output_norm(self.fusion(torch.cat(sampled, dim=1)))


class ScaleDecoupledReassemblyProjectorV31(
    ScaleDecoupledReassemblyProjectorV23
):
    """Keep v23 intact and open one additional deepest-layer semantic basis."""

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
        semantic_groups: int = 16,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            levels=levels,
            rank_channels=rank_channels,
            cross_scale_mode=cross_scale_mode,
            cross_scale_rank=cross_scale_rank,
            use_phase_downsample=use_phase_downsample,
            num_blocks=num_blocks,
            detail_groups=detail_groups,
        )
        if "P5" in self.levels:
            self.branches["P5"] = ComplementarySemanticP5Fusion(
                self.branches["P5"],
                in_channels=in_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                semantic_groups=semantic_groups,
            )
