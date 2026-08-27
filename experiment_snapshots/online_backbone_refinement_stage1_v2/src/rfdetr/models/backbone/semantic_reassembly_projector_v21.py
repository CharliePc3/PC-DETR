# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Two-basis P5 reassembly with a low-rank spatial-semantic residual."""

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


class LowRankSpatialSemanticResidual(nn.Module):
    """Couple local spatial sampling and channel mixing at low rank."""

    def __init__(self, channels: int, rank_channels: int):
        super().__init__()
        rank_channels = min(channels, rank_channels)
        self.pre_norm = LayerNorm(channels)
        self.reduce = nn.Conv2d(
            channels, rank_channels, kernel_size=1, bias=False
        )
        self.spatial = nn.Conv2d(
            rank_channels,
            rank_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=rank_channels,
            bias=False,
        )
        self.expand = nn.Conv2d(
            rank_channels, channels, kernel_size=1, bias=False
        )
        self.output_norm = LayerNorm(channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = self.pre_norm(feature)
        feature = self.activation(self.reduce(feature))
        feature = self.activation(self.spatial(feature))
        return self.output_norm(self.expand(feature))


class SpatialSemanticTwoBasisDownsample(nn.Module):
    """Augment the v11 detail basis without replacing its stable low pass."""

    def __init__(
        self,
        channels: int,
        rank_channels: int,
        anti_alias: bool,
        residual_init: float = 0.02,
    ):
        super().__init__()
        self.base = TwoBasisDownsample(channels, anti_alias=anti_alias)
        self.residual = LowRankSpatialSemanticResidual(
            channels, rank_channels=rank_channels
        )
        self.residual_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), residual_init)
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        primary, detail = self.base(feature).chunk(2, dim=1)
        residual = self.residual(feature)
        detail = detail + torch.tanh(self.residual_scale).to(detail.dtype) * residual
        return torch.cat([primary, detail], dim=1)


class SpatialSemanticTwoBasisP5Fusion(nn.Module):
    """Fuse v11 bases after adding inexpensive local semantic coupling."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_blocks: int,
        anti_alias: bool,
        rank_channels: int,
    ):
        super().__init__()
        self.num_features = len(in_channels)
        self.downsampling = nn.ModuleList()
        for channels in in_channels:
            base = TwoBasisDownsample(channels, anti_alias=anti_alias)
            with torch.random.fork_rng(devices=[]):
                sampler = SpatialSemanticTwoBasisDownsample(
                    channels,
                    rank_channels=rank_channels,
                    anti_alias=anti_alias,
                )
            sampler.base = base
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


class ScaleDecoupledReassemblyProjectorV21(nn.Module):
    """Preserve v11 while coupling P5 spatial detail and layer semantics."""

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
    ):
        super().__init__()
        unsupported = set(levels) - {"P3", "P4", "P5"}
        if unsupported:
            raise ValueError(
                f"SDSR-v21 Projector does not support levels: {sorted(unsupported)}"
            )
        if not levels:
            raise ValueError("levels must contain at least one pyramid level")
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if rank_channels < 1:
            raise ValueError("rank_channels must be positive")

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
            self.branches["P5"] = SpatialSemanticTwoBasisP5Fusion(
                in_channels,
                out_channels,
                num_blocks=num_blocks,
                anti_alias=use_phase_downsample,
                rank_channels=rank_channels,
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
