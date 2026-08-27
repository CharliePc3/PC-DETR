# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Scale-decoupled projector with a two-basis P5 phase representation."""

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


class TwoBasisDownsample(nn.Module):
    """Keep low-frequency and learnable detail responses as separate channels."""

    def __init__(self, channels: int, anti_alias: bool):
        super().__init__()
        self.anti_alias = anti_alias
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
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=channels,
                bias=False,
            )
        )
        self.detail = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.primary_norm = LayerNorm(channels)
        self.detail_norm = LayerNorm(channels)
        self.detail_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), 0.1)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if self.primary is None:
            primary = F.conv2d(
                feature,
                self.blur_kernel.to(dtype=feature.dtype).expand(
                    feature.shape[1], 1, -1, -1
                ),
                stride=2,
                padding=1,
                groups=feature.shape[1],
            )
        else:
            primary = self.primary(feature)
        detail = self.detail(feature)
        primary = self.activation(self.primary_norm(primary))
        detail = self.activation(self.detail_norm(detail))
        detail = (
            torch.tanh(self.detail_scale).to(dtype=detail.dtype) * detail
        )
        return torch.cat([primary, detail], dim=1)


class TwoBasisP5Fusion(nn.Module):
    """Expose two spatial bases per input channel before semantic mixing."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_blocks: int,
        anti_alias: bool,
    ):
        super().__init__()
        self.num_features = len(in_channels)
        self.downsampling = nn.ModuleList(
            [
                TwoBasisDownsample(channels, anti_alias=anti_alias)
                for channels in in_channels
            ]
        )
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


class ScaleDecoupledReassemblyProjectorV11(nn.Module):
    """Retain exact P3/P4 fusion and enrich efficient P5 spatial rank."""

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
        del rank_channels
        unsupported = set(levels) - {"P3", "P4", "P5"}
        if unsupported:
            raise ValueError(
                f"SDSR-v11 Projector does not support levels: {sorted(unsupported)}"
            )
        if not levels:
            raise ValueError("levels must contain at least one pyramid level")
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")

        self.levels = list(levels)
        self.branches = nn.ModuleDict()
        if "P3" in self.levels:
            self.branches["P3"] = ExactScaleFusion(
                in_channels,
                out_channels,
                scale=2,
                num_blocks=num_blocks,
            )
        if "P4" in self.levels:
            self.branches["P4"] = ExactScaleFusion(
                in_channels,
                out_channels,
                scale=1,
                num_blocks=num_blocks,
            )
        if "P5" in self.levels:
            self.branches["P5"] = TwoBasisP5Fusion(
                in_channels,
                out_channels,
                num_blocks=num_blocks,
                anti_alias=use_phase_downsample,
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
