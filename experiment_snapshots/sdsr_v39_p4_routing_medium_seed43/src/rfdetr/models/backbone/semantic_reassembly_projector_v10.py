# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Efficient scale-decoupled projector with an alias-aware P5 branch."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.backbone.projector import C2f, LayerNorm
from rfdetr.models.backbone.semantic_reassembly_projector_v7 import (
    BidirectionalScaleCalibration,
)


class ExactScaleFusion(nn.Module):
    """Keep the proven MultiScaleProjector fusion at P3 or P4."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        scale: int,
        num_blocks: int,
    ):
        super().__init__()
        if scale not in {1, 2}:
            raise ValueError("ExactScaleFusion only supports scale 1 or 2")

        self.num_features = len(in_channels)
        if scale == 2:
            self.sampling = nn.ModuleList(
                [
                    nn.ConvTranspose2d(
                        channels,
                        channels // 2,
                        kernel_size=2,
                        stride=2,
                    )
                    for channels in in_channels
                ]
            )
            fused_channels = sum(channels // 2 for channels in in_channels)
        else:
            self.sampling = nn.ModuleList(
                [nn.Identity() for _ in in_channels]
            )
            fused_channels = sum(in_channels)

        self.fusion = C2f(
            fused_channels,
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
            sampling(feature)
            for sampling, feature in zip(self.sampling, features)
        ]
        return self.output_norm(self.fusion(torch.cat(sampled, dim=1)))


class AliasAwareDownsample(nn.Module):
    """Downsample without discarding the pretrained channel representation."""

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
        self.detail = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.detail_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), 0.1)
        )
        self.norm = LayerNorm(channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        detail = self.detail(feature)
        if self.anti_alias:
            low_pass = F.conv2d(
                feature,
                self.blur_kernel.to(dtype=feature.dtype).expand(
                    feature.shape[1], 1, -1, -1
                ),
                stride=2,
                padding=1,
                groups=feature.shape[1],
            )
            feature = (
                low_pass
                + torch.tanh(self.detail_scale).to(dtype=detail.dtype) * detail
            )
        else:
            feature = detail
        return self.activation(self.norm(feature))


class AliasAwareP5Fusion(nn.Module):
    """Fuse all ViT layers after inexpensive phase-preserving downsampling."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_blocks: int,
        anti_alias: bool,
    ):
        super().__init__()
        self.num_features = len(in_channels)
        # Initialize the efficient samplers without perturbing the global RNG.
        # Advancing it with throwaway dense kernels keeps the following C2f
        # bit-identical to the corresponding MultiScaleProjector P5 fusion.
        with torch.random.fork_rng(devices=[]):
            self.downsampling = nn.ModuleList(
                [
                    AliasAwareDownsample(channels, anti_alias=anti_alias)
                    for channels in in_channels
                ]
            )
        for channels in in_channels:
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
        self.fusion = C2f(
            sum(in_channels),
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


class ScaleDecoupledReassemblyProjectorV10(nn.Module):
    """Preserve independent multilevel fusion while removing dense P5 sampling."""

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
                f"SDSR-v10 Projector does not support levels: {sorted(unsupported)}"
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
            self.branches["P5"] = AliasAwareP5Fusion(
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
