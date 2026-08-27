# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Anchor-preserving scale-decoupled semantic reassembly projector."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.backbone.projector import C2f, LayerNorm
from rfdetr.models.backbone.semantic_reassembly_projector import LightRefineBlock
from rfdetr.models.backbone.semantic_reassembly_projector_v7 import (
    BidirectionalScaleCalibration,
)


def _depth_gate_logits(
    num_layers: int,
    rank_channels: int,
    shallow_first: bool,
) -> torch.Tensor:
    depth = torch.linspace(-1.0, 1.0, num_layers)
    slope = -0.1 if shallow_first else 0.1
    values = 1.0 + slope * depth
    logits = torch.log(values / (2.0 - values))
    return logits[:, None].expand(-1, rank_channels).clone()


class SemanticAnchorP4(nn.Module):
    """Retain the strong full-channel P4 fusion used by MultiScaleProjector."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_blocks: int,
    ):
        super().__init__()
        self.num_features = len(in_channels)
        self.fusion = C2f(
            sum(in_channels),
            out_channels,
            n=num_blocks,
            layer_norm=True,
        )
        self.output_norm = LayerNorm(out_channels)

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(features) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {len(features)}"
            )
        target_size = features[-1].shape[-2:]
        aligned = [
            (
                feature
                if feature.shape[-2:] == target_size
                else F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            for feature in features
        ]
        return self.output_norm(self.fusion(torch.cat(aligned, dim=1)))


class LayerPhaseResidualP3(nn.Module):
    """Decode sub-patch phases as a residual around a geometry-safe P4 resize."""

    def __init__(
        self,
        in_channels: Sequence[int],
        rank_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm(channels),
                    nn.Conv2d(
                        channels,
                        rank_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                )
                for channels in in_channels
            ]
        )
        self.layer_gate_logits = nn.Parameter(
            _depth_gate_logits(
                len(in_channels),
                rank_channels,
                shallow_first=True,
            )
        )
        fused_channels = len(in_channels) * rank_channels
        self.phase_projection = nn.Conv2d(
            fused_channels,
            4 * out_channels,
            kernel_size=1,
            bias=False,
        )
        self.local_mixing = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=out_channels,
            bias=False,
        )
        self.detail_norm = LayerNorm(out_channels)
        self.output_norm = LayerNorm(out_channels)
        self.residual_scale = nn.Parameter(
            torch.full((1, out_channels, 1, 1), 0.1)
        )

    def gate_values(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.layer_gate_logits)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        if len(features) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} features, got {len(features)}"
            )
        target_size = anchor.shape[-2:]
        gates = self.gate_values().to(dtype=anchor.dtype)
        projected = []
        for index, (feature, projection) in enumerate(
            zip(features, self.projections)
        ):
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(
                projection(feature) * gates[index][None, :, None, None]
            )

        detail = F.pixel_shuffle(
            self.phase_projection(torch.cat(projected, dim=1)),
            upscale_factor=2,
        )
        detail = self.detail_norm(detail + self.local_mixing(detail))
        base = F.interpolate(
            anchor,
            size=detail.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.output_norm(
            base + torch.tanh(self.residual_scale) * detail
        )


class SemanticResidualP5(nn.Module):
    """Add a deep-layer residual to an anti-aliased P4 low-pass anchor."""

    def __init__(
        self,
        in_channels: Sequence[int],
        rank_channels: int,
        out_channels: int,
        anti_alias: bool,
    ):
        super().__init__()
        self.anti_alias = anti_alias
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm(channels),
                    nn.Conv2d(
                        channels,
                        rank_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                )
                for channels in in_channels
            ]
        )
        self.layer_gate_logits = nn.Parameter(
            _depth_gate_logits(
                len(in_channels),
                rank_channels,
                shallow_first=False,
            )
        )
        fused_channels = len(in_channels) * rank_channels
        self.downsample = nn.Conv2d(
            fused_channels,
            fused_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=fused_channels,
            bias=False,
        )
        self.mix = nn.Conv2d(
            fused_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.residual_norm = LayerNorm(out_channels)
        self.output_norm = LayerNorm(out_channels)
        self.residual_scale = nn.Parameter(
            torch.full((1, out_channels, 1, 1), 0.1)
        )

    def gate_values(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.layer_gate_logits)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        if len(features) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} features, got {len(features)}"
            )
        target_size = anchor.shape[-2:]
        gates = self.gate_values().to(dtype=anchor.dtype)
        projected = []
        for index, (feature, projection) in enumerate(
            zip(features, self.projections)
        ):
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(
                projection(feature) * gates[index][None, :, None, None]
            )

        residual = self.residual_norm(
            self.mix(self.downsample(torch.cat(projected, dim=1)))
        )
        if not self.anti_alias:
            return self.output_norm(residual)
        base = F.avg_pool2d(
            anchor,
            kernel_size=2,
            stride=2,
            ceil_mode=True,
        )
        return self.output_norm(
            base + torch.tanh(self.residual_scale) * residual
        )


class ScaleDecoupledReassemblyProjectorV9(nn.Module):
    """Use a full P4 anchor and lightweight geometry-preserving scale residuals."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        levels: Sequence[str],
        rank_channels: int = 64,
        cross_scale_mode: str = "none",
        cross_scale_rank: int = 32,
        use_phase_downsample: bool = True,
        anchor_blocks: int = 3,
    ):
        super().__init__()
        unsupported = set(levels) - {"P3", "P4", "P5"}
        if unsupported:
            raise ValueError(
                f"SDSR-v9 Projector does not support levels: {sorted(unsupported)}"
            )
        if not levels:
            raise ValueError("levels must contain at least one pyramid level")
        if rank_channels < 1:
            raise ValueError("rank_channels must be positive")
        if anchor_blocks < 1:
            raise ValueError("anchor_blocks must be positive")

        self.levels = list(levels)
        self.anchor = SemanticAnchorP4(
            in_channels,
            out_channels,
            num_blocks=anchor_blocks,
        )
        self.p3_reassembly = (
            LayerPhaseResidualP3(
                in_channels,
                rank_channels,
                out_channels,
            )
            if "P3" in self.levels
            else None
        )
        self.p5_reassembly = (
            SemanticResidualP5(
                in_channels,
                rank_channels,
                out_channels,
                anti_alias=use_phase_downsample,
            )
            if "P5" in self.levels
            else None
        )
        self.scale_calibration = BidirectionalScaleCalibration(
            channels=out_channels,
            levels=self.levels,
            rank_channels=cross_scale_rank,
            mode=cross_scale_mode,
        )
        self.refine = nn.ModuleDict(
            {
                level: LightRefineBlock(out_channels)
                for level in self.levels
            }
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        del image
        anchor = self.anchor(features)
        outputs = {}
        if self.p3_reassembly is not None:
            outputs["P3"] = self.p3_reassembly(features, anchor)
        if "P4" in self.levels:
            outputs["P4"] = anchor
        if self.p5_reassembly is not None:
            outputs["P5"] = self.p5_reassembly(features, anchor)

        outputs = self.scale_calibration(outputs)
        refined = []
        for level in self.levels:
            feature = self.refine[level](outputs[level])
            if mask is not None:
                output_mask = F.interpolate(
                    mask[:, None].float(),
                    size=feature.shape[-2:],
                    mode="nearest",
                ).to(torch.bool)
                feature = feature.masked_fill(output_mask, 0)
            refined.append(feature)
        return refined
