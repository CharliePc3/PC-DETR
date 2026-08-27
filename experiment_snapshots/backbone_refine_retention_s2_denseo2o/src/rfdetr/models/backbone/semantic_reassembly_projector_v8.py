# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Layer-preserving projector with efficient CSP semantic mixing."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.backbone.projector import LayerNorm
from rfdetr.models.backbone.semantic_reassembly_projector import LightRefineBlock
from rfdetr.models.backbone.semantic_reassembly_projector_v3 import RankDownsample
from rfdetr.models.backbone.semantic_reassembly_projector_v7 import (
    BidirectionalScaleCalibration,
)


class InvertedDepthwiseBottleneck(nn.Module):
    """Mix local context and channels on one CSP branch."""

    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden_channels = channels * expansion
        self.input_norm = LayerNorm(channels)
        self.expand = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=1,
            bias=False,
        )
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=False,
        )
        self.hidden_norm = LayerNorm(hidden_channels)
        self.project = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.output_norm = LayerNorm(channels)
        self.channel_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), 0.5)
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        residual = self.input_norm(feature)
        residual = F.silu(self.expand(residual))
        residual = F.silu(self.hidden_norm(self.depthwise(residual)))
        residual = self.output_norm(self.project(residual))
        return feature + torch.tanh(self.channel_scale) * residual


class EfficientCSPScaleFusion(nn.Module):
    """Use CSP feature reuse to deepen local mixing without a heavy C2f block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
    ):
        super().__init__()
        hidden_channels = max(out_channels // 2, 1)
        self.hidden_channels = hidden_channels
        self.input_projection = nn.Conv2d(
            in_channels,
            2 * hidden_channels,
            kernel_size=1,
            bias=False,
        )
        self.input_norm = LayerNorm(2 * hidden_channels)
        self.blocks = nn.ModuleList(
            [
                InvertedDepthwiseBottleneck(hidden_channels)
                for _ in range(num_blocks)
            ]
        )
        self.output_projection = nn.Conv2d(
            (2 + num_blocks) * hidden_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.output_norm = LayerNorm(out_channels)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        branches = list(
            F.silu(
                self.input_norm(self.input_projection(feature))
            ).split((self.hidden_channels, self.hidden_channels), dim=1)
        )
        for block in self.blocks:
            branches.append(block(branches[-1]))
        return self.output_norm(
            self.output_projection(torch.cat(branches, dim=1))
        )


class ResidualP3Transform(nn.Module):
    """Add cheap local processing after bilinear P3 resizing."""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm = LayerNorm(channels)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        feature: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        feature = F.interpolate(
            feature,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        local = self.norm(self.depthwise(feature))
        return feature + torch.tanh(self.residual_scale) * local


class CSPTargetScaleLayerFusion(nn.Module):
    """Route low-rank ViT layers separately into deeper target-scale fusions."""

    def __init__(
        self,
        in_channels: Sequence[int],
        rank_channels: int,
        out_channels: int,
        levels: Sequence[str],
        anti_alias_p5: bool,
        num_fusion_blocks: int,
    ):
        super().__init__()
        if not in_channels:
            raise ValueError("in_channels must contain at least one feature")
        if rank_channels < 1:
            raise ValueError("rank_channels must be positive")

        self.levels = list(levels)
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
        self.scale_gate_logits = nn.Parameter(
            self._initial_gate_logits(
                self.levels,
                len(in_channels),
                rank_channels,
            )
        )
        fused_channels = len(in_channels) * rank_channels
        self.fusions = nn.ModuleDict(
            {
                level: EfficientCSPScaleFusion(
                    fused_channels,
                    out_channels,
                    num_blocks=num_fusion_blocks,
                )
                for level in self.levels
            }
        )
        self.p3_transforms = (
            nn.ModuleList(
                [ResidualP3Transform(rank_channels) for _ in in_channels]
            )
            if "P3" in self.levels
            else None
        )
        self.p5_downsamples = (
            nn.ModuleList(
                [
                    RankDownsample(
                        rank_channels,
                        anti_alias=anti_alias_p5,
                    )
                    for _ in in_channels
                ]
            )
            if "P5" in self.levels
            else None
        )

    @staticmethod
    def _initial_gate_logits(
        levels: Sequence[str],
        num_layers: int,
        rank_channels: int,
    ) -> torch.Tensor:
        depth = torch.linspace(-1.0, 1.0, num_layers)
        level_slopes = {"P3": -0.1, "P4": 0.0, "P5": 0.1}
        gate_logits = []
        for level in levels:
            values = 1.0 + level_slopes[level] * depth
            logits = torch.log(values / (2.0 - values))
            gate_logits.append(
                logits[:, None].expand(-1, rank_channels)
            )
        return torch.stack(gate_logits)

    def gate_values(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.scale_gate_logits)

    def _project(
        self,
        features: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if len(features) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} features, got {len(features)}"
            )
        base_size = features[-1].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            if feature.shape[-2:] != base_size:
                feature = F.interpolate(
                    feature,
                    size=base_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(projection(feature))
        return projected

    def _fuse(
        self,
        level: str,
        features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        level_index = self.levels.index(level)
        gates = self.gate_values()[level_index].to(features[0].dtype)
        gated = [
            feature * gates[index][None, :, None, None]
            for index, feature in enumerate(features)
        ]
        return self.fusions[level](torch.cat(gated, dim=1))

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        projected = self._project(features)
        height, width = projected[-1].shape[-2:]
        outputs = {}

        if self.p3_transforms is not None:
            p3_layers = [
                transform(feature, (height * 2, width * 2))
                for feature, transform in zip(
                    projected,
                    self.p3_transforms,
                )
            ]
            outputs["P3"] = self._fuse("P3", p3_layers)

        if "P4" in self.levels:
            outputs["P4"] = self._fuse("P4", projected)

        if self.p5_downsamples is not None:
            p5_layers = [
                downsample(feature)
                for feature, downsample in zip(
                    projected,
                    self.p5_downsamples,
                )
            ]
            outputs["P5"] = self._fuse("P5", p5_layers)
        return outputs


class ScaleDecoupledReassemblyProjectorV8(nn.Module):
    """Combine layer routing, efficient CSP mixing, and optional scale messages."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        levels: Sequence[str],
        rank_channels: int = 64,
        cross_scale_mode: str = "none",
        cross_scale_rank: int = 32,
        use_phase_downsample: bool = False,
        num_fusion_blocks: int = 2,
    ):
        super().__init__()
        unsupported = set(levels) - {"P3", "P4", "P5"}
        if unsupported:
            raise ValueError(
                f"SDSR-v8 Projector does not support levels: {sorted(unsupported)}"
            )
        if not levels:
            raise ValueError("levels must contain at least one pyramid level")
        if num_fusion_blocks < 1:
            raise ValueError("num_fusion_blocks must be positive")

        self.levels = list(levels)
        self.layer_fusion = CSPTargetScaleLayerFusion(
            in_channels=in_channels,
            rank_channels=rank_channels,
            out_channels=out_channels,
            levels=self.levels,
            anti_alias_p5=use_phase_downsample,
            num_fusion_blocks=num_fusion_blocks,
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
        outputs = self.scale_calibration(self.layer_fusion(features))
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
