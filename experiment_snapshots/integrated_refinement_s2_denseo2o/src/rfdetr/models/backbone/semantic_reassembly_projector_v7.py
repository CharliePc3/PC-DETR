# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Scale-first low-rank projector with optional cross-scale calibration."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.backbone.projector import LayerNorm
from rfdetr.models.backbone.semantic_reassembly_projector import LightRefineBlock
from rfdetr.models.backbone.semantic_reassembly_projector_v2 import (
    LightweightScaleFusion,
)


class FullChannelPhaseUpsample(nn.Module):
    """Decode P3 phases before compressing a ViT feature to its low-rank subspace."""

    def __init__(self, in_channels: int, rank_channels: int):
        super().__init__()
        self.phase_projection = nn.Conv2d(
            in_channels,
            4 * rank_channels,
            kernel_size=1,
            bias=False,
        )
        self.local_mixing = nn.Conv2d(
            rank_channels,
            rank_channels,
            kernel_size=3,
            padding=1,
            groups=rank_channels,
            bias=False,
        )
        self.norm = LayerNorm(rank_channels)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = F.pixel_shuffle(self.phase_projection(feature), upscale_factor=2)
        return self.norm(feature + self.local_mixing(feature))


class FullChannelDownsample(nn.Module):
    """Construct one low-rank P5 layer with optional low-pass anchoring."""

    def __init__(self, in_channels: int, rank_channels: int, anti_alias: bool):
        super().__init__()
        self.anti_alias = anti_alias
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.projection = nn.Conv2d(
            in_channels,
            rank_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm = LayerNorm(rank_channels)
        self.learned_scale = nn.Parameter(torch.tensor(0.1)) if anti_alias else None

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        learned = self.depthwise(feature)
        if self.anti_alias:
            low_pass = F.avg_pool2d(feature, kernel_size=2, stride=2, ceil_mode=True)
            learned = low_pass + torch.tanh(self.learned_scale) * learned
        return self.norm(self.projection(learned))


class ScaleFirstLayerFusion(nn.Module):
    """Move every ViT layer to its target scale before low-rank semantic fusion."""

    def __init__(
        self,
        in_channels: Sequence[int],
        rank_channels: int,
        out_channels: int,
        levels: Sequence[str],
        anti_alias_p5: bool,
    ):
        super().__init__()
        if not in_channels:
            raise ValueError("in_channels must contain at least one feature")
        if rank_channels < 1:
            raise ValueError("rank_channels must be positive")

        self.levels = list(levels)
        self.input_norms = nn.ModuleList(
            [LayerNorm(channels) for channels in in_channels]
        )
        self.p3_projections = (
            nn.ModuleList(
                [
                    FullChannelPhaseUpsample(channels, rank_channels)
                    for channels in in_channels
                ]
            )
            if "P3" in self.levels
            else None
        )
        self.p4_projections = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            channels,
                            rank_channels,
                            kernel_size=1,
                            bias=False,
                        ),
                        LayerNorm(rank_channels),
                    )
                    for channels in in_channels
                ]
            )
            if "P4" in self.levels
            else None
        )
        self.p5_projections = (
            nn.ModuleList(
                [
                    FullChannelDownsample(
                        channels,
                        rank_channels,
                        anti_alias=anti_alias_p5,
                    )
                    for channels in in_channels
                ]
            )
            if "P5" in self.levels
            else None
        )

        fused_channels = len(in_channels) * rank_channels
        self.fusions = nn.ModuleDict(
            {
                level: LightweightScaleFusion(fused_channels, out_channels)
                for level in self.levels
            }
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if len(features) != len(self.input_norms):
            raise ValueError(
                f"Expected {len(self.input_norms)} features, got {len(features)}"
            )

        base_size = features[-1].shape[-2:]
        normalized = []
        for feature, norm in zip(features, self.input_norms):
            if feature.shape[-2:] != base_size:
                feature = F.interpolate(
                    feature,
                    size=base_size,
                    mode="bilinear",
                    align_corners=False,
                )
            normalized.append(norm(feature))

        outputs = {}
        if self.p3_projections is not None:
            p3_layers = [
                projection(feature)
                for feature, projection in zip(normalized, self.p3_projections)
            ]
            outputs["P3"] = self.fusions["P3"](torch.cat(p3_layers, dim=1))

        if self.p4_projections is not None:
            p4_layers = [
                projection(feature)
                for feature, projection in zip(normalized, self.p4_projections)
            ]
            outputs["P4"] = self.fusions["P4"](torch.cat(p4_layers, dim=1))

        if self.p5_projections is not None:
            p5_layers = [
                projection(feature)
                for feature, projection in zip(normalized, self.p5_projections)
            ]
            outputs["P5"] = self.fusions["P5"](torch.cat(p5_layers, dim=1))
        return outputs


class LowRankScaleMessage(nn.Module):
    """Send a gated low-rank residual between adjacent pyramid levels."""

    def __init__(self, channels: int, rank_channels: int):
        super().__init__()
        self.input_norm = LayerNorm(channels)
        self.reduce = nn.Conv2d(channels, rank_channels, kernel_size=1, bias=False)
        self.local_mixing = nn.Conv2d(
            rank_channels,
            rank_channels,
            kernel_size=3,
            padding=1,
            groups=rank_channels,
            bias=False,
        )
        self.expand = nn.Conv2d(rank_channels, channels, kernel_size=1, bias=False)
        self.output_norm = LayerNorm(channels)
        self.channel_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(
        self,
        source: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        message = F.silu(self.reduce(self.input_norm(source)))
        if message.shape[-2:] != target_size:
            if message.shape[-2] > target_size[0]:
                message = F.adaptive_avg_pool2d(message, target_size)
            else:
                message = F.interpolate(
                    message,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
        message = message + self.local_mixing(message)
        message = self.output_norm(self.expand(F.silu(message)))
        return torch.tanh(self.channel_scale) * message


class BidirectionalScaleCalibration(nn.Module):
    """Apply one lightweight top-down pass and an optional bottom-up pass."""

    def __init__(
        self,
        channels: int,
        levels: Sequence[str],
        rank_channels: int,
        mode: str,
    ):
        super().__init__()
        if mode not in {"none", "topdown", "bidirectional"}:
            raise ValueError(f"Unsupported cross-scale calibration mode: {mode}")
        self.levels = list(levels)
        self.mode = mode
        num_edges = max(len(self.levels) - 1, 0) if mode != "none" else 0
        self.top_down = nn.ModuleList(
            [LowRankScaleMessage(channels, rank_channels) for _ in range(num_edges)]
        )
        self.bottom_up = (
            nn.ModuleList(
                [LowRankScaleMessage(channels, rank_channels) for _ in range(num_edges)]
            )
            if mode == "bidirectional"
            else None
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.mode == "none" or len(self.levels) < 2:
            return features

        outputs = dict(features)
        for edge_index, high_index in enumerate(range(len(self.levels) - 1, 0, -1)):
            high_level = self.levels[high_index]
            low_level = self.levels[high_index - 1]
            outputs[low_level] = outputs[low_level] + self.top_down[edge_index](
                outputs[high_level],
                outputs[low_level].shape[-2:],
            )

        if self.bottom_up is not None:
            for edge_index in range(len(self.levels) - 1):
                low_level = self.levels[edge_index]
                high_level = self.levels[edge_index + 1]
                outputs[high_level] = outputs[high_level] + self.bottom_up[edge_index](
                    outputs[low_level],
                    outputs[high_level].shape[-2:],
                )
        return outputs


class ScaleDecoupledReassemblyProjectorV7(nn.Module):
    """Preserve full-channel scale transformations and fuse them efficiently."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        levels: Sequence[str],
        rank_channels: int = 64,
        cross_scale_mode: str = "none",
        cross_scale_rank: int = 32,
        use_phase_downsample: bool = False,
    ):
        super().__init__()
        unsupported = set(levels) - {"P3", "P4", "P5"}
        if unsupported:
            raise ValueError(
                f"SDSR-v7 Projector does not support levels: {sorted(unsupported)}"
            )
        if not levels:
            raise ValueError("levels must contain at least one pyramid level")
        if cross_scale_rank < 1:
            raise ValueError("cross_scale_rank must be positive")

        self.levels = list(levels)
        self.layer_fusion = ScaleFirstLayerFusion(
            in_channels=in_channels,
            rank_channels=rank_channels,
            out_channels=out_channels,
            levels=self.levels,
            anti_alias_p5=use_phase_downsample,
        )
        self.scale_calibration = BidirectionalScaleCalibration(
            channels=out_channels,
            levels=self.levels,
            rank_channels=cross_scale_rank,
            mode=cross_scale_mode,
        )
        self.refine = nn.ModuleDict(
            {level: LightRefineBlock(out_channels) for level in self.levels}
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
