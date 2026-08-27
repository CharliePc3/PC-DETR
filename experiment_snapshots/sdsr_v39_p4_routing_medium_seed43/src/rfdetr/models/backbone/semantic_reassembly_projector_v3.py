# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Target-scale layer fusion for detector-ready DINOv3 features."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.backbone.projector import LayerNorm
from rfdetr.models.backbone.semantic_reassembly_projector import (
    DirectionalDetailStem,
    LightRefineBlock,
    _group_norm,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v2 import LightweightScaleFusion


class RankDownsample(nn.Module):
    """Downsample one low-rank ViT layer before cross-layer fusion."""

    def __init__(self, channels: int, anti_alias: bool):
        super().__init__()
        self.anti_alias = anti_alias
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = LayerNorm(channels)
        self.residual_scale = nn.Parameter(torch.tensor(0.1)) if anti_alias else None

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        learned = self.norm(self.pointwise(self.depthwise(feature)))
        if not self.anti_alias:
            return learned

        low_pass = F.avg_pool2d(feature, kernel_size=2, stride=2, ceil_mode=True)
        return low_pass + torch.tanh(self.residual_scale) * learned


class TargetScaleLayerFusion(nn.Module):
    """Project layers cheaply, transform each to its target scale, then fuse."""

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
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm(channels),
                    nn.Conv2d(channels, rank_channels, kernel_size=1, bias=False),
                )
                for channels in in_channels
            ]
        )
        fused_channels = len(in_channels) * rank_channels
        self.fusions = nn.ModuleDict(
            {level: LightweightScaleFusion(fused_channels, out_channels) for level in self.levels}
        )
        self.p5_downsamples = (
            nn.ModuleList(
                [RankDownsample(rank_channels, anti_alias=anti_alias_p5) for _ in in_channels]
            )
            if "P5" in self.levels
            else None
        )

    def _project(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} features, got {len(features)}")

        base_size = features[-1].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            if feature.shape[-2:] != base_size:
                feature = F.interpolate(feature, size=base_size, mode="bilinear", align_corners=False)
            projected.append(projection(feature))
        return projected

    def forward(self, features: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        projected = self._project(features)
        base_height, base_width = projected[-1].shape[-2:]
        outputs = {}

        if "P3" in self.levels:
            p3_size = (base_height * 2, base_width * 2)
            p3_layers = [
                F.interpolate(feature, size=p3_size, mode="bilinear", align_corners=False)
                for feature in projected
            ]
            outputs["P3"] = self.fusions["P3"](torch.cat(p3_layers, dim=1))

        if "P4" in self.levels:
            outputs["P4"] = self.fusions["P4"](torch.cat(projected, dim=1))

        if "P5" in self.levels:
            p5_layers = [
                downsample(feature)
                for feature, downsample in zip(projected, self.p5_downsamples)
            ]
            outputs["P5"] = self.fusions["P5"](torch.cat(p5_layers, dim=1))

        return outputs


class SpatialConfidenceDetailRefinement(nn.Module):
    """Inject stride-8 image detail only where semantic and local cues agree."""

    def __init__(self, channels: int, detail_channels: int):
        super().__init__()
        self.semantic_context = nn.Sequential(
            nn.Conv2d(channels, detail_channels, kernel_size=1, bias=False),
            _group_norm(detail_channels),
            nn.GELU(),
        )
        self.confidence = nn.Sequential(
            nn.Conv2d(
                2 * detail_channels,
                detail_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _group_norm(detail_channels),
            nn.GELU(),
            nn.Conv2d(detail_channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.confidence[-1].weight)
        nn.init.constant_(self.confidence[-1].bias, -2.0)

        self.detail_residual = nn.Sequential(
            nn.Conv2d(
                detail_channels,
                detail_channels,
                kernel_size=3,
                padding=1,
                groups=detail_channels,
                bias=False,
            ),
            _group_norm(detail_channels),
            nn.GELU(),
            nn.Conv2d(detail_channels, channels, kernel_size=1, bias=False),
            LayerNorm(channels),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        semantic: torch.Tensor,
        detail: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        detail = F.interpolate(detail, size=semantic.shape[-2:], mode="bilinear", align_corners=False)
        context = self.semantic_context(semantic)
        confidence = torch.sigmoid(self.confidence(torch.cat((context, detail), dim=1)))
        residual = self.detail_residual(detail)
        output = semantic + torch.tanh(self.residual_scale) * confidence * residual

        if mask is not None:
            output_mask = F.interpolate(
                mask[:, None].float(), size=semantic.shape[-2:], mode="nearest"
            ).to(torch.bool)
            output = output.masked_fill(output_mask, 0)
        return output


class ScaleDecoupledReassemblyProjectorV3(nn.Module):
    """Fuse ViT layers after each layer has reached P3, P4, or P5 resolution."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        levels: Sequence[str],
        rank_channels: int = 64,
        detail_channels: int = 32,
        use_local_reassembly: bool = True,
        use_directional_guide: bool = True,
        use_phase_downsample: bool = False,
    ):
        super().__init__()
        unsupported = set(levels) - {"P3", "P4", "P5"}
        if unsupported:
            raise ValueError(f"SDSR-v3 Projector does not support levels: {sorted(unsupported)}")
        if not levels:
            raise ValueError("levels must contain at least one pyramid level")

        self.levels = list(levels)
        self.use_spatial_detail = (
            use_local_reassembly and use_directional_guide and "P3" in self.levels
        )
        self.layer_fusion = TargetScaleLayerFusion(
            in_channels=in_channels,
            rank_channels=rank_channels,
            out_channels=out_channels,
            levels=self.levels,
            anti_alias_p5=use_phase_downsample,
        )
        self.detail_stem = DirectionalDetailStem(detail_channels) if self.use_spatial_detail else None
        self.spatial_detail = (
            SpatialConfidenceDetailRefinement(out_channels, detail_channels)
            if self.use_spatial_detail
            else None
        )
        self.refine = nn.ModuleDict({level: LightRefineBlock(out_channels) for level in self.levels})

    def forward(
        self,
        features: Sequence[torch.Tensor],
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        outputs = self.layer_fusion(features)

        if self.spatial_detail is not None:
            if image is None:
                raise ValueError("SDSR-v3 spatial detail refinement requires the input image")
            guide = self.detail_stem(image, mask)
            outputs["P3"] = self.spatial_detail(outputs["P3"], guide, mask)

        return [self.refine[level](outputs[level]) for level in self.levels]
