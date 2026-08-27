# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Joint semantic reassembly with a shallow-token phase detail path."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.backbone.projector import LayerNorm
from rfdetr.models.backbone.semantic_reassembly_projector import (
    DirectionalDetailStem,
    LightRefineBlock,
    SemanticLocalReassembly,
    _group_norm,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v3 import TargetScaleLayerFusion


class ShallowPhaseDetail(nn.Module):
    """Decode sub-token phase detail from the two shallowest projected layers."""

    def __init__(
        self,
        rank_channels: int,
        num_sources: int,
        detail_channels: int,
        output_channels: int,
    ):
        super().__init__()
        input_channels = rank_channels * num_sources
        self.num_sources = num_sources
        self.input_norm = LayerNorm(input_channels)
        self.local_mixing = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size=3,
            padding=1,
            groups=input_channels,
            bias=False,
        )
        self.phase_expansion = nn.Conv2d(
            input_channels,
            4 * detail_channels,
            kernel_size=1,
            bias=False,
        )
        self.detail_refine = nn.Sequential(
            _group_norm(detail_channels),
            nn.GELU(),
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
            nn.Conv2d(detail_channels, output_channels, kernel_size=1, bias=False),
            LayerNorm(output_channels),
        )
        self.channel_scale = nn.Parameter(
            torch.full((1, output_channels, 1, 1), 0.1)
        )

    def forward(
        self,
        projected: Sequence[torch.Tensor],
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        source = torch.cat(projected[: self.num_sources], dim=1)
        source = self.input_norm(source)
        source = source + self.local_mixing(source)
        detail = F.pixel_shuffle(self.phase_expansion(F.gelu(source)), upscale_factor=2)
        detail = torch.tanh(self.channel_scale) * self.detail_refine(detail)

        if mask is not None:
            output_mask = F.interpolate(
                mask[:, None].float(), size=detail.shape[-2:], mode="nearest"
            ).to(torch.bool)
            detail = detail.masked_fill(output_mask, 0)
        return detail


class PhaseAwareJointReassemblyFusion(TargetScaleLayerFusion):
    """Add learned channel-to-space phase detail before the P3 scale fusion."""

    def __init__(
        self,
        in_channels: Sequence[int],
        rank_channels: int,
        out_channels: int,
        levels: Sequence[str],
        detail_channels: int,
        use_local_reassembly: bool,
        use_directional_guide: bool,
        anti_alias_p5: bool,
    ):
        super().__init__(
            in_channels=in_channels,
            rank_channels=rank_channels,
            out_channels=out_channels,
            levels=levels,
            anti_alias_p5=anti_alias_p5,
        )
        fused_channels = len(in_channels) * rank_channels
        self.p3_reassembly = (
            SemanticLocalReassembly(
                fused_channels,
                detail_channels,
                use_directional_guide=use_directional_guide,
            )
            if use_local_reassembly and "P3" in levels
            else None
        )
        self.phase_detail = (
            ShallowPhaseDetail(
                rank_channels=rank_channels,
                num_sources=min(2, len(in_channels)),
                detail_channels=detail_channels,
                output_channels=fused_channels,
            )
            if "P3" in levels
            else None
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        detail_guide: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        projected = self._project(features)
        joint = torch.cat(projected, dim=1)
        base_height, base_width = projected[-1].shape[-2:]
        outputs = {}

        if "P3" in self.levels:
            if self.p3_reassembly is None:
                p3 = F.interpolate(
                    joint,
                    size=(base_height * 2, base_width * 2),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                p3 = self.p3_reassembly(joint, detail_guide, mask)
            p3 = p3 + self.phase_detail(projected, mask)
            outputs["P3"] = self.fusions["P3"](p3)

        if "P4" in self.levels:
            outputs["P4"] = self.fusions["P4"](joint)

        if "P5" in self.levels:
            p5_layers = [
                downsample(feature)
                for feature, downsample in zip(projected, self.p5_downsamples)
            ]
            outputs["P5"] = self.fusions["P5"](torch.cat(p5_layers, dim=1))

        return outputs


class ScaleDecoupledReassemblyProjectorV6(nn.Module):
    """Preserve layer semantics and restore P3 sub-token phase information."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        levels: Sequence[str],
        rank_channels: int = 64,
        detail_channels: int = 32,
        use_local_reassembly: bool = True,
        use_directional_guide: bool = False,
        use_phase_downsample: bool = False,
    ):
        super().__init__()
        unsupported = set(levels) - {"P3", "P4", "P5"}
        if unsupported:
            raise ValueError(f"SDSR-v6 Projector does not support levels: {sorted(unsupported)}")
        if not levels:
            raise ValueError("levels must contain at least one pyramid level")

        self.levels = list(levels)
        self.use_local_reassembly = use_local_reassembly and "P3" in self.levels
        self.use_directional_guide = use_directional_guide and self.use_local_reassembly
        self.detail_stem = DirectionalDetailStem(detail_channels) if self.use_directional_guide else None
        self.layer_fusion = PhaseAwareJointReassemblyFusion(
            in_channels=in_channels,
            rank_channels=rank_channels,
            out_channels=out_channels,
            levels=self.levels,
            detail_channels=detail_channels,
            use_local_reassembly=self.use_local_reassembly,
            use_directional_guide=self.use_directional_guide,
            anti_alias_p5=use_phase_downsample,
        )
        self.refine = nn.ModuleDict({level: LightRefineBlock(out_channels) for level in self.levels})

    def forward(
        self,
        features: Sequence[torch.Tensor],
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        detail_guide = None
        if self.detail_stem is not None:
            if image is None:
                raise ValueError("SDSR-v6 directional reassembly requires the input image")
            detail_guide = self.detail_stem(image, mask)

        outputs = self.layer_fusion(features, detail_guide=detail_guide, mask=mask)
        return [self.refine[level](outputs[level]) for level in self.levels]
