# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Joint spatial reassembly with layer-preserving target-scale fusion."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.backbone.semantic_reassembly_projector import (
    DirectionalDetailStem,
    LightRefineBlock,
    SemanticLocalReassembly,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v3 import TargetScaleLayerFusion


class JointReassemblyTargetFusion(TargetScaleLayerFusion):
    """Use one all-layer spatial field while retaining layer channels until P3 fusion."""

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


class ScaleDecoupledReassemblyProjectorV5(nn.Module):
    """Reassemble all layer subspaces with one shared field, then fuse at each scale."""

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
            raise ValueError(f"SDSR-v5 Projector does not support levels: {sorted(unsupported)}")
        if not levels:
            raise ValueError("levels must contain at least one pyramid level")

        self.levels = list(levels)
        self.use_local_reassembly = use_local_reassembly and "P3" in self.levels
        self.use_directional_guide = use_directional_guide and self.use_local_reassembly
        self.detail_stem = DirectionalDetailStem(detail_channels) if self.use_directional_guide else None
        self.layer_fusion = JointReassemblyTargetFusion(
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
                raise ValueError("SDSR-v5 directional reassembly requires the input image")
            detail_guide = self.detail_stem(image, mask)

        outputs = self.layer_fusion(features, detail_guide=detail_guide, mask=mask)
        return [self.refine[level](outputs[level]) for level in self.levels]
