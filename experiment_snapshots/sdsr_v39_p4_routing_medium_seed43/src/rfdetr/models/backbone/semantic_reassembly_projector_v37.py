# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Layer-preserving scale routing on top of the SDSR-v23 fusion paths."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.backbone.semantic_reassembly_projector_v10 import (
    ExactScaleFusion,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v23 import (
    DepthAdaptiveGroupedDetailP5Fusion,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v7 import (
    BidirectionalScaleCalibration,
)


class LayerPreservingRoutedScaleFusion(ExactScaleFusion):
    """Modulate every sampled layer/channel without collapsing layer identity."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        scale: int,
        num_blocks: int,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            scale=scale,
            num_blocks=num_blocks,
        )
        sampled_channels = [
            channels // 2 if scale == 2 else channels
            for channels in in_channels
        ]
        self.route_logits = nn.ParameterList(
            [nn.Parameter(torch.zeros(channels)) for channels in sampled_channels]
        )

    def gate_values(self) -> list[torch.Tensor]:
        return [1.0 + torch.tanh(logits) for logits in self.route_logits]

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {len(features)}"
            )
        gates = self.gate_values()
        sampled = [
            sampling(feature) * gate[None, :, None, None].to(feature.dtype)
            for sampling, feature, gate in zip(
                self.sampling, features, gates
            )
        ]
        return self.output_norm(self.fusion(torch.cat(sampled, dim=1)))


class ScaleDecoupledReassemblyProjectorV37(nn.Module):
    """Retain v23 full-layer C2f fusion and route selected P3/P4 branches."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        levels: Sequence[str],
        routed_levels: Sequence[str],
        rank_channels: int = 64,
        cross_scale_mode: str = "none",
        cross_scale_rank: int = 32,
        use_phase_downsample: bool = True,
        num_blocks: int = 3,
        detail_groups: int = 16,
    ):
        super().__init__()
        del rank_channels
        unsupported = set(levels) - {"P3", "P4", "P5"}
        if unsupported:
            raise ValueError(
                f"SDSR-v37 does not support levels: {sorted(unsupported)}"
            )
        if not levels:
            raise ValueError("levels must contain at least one pyramid level")
        routed_levels = frozenset(routed_levels)
        unsupported_routes = routed_levels - {"P3", "P4"}
        if unsupported_routes:
            raise ValueError(
                "SDSR-v37 routing is only supported for P3/P4, got "
                f"{sorted(unsupported_routes)}"
            )
        missing_routes = routed_levels - set(levels)
        if missing_routes:
            raise ValueError(
                f"Routed levels are not produced: {sorted(missing_routes)}"
            )

        self.levels = list(levels)
        self.routed_levels = routed_levels
        self.branches = nn.ModuleDict()
        for level, scale in (("P3", 2), ("P4", 1)):
            if level not in self.levels:
                continue
            fusion_type = (
                LayerPreservingRoutedScaleFusion
                if level in routed_levels
                else ExactScaleFusion
            )
            self.branches[level] = fusion_type(
                in_channels,
                out_channels,
                scale=scale,
                num_blocks=num_blocks,
            )
        if "P5" in self.levels:
            self.branches["P5"] = DepthAdaptiveGroupedDetailP5Fusion(
                in_channels,
                out_channels,
                num_blocks=num_blocks,
                anti_alias=use_phase_downsample,
                maximum_groups=detail_groups,
                grouped_feature_index=-1,
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


class ScaleDecoupledReassemblyProjectorV37R1(
    ScaleDecoupledReassemblyProjectorV37
):
    """R1: route P3 while retaining exact v23 P4/P5."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, routed_levels=("P3",), **kwargs)


class ScaleDecoupledReassemblyProjectorV37R2(
    ScaleDecoupledReassemblyProjectorV37
):
    """R2: route P4 while retaining exact v23 P3/P5."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, routed_levels=("P4",), **kwargs)


class ScaleDecoupledReassemblyProjectorV37R3(
    ScaleDecoupledReassemblyProjectorV37
):
    """R3: independently route P3 and P4 while retaining all layers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, routed_levels=("P3", "P4"), **kwargs)
