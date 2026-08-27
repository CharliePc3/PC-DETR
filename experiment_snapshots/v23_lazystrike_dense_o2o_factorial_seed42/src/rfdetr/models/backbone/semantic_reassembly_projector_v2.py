# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Layer-preserving scale-decoupled semantic reassembly projector."""

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
    StridedProjection,
)


class LightweightScaleFusion(nn.Module):
    """Fuse concatenated low-rank layer features with lightweight spatial mixing."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.input_projection = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=1, bias=False)
        self.input_norm = LayerNorm(2 * out_channels)
        self.activation = nn.SiLU()
        self.local_mixing = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            LayerNorm(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )
        self.output_projection = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1, bias=False)
        self.output_norm = LayerNorm(out_channels)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        semantic, spatial = self.activation(self.input_norm(self.input_projection(feature))).chunk(2, dim=1)
        spatial = spatial + self.local_mixing(spatial)
        fused = self.output_projection(torch.cat((semantic, spatial), dim=1))
        return self.output_norm(fused)


class LayerPreservingScaleFusion(nn.Module):
    """Keep each ViT layer in a separate low-rank subspace until scale fusion."""

    def __init__(
        self,
        in_channels: Sequence[int],
        rank_channels: int,
        out_channels: int,
        levels: Sequence[str],
    ):
        super().__init__()
        if not in_channels:
            raise ValueError("in_channels must contain at least one feature")
        if rank_channels < 1:
            raise ValueError("rank_channels must be positive")

        self.levels = list(levels)
        self.rank_channels = rank_channels
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm(channels),
                    nn.Conv2d(channels, rank_channels, kernel_size=1, bias=False),
                )
                for channels in in_channels
            ]
        )
        self.scale_gate_logits = nn.Parameter(
            self._initial_gate_logits(self.levels, len(in_channels), rank_channels)
        )
        fused_channels = len(in_channels) * rank_channels
        self.fusions = nn.ModuleDict(
            {level: LightweightScaleFusion(fused_channels, out_channels) for level in self.levels}
        )

    @staticmethod
    def _initial_gate_logits(levels: Sequence[str], num_layers: int, rank_channels: int) -> torch.Tensor:
        depth = torch.linspace(-1.0, 1.0, num_layers)
        level_slopes = {"P3": -0.1, "P4": 0.0, "P5": 0.1}
        gate_values = []
        for level in levels:
            values = 1.0 + level_slopes[level] * depth
            logits = torch.log(values / (2.0 - values))
            gate_values.append(logits[:, None].expand(-1, rank_channels))
        return torch.stack(gate_values)

    def gate_values(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.scale_gate_logits)

    def forward(self, features: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        if len(features) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} features, got {len(features)}")

        target_size = features[-1].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            projected.append(projection(feature))

        stacked = torch.stack(projected, dim=1)
        gates = self.gate_values().to(dtype=stacked.dtype)
        outputs = {}
        for level_index, level in enumerate(self.levels):
            gated = stacked * gates[level_index][None, :, :, None, None]
            concatenated = gated.flatten(1, 2)
            outputs[level] = self.fusions[level](concatenated)
        return outputs


class ScaleDecoupledReassemblyProjectorV2(nn.Module):
    """Build P3/P4/P5 without collapsing intermediate ViT layers into a sum."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        levels: Sequence[str],
        rank_channels: int = 64,
        detail_channels: int = 32,
        use_local_reassembly: bool = True,
        use_directional_guide: bool = True,
    ):
        super().__init__()
        unsupported = set(levels) - {"P3", "P4", "P5"}
        if unsupported:
            raise ValueError(f"SDSR-v2 Projector does not support levels: {sorted(unsupported)}")
        if not levels:
            raise ValueError("levels must contain at least one pyramid level")

        self.levels = list(levels)
        self.use_local_reassembly = use_local_reassembly and "P3" in self.levels
        self.use_directional_guide = use_directional_guide and self.use_local_reassembly
        self.layer_fusion = LayerPreservingScaleFusion(
            in_channels=in_channels,
            rank_channels=rank_channels,
            out_channels=out_channels,
            levels=self.levels,
        )
        self.detail_stem = DirectionalDetailStem(detail_channels) if self.use_directional_guide else None
        self.local_reassembly = (
            SemanticLocalReassembly(out_channels, detail_channels, self.use_directional_guide)
            if self.use_local_reassembly
            else None
        )
        self.p5_downsample = StridedProjection(out_channels) if "P5" in self.levels else None
        self.refine = nn.ModuleDict({level: LightRefineBlock(out_channels) for level in self.levels})

    def forward(
        self,
        features: Sequence[torch.Tensor],
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        scale_sources = self.layer_fusion(features)
        outputs = {}

        if "P3" in self.levels:
            p3_source = scale_sources["P3"]
            if self.local_reassembly is not None:
                guide = None
                if self.detail_stem is not None:
                    if image is None:
                        raise ValueError("SDSR-v2 directional guide requires the input image")
                    guide = self.detail_stem(image, mask)
                p3 = self.local_reassembly(p3_source, guide, mask)
            else:
                p3 = F.interpolate(p3_source, scale_factor=2.0, mode="bilinear", align_corners=False)
            outputs["P3"] = self.refine["P3"](p3)

        if "P4" in self.levels:
            outputs["P4"] = self.refine["P4"](scale_sources["P4"])

        if "P5" in self.levels:
            p5 = self.p5_downsample(scale_sources["P5"])
            outputs["P5"] = self.refine["P5"](p5)

        return [outputs[level] for level in self.levels]
