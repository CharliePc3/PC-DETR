# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Non-destructive deep semantic basis expansion for P5 reassembly."""

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
from rfdetr.models.backbone.semantic_reassembly_projector_v11 import (
    TwoBasisDownsample,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v22 import (
    _largest_group_divisor,
)


def _expand_fusion_input_exactly(
    base_fusion: C2f,
    base_channels: int,
    expanded_channels: int,
    out_channels: int,
    num_blocks: int,
) -> C2f:
    """Append zero-initialized C2f input columns while preserving all outputs."""

    with torch.random.fork_rng(devices=[]):
        expanded = C2f(
            expanded_channels,
            out_channels,
            n=num_blocks,
            layer_norm=True,
        )
    base_parameters = dict(base_fusion.named_parameters())
    with torch.no_grad():
        for name, parameter in expanded.named_parameters():
            source = base_parameters[name]
            if name == "cv1.conv.weight":
                parameter.zero_()
                parameter[:, :base_channels].copy_(source)
            else:
                parameter.copy_(source)
    return expanded


class DeepSemanticBasis(nn.Module):
    """Produce a grouped spatial-semantic basis from the deepest ViT feature."""

    def __init__(self, channels: int, maximum_groups: int):
        super().__init__()
        groups = _largest_group_divisor(channels, maximum_groups)
        self.sampling = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.norm = LayerNorm(channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.sampling(feature)))


class ResidualBasisP5Fusion(nn.Module):
    """Keep both v11 bases and append a learnable deepest-layer basis."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_blocks: int,
        anti_alias: bool,
        maximum_groups: int,
    ):
        super().__init__()
        if not in_channels:
            raise ValueError("in_channels must contain at least one feature")
        self.num_features = len(in_channels)
        self.downsampling = nn.ModuleList(
            [
                TwoBasisDownsample(channels, anti_alias=anti_alias)
                for channels in in_channels
            ]
        )
        base_channels = 2 * sum(in_channels)
        base_fusion = C2f(
            base_channels,
            out_channels,
            n=num_blocks,
            layer_norm=True,
        )
        with torch.random.fork_rng(devices=[]):
            self.semantic_basis = DeepSemanticBasis(
                in_channels[-1], maximum_groups=maximum_groups
            )
        self.fusion = _expand_fusion_input_exactly(
            base_fusion,
            base_channels=base_channels,
            expanded_channels=base_channels + in_channels[-1],
            out_channels=out_channels,
            num_blocks=num_blocks,
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
        sampled.append(self.semantic_basis(features[-1]))
        return self.output_norm(self.fusion(torch.cat(sampled, dim=1)))


class ScaleDecoupledReassemblyProjectorV29(nn.Module):
    """Add one semantic P5 basis without replacing pretrained detail signals."""

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
        detail_groups: int = 16,
    ):
        super().__init__()
        del rank_channels
        unsupported = set(levels) - {"P3", "P4", "P5"}
        if unsupported:
            raise ValueError(
                f"SDSR-v29 Projector does not support levels: {sorted(unsupported)}"
            )
        if not levels:
            raise ValueError("levels must contain at least one pyramid level")
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if detail_groups < 1:
            raise ValueError("detail_groups must be positive")

        self.levels = list(levels)
        self.branches = nn.ModuleDict()
        if "P3" in self.levels:
            self.branches["P3"] = ExactScaleFusion(
                in_channels, out_channels, scale=2, num_blocks=num_blocks
            )
        if "P4" in self.levels:
            self.branches["P4"] = ExactScaleFusion(
                in_channels, out_channels, scale=1, num_blocks=num_blocks
            )
        if "P5" in self.levels:
            self.branches["P5"] = ResidualBasisP5Fusion(
                in_channels,
                out_channels,
                num_blocks=num_blocks,
                anti_alias=use_phase_downsample,
                maximum_groups=detail_groups,
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
