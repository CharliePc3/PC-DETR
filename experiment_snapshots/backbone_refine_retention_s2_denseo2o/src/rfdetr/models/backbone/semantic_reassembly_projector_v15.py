# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""V11 reassembly with zero-initialized bottom-up scale calibration."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from rfdetr.models.backbone.semantic_reassembly_projector_v7 import (
    LowRankScaleMessage,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v11 import (
    ScaleDecoupledReassemblyProjectorV11,
)


class ZeroInitBottomUpScaleCalibration(nn.Module):
    """Propagate fine-to-coarse context without perturbing initial features."""

    def __init__(
        self,
        channels: int,
        levels: Sequence[str],
        rank_channels: int,
    ):
        super().__init__()
        self.levels = list(levels)
        self.bottom_up = nn.ModuleList(
            [
                LowRankScaleMessage(channels, rank_channels)
                for _ in range(max(len(self.levels) - 1, 0))
            ]
        )
        for message in self.bottom_up:
            nn.init.zeros_(message.channel_scale)

    def forward(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        outputs = dict(features)
        for edge_index in range(len(self.levels) - 1):
            low_level = self.levels[edge_index]
            high_level = self.levels[edge_index + 1]
            outputs[high_level] = outputs[high_level] + self.bottom_up[edge_index](
                outputs[low_level],
                outputs[high_level].shape[-2:],
            )
        return outputs


class ScaleDecoupledReassemblyProjectorV15(
    ScaleDecoupledReassemblyProjectorV11
):
    """Add stable fine-to-coarse semantic calibration to the v11 projector."""

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
        del cross_scale_mode
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            levels=levels,
            rank_channels=rank_channels,
            cross_scale_mode="none",
            cross_scale_rank=cross_scale_rank,
            use_phase_downsample=use_phase_downsample,
            num_blocks=num_blocks,
        )
        self.scale_calibration = ZeroInitBottomUpScaleCalibration(
            channels=out_channels,
            levels=levels,
            rank_channels=cross_scale_rank,
        )
