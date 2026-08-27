# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""V11 reassembly with a zero-initialized deterministic P4-to-P5 residual."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.backbone.semantic_reassembly_projector_v11 import (
    ScaleDecoupledReassemblyProjectorV11,
)


class P4ToP5PoolCalibration(nn.Module):
    """Calibrate P5 with pooled P4 semantics using only a channel-wise gate."""

    def __init__(self, channels: int, levels: Sequence[str]):
        super().__init__()
        self.enabled = "P4" in levels and "P5" in levels
        if self.enabled:
            self.channel_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.register_parameter("channel_scale", None)

    def forward(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return features

        outputs = dict(features)
        pooled_p4 = F.adaptive_avg_pool2d(
            outputs["P4"], outputs["P5"].shape[-2:]
        )
        scale = torch.tanh(self.channel_scale).to(dtype=pooled_p4.dtype)
        outputs["P5"] = outputs["P5"] + scale * pooled_p4
        return outputs


class ScaleDecoupledReassemblyProjectorV16(
    ScaleDecoupledReassemblyProjectorV11
):
    """Inject fine-scale semantics into P5 without a learned spatial projection."""

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
        self.scale_calibration = P4ToP5PoolCalibration(
            channels=out_channels,
            levels=levels,
        )
