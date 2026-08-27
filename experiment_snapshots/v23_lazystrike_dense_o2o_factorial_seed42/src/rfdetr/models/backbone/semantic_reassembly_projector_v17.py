# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""V16 reassembly with a faster zero-initialized P4-to-P5 gate."""

from __future__ import annotations

from typing import Sequence

import torch

from rfdetr.models.backbone.semantic_reassembly_projector_v16 import (
    P4ToP5PoolCalibration,
    ScaleDecoupledReassemblyProjectorV16,
)


class FastP4ToP5PoolCalibration(P4ToP5PoolCalibration):
    """Preserve zero output while increasing the gate's initial gradient."""

    def __init__(
        self,
        channels: int,
        levels: Sequence[str],
        gate_gain: float = 2.0,
    ):
        super().__init__(channels=channels, levels=levels)
        self.gate_gain = gate_gain

    def forward(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return features

        outputs = dict(features)
        pooled_p4 = torch.nn.functional.adaptive_avg_pool2d(
            outputs["P4"], outputs["P5"].shape[-2:]
        )
        scale = torch.tanh(self.gate_gain * self.channel_scale).to(
            dtype=pooled_p4.dtype
        )
        outputs["P5"] = outputs["P5"] + scale * pooled_p4
        return outputs


class ScaleDecoupledReassemblyProjectorV17(
    ScaleDecoupledReassemblyProjectorV16
):
    """Accelerate v16's deterministic semantic calibration path."""

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
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            levels=levels,
            rank_channels=rank_channels,
            cross_scale_mode=cross_scale_mode,
            cross_scale_rank=cross_scale_rank,
            use_phase_downsample=use_phase_downsample,
            num_blocks=num_blocks,
        )
        self.scale_calibration = FastP4ToP5PoolCalibration(
            channels=out_channels,
            levels=levels,
        )
