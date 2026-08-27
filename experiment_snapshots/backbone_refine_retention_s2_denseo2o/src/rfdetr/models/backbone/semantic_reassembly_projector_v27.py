# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Wider deepest-layer grouped detail sampling for P5 reassembly."""

from __future__ import annotations

from typing import Sequence

from rfdetr.models.backbone.semantic_reassembly_projector_v23 import (
    ScaleDecoupledReassemblyProjectorV23,
)


class ScaleDecoupledReassemblyProjectorV27(
    ScaleDecoupledReassemblyProjectorV23
):
    """Let each deepest-layer detail group mix 48 DINOv3-S channels."""

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
            detail_groups=8,
        )
