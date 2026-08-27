# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Dense early-layer detail sampling with an exact lightweight start."""

from __future__ import annotations

from typing import Sequence

from rfdetr.models.backbone.semantic_reassembly_projector_v24 import (
    ScaleDecoupledReassemblyProjectorV24,
)


class ScaleDecoupledReassemblyProjectorV25(
    ScaleDecoupledReassemblyProjectorV24
):
    """Use dense detail sampling only where shallow spatial features need it."""

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
            detail_groups=1,
        )
