# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""V11 reassembly with learnable binomial-initialized P5 low-pass bases."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from rfdetr.models.backbone.semantic_reassembly_projector_v11 import (
    ScaleDecoupledReassemblyProjectorV11,
)


class ScaleDecoupledReassemblyProjectorV18(
    ScaleDecoupledReassemblyProjectorV11
):
    """Let each P5 low-pass basis adapt while retaining v11 initialization."""

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
        if not use_phase_downsample:
            raise ValueError("SDSR-v18 requires phase-aware downsampling")
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            levels=levels,
            rank_channels=rank_channels,
            cross_scale_mode=cross_scale_mode,
            cross_scale_rank=cross_scale_rank,
            use_phase_downsample=True,
            num_blocks=num_blocks,
        )

        if "P5" not in self.branches:
            return
        for channels, downsample in zip(
            in_channels, self.branches["P5"].downsampling
        ):
            rng_state = torch.get_rng_state()
            primary = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=channels,
                bias=False,
            )
            torch.set_rng_state(rng_state)
            with torch.no_grad():
                primary.weight.copy_(
                    downsample.blur_kernel.expand(channels, 1, -1, -1)
                )
            downsample.primary = primary
