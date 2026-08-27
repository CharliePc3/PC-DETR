# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Deep P5 mixing with a conservative image-detail residual at P3."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.backbone.projector import LayerNorm
from rfdetr.models.backbone.semantic_reassembly_projector import (
    DirectionalDetailStem,
    _group_norm,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v23 import (
    ScaleDecoupledReassemblyProjectorV23,
)


class SemanticGatedImageDetail(nn.Module):
    """Inject local image detail only where it agrees with the P3 semantics."""

    def __init__(
        self,
        channels: int,
        detail_channels: int,
    ):
        super().__init__()
        self.detail_stem = DirectionalDetailStem(detail_channels)
        self.semantic_context = nn.Sequential(
            nn.Conv2d(channels, detail_channels, kernel_size=1, bias=False),
            _group_norm(detail_channels),
            nn.SiLU(inplace=True),
        )
        self.confidence = nn.Sequential(
            nn.Conv2d(
                2 * detail_channels,
                detail_channels,
                kernel_size=3,
                padding=1,
                groups=detail_channels,
                bias=False,
            ),
            _group_norm(detail_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(detail_channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.confidence[-1].weight)
        nn.init.constant_(self.confidence[-1].bias, -1.5)

        self.detail_projection = nn.Sequential(
            nn.Conv2d(
                detail_channels,
                detail_channels,
                kernel_size=3,
                padding=1,
                groups=detail_channels,
                bias=False,
            ),
            _group_norm(detail_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(detail_channels, channels, kernel_size=1, bias=False),
            LayerNorm(channels),
        )
        self.channel_gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(
        self,
        semantic: torch.Tensor,
        image: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        detail = self.detail_stem(image, mask)
        if detail.shape[-2:] != semantic.shape[-2:]:
            detail = F.interpolate(
                detail,
                size=semantic.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        context = self.semantic_context(semantic)
        confidence = torch.sigmoid(
            self.confidence(torch.cat((context, detail), dim=1))
        )
        residual = self.detail_projection(detail)
        output = semantic + torch.tanh(self.channel_gate) * confidence * residual

        if mask is not None:
            output_mask = F.interpolate(
                mask[:, None].float(),
                size=semantic.shape[-2:],
                mode="nearest",
            ).to(torch.bool)
            output = output.masked_fill(output_mask, 0)
        return output


class ScaleDecoupledReassemblyProjectorV30(
    ScaleDecoupledReassemblyProjectorV23
):
    """Preserve v23 and add an initially inactive P3 image-detail path."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        levels: Sequence[str],
        rank_channels: int = 64,
        detail_channels: int = 32,
        cross_scale_mode: str = "none",
        cross_scale_rank: int = 32,
        use_phase_downsample: bool = True,
        num_blocks: int = 3,
        p5_detail_groups: int = 16,
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
            detail_groups=p5_detail_groups,
        )
        self.p3_index = self.levels.index("P3") if "P3" in self.levels else None
        with torch.random.fork_rng(devices=[]):
            self.p3_detail = (
                SemanticGatedImageDetail(out_channels, detail_channels)
                if self.p3_index is not None
                else None
            )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        outputs = super().forward(features, image=None, mask=mask)
        if self.p3_detail is not None:
            if image is None:
                raise ValueError("SDSR-v30 P3 detail refinement requires the input image")
            outputs[self.p3_index] = self.p3_detail(
                outputs[self.p3_index], image, mask
            )
        return outputs
