# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Training-only annealed spatial routing on top of SDSR-v45."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from rfdetr.models.backbone.semantic_reassembly_projector_v45 import (
    MeanCenteredSpatialResidualP4Fusion,
    ScaleDecoupledReassemblyProjectorV45SpatialCentered020,
)


class AnnealedSpatialResidualP4Fusion(MeanCenteredSpatialResidualP4Fusion):
    """Anneal local routing to the v40 global prior before the LR drop."""

    def __init__(self, *args, full_until_epoch: int, zero_at_epoch: int, **kwargs):
        super().__init__(*args, **kwargs)
        if full_until_epoch < 0:
            raise ValueError("full_until_epoch must be non-negative")
        if zero_at_epoch <= full_until_epoch:
            raise ValueError("zero_at_epoch must be greater than full_until_epoch")
        self.full_until_epoch = int(full_until_epoch)
        self.zero_at_epoch = int(zero_at_epoch)
        self.register_buffer("routing_scale", torch.ones(()), persistent=True)

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch <= self.full_until_epoch:
            scale = 1.0
        elif epoch >= self.zero_at_epoch:
            scale = 0.0
        else:
            progress = (epoch - self.full_until_epoch) / (
                self.zero_at_epoch - self.full_until_epoch
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        self.routing_scale.fill_(scale)

    def spatial_gates(
        self,
        sampled: Sequence[torch.Tensor],
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = float(self.routing_scale.item())
        if scale == 0.0:
            return self.base.normalized_gates()[None, :, None, None]
        residual = self.spatial_residual(sampled, padding_mask=padding_mask)
        logits = (
            self.base.prior_logits[None, :, None, None].float()
            + scale * residual
        )
        return self.base.base.num_features * logits.softmax(dim=1)


class ScaleDecoupledReassemblyProjectorV46AnnealedCentered020(
    ScaleDecoupledReassemblyProjectorV45SpatialCentered020
):
    """Use centered spatial routing as an optimization curriculum only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        current = self.branches["P4"]
        in_channels = tuple(args[0] if args else kwargs["in_channels"])
        rank_channels = max(8, int(kwargs.get("rank_channels", 64)) // 8)

        rng_state = torch.get_rng_state()
        try:
            self.branches["P4"] = AnnealedSpatialResidualP4Fusion(
                current.base,
                channels=in_channels[0],
                rank_channels=rank_channels,
                max_logit_delta=0.20,
                full_until_epoch=9,
                zero_at_epoch=20,
            )
        finally:
            torch.set_rng_state(rng_state)

    def set_epoch(self, epoch: int) -> None:
        self.branches["P4"].set_epoch(epoch)
