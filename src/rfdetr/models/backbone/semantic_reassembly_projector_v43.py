# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Spatially conditioned P4 residual routing on top of SDSR-v40."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.backbone.projector import ConvX
from rfdetr.models.backbone.semantic_reassembly_projector_v40 import (
    ScaleDecoupledReassemblyProjectorV40P4Learnable,
    StrongPriorP4Fusion,
)


class BoundedSpatialDepthResidual(nn.Module):
    """Predict a smooth, bounded depth correction at each P4 location."""

    def __init__(
        self,
        channels: int,
        num_features: int,
        rank_channels: int,
        max_logit_delta: float,
    ):
        super().__init__()
        if rank_channels < 1:
            raise ValueError("rank_channels must be positive")
        if max_logit_delta <= 0:
            raise ValueError("max_logit_delta must be positive")
        self.num_features = num_features
        self.max_logit_delta = max_logit_delta
        self.shared_projection = ConvX(
            channels,
            rank_channels,
            kernel=1,
            act="silu",
            layer_norm=True,
        )
        route_channels = num_features * rank_channels
        self.local_mixing = ConvX(
            route_channels,
            route_channels,
            kernel=3,
            groups=route_channels,
            act="silu",
            layer_norm=True,
        )
        self.logit_projection = nn.Conv2d(
            route_channels,
            num_features,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.logit_projection.weight)
        nn.init.zeros_(self.logit_projection.bias)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(features) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {len(features)}"
            )
        route_features = list(features)
        resized_mask = None
        if padding_mask is not None:
            resized_mask = F.interpolate(
                padding_mask[:, None].float(),
                size=features[0].shape[-2:],
                mode="nearest",
            ).to(torch.bool)
            route_features = [
                feature.masked_fill(resized_mask, 0) for feature in route_features
            ]
        projected = [self.shared_projection(feature) for feature in route_features]
        logits = self.logit_projection(
            self.local_mixing(torch.cat(projected, dim=1))
        )
        residual = self.max_logit_delta * torch.tanh(logits.float())
        if resized_mask is not None:
            residual = residual.masked_fill(resized_mask, 0)
        return residual


class SpatialResidualP4Fusion(nn.Module):
    """Make the proven v40 P4 prior locally adaptive without replacing C2f."""

    def __init__(
        self,
        base: StrongPriorP4Fusion,
        channels: int,
        rank_channels: int,
        max_logit_delta: float,
    ):
        super().__init__()
        if base.dynamic_residual is not None:
            raise ValueError("Spatial routing expects the non-dynamic v40 P4 branch")
        self.base = base
        self.spatial_residual = BoundedSpatialDepthResidual(
            channels=channels,
            num_features=base.base.num_features,
            rank_channels=rank_channels,
            max_logit_delta=max_logit_delta,
        )

    def spatial_gates(
        self,
        sampled: Sequence[torch.Tensor],
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self.spatial_residual(sampled, padding_mask=padding_mask)
        logits = self.base.prior_logits[None, :, None, None].float() + residual
        return self.base.base.num_features * logits.softmax(dim=1)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(features) != self.base.base.num_features:
            raise ValueError(
                f"Expected {self.base.base.num_features} features, got {len(features)}"
            )
        sampled = [
            sampling(feature)
            for sampling, feature in zip(self.base.base.sampling, features)
        ]
        gates = self.spatial_gates(sampled, padding_mask=padding_mask)
        weighted = [
            feature * gates[:, index : index + 1].to(feature.dtype)
            for index, feature in enumerate(sampled)
        ]
        return self.base.base.output_norm(
            self.base.base.fusion(torch.cat(weighted, dim=1))
        )


class ScaleDecoupledReassemblyProjectorV43(
    ScaleDecoupledReassemblyProjectorV40P4Learnable
):
    """Use bounded spatial corrections around the v40 global P4 prior."""

    def __init__(self, *args, max_logit_delta: float, **kwargs):
        super().__init__(*args, **kwargs)
        if "P4" not in self.branches:
            raise ValueError("v43 spatial routing requires a P4 output")
        in_channels = tuple(args[0] if args else kwargs["in_channels"])
        if not in_channels or len(set(in_channels)) != 1:
            raise ValueError("v43 expects equal-channel ViT features")
        rank_channels = max(8, int(kwargs.get("rank_channels", 64)) // 8)

        rng_state = torch.get_rng_state()
        try:
            self.branches["P4"] = SpatialResidualP4Fusion(
                self.branches["P4"],
                channels=in_channels[0],
                rank_channels=rank_channels,
                max_logit_delta=max_logit_delta,
            )
        finally:
            torch.set_rng_state(rng_state)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        del image
        outputs = {}
        for level in self.levels:
            if level == "P4":
                outputs[level] = self.branches[level](features, padding_mask=mask)
            else:
                outputs[level] = self.branches[level](features)
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


class ScaleDecoupledReassemblyProjectorV43Spatial010(
    ScaleDecoupledReassemblyProjectorV43
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, max_logit_delta=0.10, **kwargs)


class ScaleDecoupledReassemblyProjectorV43Spatial020(
    ScaleDecoupledReassemblyProjectorV43
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, max_logit_delta=0.20, **kwargs)
