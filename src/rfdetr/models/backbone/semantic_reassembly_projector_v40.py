# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Strong P4 depth-prior controls derived from the v38 routing study."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from rfdetr.models.backbone.semantic_reassembly_projector_v10 import (
    ExactScaleFusion,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v23 import (
    ScaleDecoupledReassemblyProjectorV23,
)


STRONG_P4_GATES = (0.2485995, 0.4129705, 0.7943345, 2.5440950)


class BoundedImageConditionedResidual(nn.Module):
    """Predict a small per-image correction around a fixed depth prior."""

    def __init__(
        self,
        channels: int,
        num_features: int,
        rank_channels: int,
        max_logit_delta: float,
    ):
        super().__init__()
        self.num_features = num_features
        self.max_logit_delta = max_logit_delta
        self.descriptor_projection = nn.Sequential(
            nn.Linear(channels, rank_channels),
            nn.GELU(),
        )
        hidden_channels = max(rank_channels, num_features * 2)
        self.router = nn.Sequential(
            nn.Linear(num_features * rank_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, num_features),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        descriptors = [
            self.descriptor_projection(feature.mean(dim=(-2, -1)))
            for feature in features
        ]
        logits = self.router(torch.cat(descriptors, dim=-1))
        return self.max_logit_delta * torch.tanh(logits.float())


class StrongPriorP4Fusion(nn.Module):
    """Apply a fixed or learnable global prior and an optional bounded residual."""

    def __init__(
        self,
        base: ExactScaleFusion,
        in_channels: Sequence[int],
        rank_channels: int,
        learnable_prior: bool,
        use_dynamic_residual: bool,
        max_logit_delta: float = 0.25,
    ):
        super().__init__()
        if base.num_features != len(STRONG_P4_GATES):
            raise ValueError("The strong P4 prior expects exactly four ViT features")
        if not in_channels or len(set(in_channels)) != 1:
            raise ValueError("The P4 dynamic residual expects equal input channels")
        self.base = base
        prior_logits = torch.tensor(
            [math.log(gate) for gate in STRONG_P4_GATES], dtype=torch.float32
        )
        if learnable_prior:
            self.prior_logits = nn.Parameter(prior_logits)
        else:
            self.register_buffer("prior_logits", prior_logits)
        self.dynamic_residual = (
            BoundedImageConditionedResidual(
                channels=in_channels[0],
                num_features=base.num_features,
                rank_channels=rank_channels,
                max_logit_delta=max_logit_delta,
            )
            if use_dynamic_residual
            else None
        )

    def normalized_gates(
        self, features: Sequence[torch.Tensor] | None = None
    ) -> torch.Tensor:
        logits = self.prior_logits
        if self.dynamic_residual is not None:
            if features is None:
                raise ValueError("features are required for dynamic residual routing")
            logits = logits[None] + self.dynamic_residual(features)
        return self.base.num_features * logits.float().softmax(dim=-1)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.base.num_features:
            raise ValueError(
                f"Expected {self.base.num_features} features, got {len(features)}"
            )
        gates = self.normalized_gates(features if self.dynamic_residual else None)
        if gates.ndim == 1:
            sampled = [
                sampling(feature) * gates[index].to(feature.dtype)
                for index, (sampling, feature) in enumerate(
                    zip(self.base.sampling, features)
                )
            ]
        else:
            sampled = [
                sampling(feature)
                * gates[:, index][:, None, None, None].to(feature.dtype)
                for index, (sampling, feature) in enumerate(
                    zip(self.base.sampling, features)
                )
            ]
        return self.base.output_norm(self.base.fusion(torch.cat(sampled, dim=1)))


class ScaleDecoupledReassemblyProjectorV40(
    ScaleDecoupledReassemblyProjectorV23
):
    """Wrap the v23 P4 branch with one strong-prior attribution variant."""

    def __init__(
        self,
        *args,
        learnable_prior: bool,
        use_dynamic_residual: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if "P4" not in self.branches:
            raise ValueError("P4 strong-prior routing requires a P4 output")
        in_channels = tuple(args[0] if args else kwargs["in_channels"])
        rank_channels = max(8, int(kwargs.get("rank_channels", 64)) // 4)

        rng_state = torch.get_rng_state()
        try:
            self.branches["P4"] = StrongPriorP4Fusion(
                self.branches["P4"],
                in_channels=in_channels,
                rank_channels=rank_channels,
                learnable_prior=learnable_prior,
                use_dynamic_residual=use_dynamic_residual,
            )
        finally:
            torch.set_rng_state(rng_state)


class ScaleDecoupledReassemblyProjectorV40P4Fixed(
    ScaleDecoupledReassemblyProjectorV40
):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            learnable_prior=False,
            use_dynamic_residual=False,
            **kwargs,
        )


class ScaleDecoupledReassemblyProjectorV40P4Learnable(
    ScaleDecoupledReassemblyProjectorV40
):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            learnable_prior=True,
            use_dynamic_residual=False,
            **kwargs,
        )


class ScaleDecoupledReassemblyProjectorV40P4BoundedDynamic(
    ScaleDecoupledReassemblyProjectorV40
):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            learnable_prior=False,
            use_dynamic_residual=True,
            **kwargs,
        )
