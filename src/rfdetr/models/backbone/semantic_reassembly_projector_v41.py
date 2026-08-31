# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Low-cost P3 depth priors on top of the v40 P4 reassembly path."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from rfdetr.models.backbone.semantic_reassembly_projector_v10 import (
    ExactScaleFusion,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v40 import (
    ScaleDecoupledReassemblyProjectorV40P4Learnable,
)


UNIFORM_P3_GATES = (1.0, 1.0, 1.0, 1.0)
MILD_SHALLOW_P3_GATES = (1.25, 1.10, 0.95, 0.70)


class LearnableDepthGatedFusion(nn.Module):
    """Reweight ViT depths while preserving the existing sampler and C2f."""

    def __init__(self, base: ExactScaleFusion, initial_gates: Sequence[float]):
        super().__init__()
        if base.num_features != len(initial_gates):
            raise ValueError(
                f"Expected {base.num_features} gates, got {len(initial_gates)}"
            )
        if any(gate <= 0 for gate in initial_gates):
            raise ValueError("All initial depth gates must be positive")
        self.base = base
        self.layer_logits = nn.Parameter(
            torch.tensor(
                [math.log(gate) for gate in initial_gates], dtype=torch.float32
            )
        )

    def normalized_gates(self) -> torch.Tensor:
        return self.base.num_features * self.layer_logits.float().softmax(dim=0)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.base.num_features:
            raise ValueError(
                f"Expected {self.base.num_features} features, got {len(features)}"
            )
        gates = self.normalized_gates()
        sampled = [
            sampling(feature) * gates[index].to(feature.dtype)
            for index, (sampling, feature) in enumerate(
                zip(self.base.sampling, features)
            )
        ]
        return self.base.output_norm(self.base.fusion(torch.cat(sampled, dim=1)))


class ScaleDecoupledReassemblyProjectorV41(
    ScaleDecoupledReassemblyProjectorV40P4Learnable
):
    """Add a four-scalar learnable P3 depth prior to v40 learnable P4."""

    def __init__(self, *args, p3_initial_gates: Sequence[float], **kwargs):
        super().__init__(*args, **kwargs)
        if "P3" not in self.branches:
            raise ValueError("v41 P3 depth routing requires a P3 output")
        self.branches["P3"] = LearnableDepthGatedFusion(
            self.branches["P3"], initial_gates=p3_initial_gates
        )


class ScaleDecoupledReassemblyProjectorV41P3Uniform(
    ScaleDecoupledReassemblyProjectorV41
):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            p3_initial_gates=UNIFORM_P3_GATES,
            **kwargs,
        )


class ScaleDecoupledReassemblyProjectorV41P3Shallow(
    ScaleDecoupledReassemblyProjectorV41
):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            p3_initial_gates=MILD_SHALLOW_P3_GATES,
            **kwargs,
        )
