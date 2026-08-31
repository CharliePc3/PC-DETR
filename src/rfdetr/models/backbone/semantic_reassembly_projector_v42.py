# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Low-cost P5 depth priors on top of the v40 P4 reassembly path."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from rfdetr.models.backbone.semantic_reassembly_projector_v23 import (
    DepthAdaptiveGroupedDetailP5Fusion,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v40 import (
    ScaleDecoupledReassemblyProjectorV40P4Learnable,
)


UNIFORM_P5_GATES = (1.0, 1.0, 1.0, 1.0)
MILD_DEEP_P5_GATES = (0.70, 0.90, 1.10, 1.30)


class LearnableP5DepthGatedFusion(nn.Module):
    """Reweight downsampled ViT depths before the existing P5 C2f."""

    def __init__(
        self,
        base: DepthAdaptiveGroupedDetailP5Fusion,
        initial_gates: Sequence[float],
    ):
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
            downsample(feature) * gates[index].to(feature.dtype)
            for index, (downsample, feature) in enumerate(
                zip(self.base.downsampling, features)
            )
        ]
        return self.base.output_norm(self.base.fusion(torch.cat(sampled, dim=1)))


class ScaleDecoupledReassemblyProjectorV42(
    ScaleDecoupledReassemblyProjectorV40P4Learnable
):
    """Add a four-scalar learnable P5 depth prior to v40 learnable P4."""

    def __init__(self, *args, p5_initial_gates: Sequence[float], **kwargs):
        super().__init__(*args, **kwargs)
        if "P5" not in self.branches:
            raise ValueError("v42 P5 depth routing requires a P5 output")
        self.branches["P5"] = LearnableP5DepthGatedFusion(
            self.branches["P5"], initial_gates=p5_initial_gates
        )


class ScaleDecoupledReassemblyProjectorV42P5Uniform(
    ScaleDecoupledReassemblyProjectorV42
):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            p5_initial_gates=UNIFORM_P5_GATES,
            **kwargs,
        )


class ScaleDecoupledReassemblyProjectorV42P5Deep(
    ScaleDecoupledReassemblyProjectorV42
):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            p5_initial_gates=MILD_DEEP_P5_GATES,
            **kwargs,
        )
