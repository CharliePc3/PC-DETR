# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Static normalized ViT-depth prior while retaining the v23 P4 C2f path."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from rfdetr.models.backbone.semantic_reassembly_projector_v10 import (
    ExactScaleFusion,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v23 import (
    ScaleDecoupledReassemblyProjectorV23,
)


class StaticRoutedP4Fusion(nn.Module):
    """Apply four learned normalized depth weights before the existing C2f."""

    def __init__(self, base: ExactScaleFusion):
        super().__init__()
        self.base = base
        self.layer_logits = nn.Parameter(torch.zeros(base.num_features))

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


class ScaleDecoupledReassemblyProjectorV39P4Static(
    ScaleDecoupledReassemblyProjectorV23
):
    """Test whether v38's gain comes from a global rather than dynamic prior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "P4" not in self.branches:
            raise ValueError("P4 static routing requires a P4 output")
        self.branches["P4"] = StaticRoutedP4Fusion(self.branches["P4"])
