# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Content-adaptive residual reassembly while retaining the v23 C2f paths."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.backbone.semantic_reassembly_projector import (
    _bilinear_phase_prior,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v10 import (
    ExactScaleFusion,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v23 import (
    ScaleDecoupledReassemblyProjectorV23,
)


class LowRankLocalReassemblyResidual(nn.Module):
    """Predict a shared 3x3 local-attention residual in a low-rank subspace."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank_channels: int,
        context_channels: int = 16,
    ):
        super().__init__()
        if rank_channels < 1 or context_channels < 1:
            raise ValueError("rank_channels and context_channels must be positive")
        self.kernel_size = 3
        self.rank_channels = rank_channels
        self.source_projection = nn.Conv2d(
            in_channels, rank_channels, kernel_size=1, bias=False
        )
        self.context = nn.Sequential(
            nn.Conv2d(
                rank_channels,
                context_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(1, context_channels),
            nn.GELU(),
            nn.Conv2d(
                context_channels,
                context_channels,
                kernel_size=3,
                padding=1,
                groups=context_channels,
                bias=False,
            ),
            nn.GELU(),
        )
        self.weight_predictor = nn.Conv2d(
            context_channels,
            4 * self.kernel_size**2,
            kernel_size=1,
        )
        nn.init.normal_(self.weight_predictor.weight, std=1.0e-3)
        nn.init.zeros_(self.weight_predictor.bias)
        self.output_projection = nn.Conv2d(
            rank_channels, out_channels, kernel_size=1, bias=False
        )
        self.register_buffer(
            "phase_prior",
            _bilinear_phase_prior(self.kernel_size),
            persistent=False,
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        source = self.source_projection(feature)
        batch, channels, height, width = source.shape
        logits = self.weight_predictor(self.context(source))
        logits = logits.reshape(
            batch,
            4,
            self.kernel_size**2,
            height,
            width,
        )
        logits = logits + self.phase_prior[None, :, :, None, None].to(
            dtype=logits.dtype
        )

        padding = self.kernel_size // 2
        padded = F.pad(
            source,
            (padding, padding, padding, padding),
            mode="replicate",
        )
        patches = F.unfold(padded, kernel_size=self.kernel_size)
        patches = patches.reshape(
            batch,
            channels,
            self.kernel_size**2,
            height,
            width,
        )
        weights = logits.float().softmax(dim=2).to(source.dtype)
        children = torch.einsum("bckhw,bpkhw->bcphw", patches, weights)
        local = F.pixel_shuffle(
            children.reshape(batch, channels * 4, height, width),
            upscale_factor=2,
        )
        bilinear = F.interpolate(
            source,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        return self.output_projection(local - bilinear)


class LocalResidualP3Fusion(nn.Module):
    """Retain ConvTranspose+C2f and add shared low-rank local reassembly."""

    def __init__(
        self,
        base: ExactScaleFusion,
        in_channels: Sequence[int],
        rank_channels: int,
    ):
        super().__init__()
        if not in_channels or len(set(in_channels)) != 1:
            raise ValueError("The shared P3 residual expects equal input channels")
        if not all(isinstance(layer, nn.ConvTranspose2d) for layer in base.sampling):
            raise ValueError("The P3 base must use transposed-convolution sampling")
        sampled_channels = base.sampling[0].out_channels
        if not all(
            layer.out_channels == sampled_channels for layer in base.sampling
        ):
            raise ValueError("All P3 samplers must have equal output channels")

        self.base = base
        self.local_residual = LowRankLocalReassemblyResidual(
            in_channels=in_channels[0],
            out_channels=sampled_channels,
            rank_channels=rank_channels,
        )
        self.residual_scales = nn.Parameter(torch.zeros(len(in_channels)))

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.base.num_features:
            raise ValueError(
                f"Expected {self.base.num_features} features, got {len(features)}"
            )
        sampled = []
        for index, (sampling, feature) in enumerate(
            zip(self.base.sampling, features)
        ):
            baseline = sampling(feature)
            residual = self.local_residual(feature)
            scale = torch.tanh(self.residual_scales[index]).to(
                dtype=baseline.dtype
            )
            sampled.append(baseline + scale * residual)
        return self.base.output_norm(self.base.fusion(torch.cat(sampled, dim=1)))


class ImageConditionedLayerRouter(nn.Module):
    """Produce normalized per-image layer weights from all ViT features."""

    def __init__(
        self,
        channels: int,
        num_features: int,
        rank_channels: int,
    ):
        super().__init__()
        if channels < 1 or num_features < 1 or rank_channels < 1:
            raise ValueError("Router dimensions must be positive")
        self.num_features = num_features
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
        if len(features) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {len(features)}"
            )
        descriptors = [
            self.descriptor_projection(feature.mean(dim=(-2, -1)))
            for feature in features
        ]
        logits = self.router(torch.cat(descriptors, dim=-1))
        return self.num_features * logits.float().softmax(dim=-1).to(
            dtype=features[0].dtype
        )


class DynamicRoutedP4Fusion(nn.Module):
    """Retain the P4 C2f while making layer weighting image-dependent."""

    def __init__(
        self,
        base: ExactScaleFusion,
        in_channels: Sequence[int],
        rank_channels: int,
    ):
        super().__init__()
        if not in_channels or len(set(in_channels)) != 1:
            raise ValueError("The shared P4 router expects equal input channels")
        self.base = base
        self.layer_router = ImageConditionedLayerRouter(
            channels=in_channels[0],
            num_features=len(in_channels),
            rank_channels=rank_channels,
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.base.num_features:
            raise ValueError(
                f"Expected {self.base.num_features} features, got {len(features)}"
            )
        gates = self.layer_router(features)
        sampled = [
            sampling(feature)
            * gates[:, index][:, None, None, None].to(feature.dtype)
            for index, (sampling, feature) in enumerate(
                zip(self.base.sampling, features)
            )
        ]
        return self.base.output_norm(self.base.fusion(torch.cat(sampled, dim=1)))


class ScaleDecoupledReassemblyProjectorV38(
    ScaleDecoupledReassemblyProjectorV23
):
    """Add genuine content-adaptive paths without removing v23 fusion."""

    def __init__(
        self,
        *args,
        use_p3_local_residual: bool,
        use_p4_dynamic_routing: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        in_channels = tuple(args[0] if args else kwargs["in_channels"])
        rank_channels = int(kwargs.get("rank_channels", 64))

        # Added random modules must not perturb downstream detector initialization.
        rng_state = torch.get_rng_state()
        try:
            if use_p3_local_residual:
                if "P3" not in self.branches:
                    raise ValueError("P3 local residual requires a P3 output")
                self.branches["P3"] = LocalResidualP3Fusion(
                    self.branches["P3"],
                    in_channels=in_channels,
                    rank_channels=rank_channels,
                )
            if use_p4_dynamic_routing:
                if "P4" not in self.branches:
                    raise ValueError("P4 dynamic routing requires a P4 output")
                self.branches["P4"] = DynamicRoutedP4Fusion(
                    self.branches["P4"],
                    in_channels=in_channels,
                    rank_channels=max(8, rank_channels // 4),
                )
        finally:
            torch.set_rng_state(rng_state)


class ScaleDecoupledReassemblyProjectorV38P3(
    ScaleDecoupledReassemblyProjectorV38
):
    """P3 low-rank local reassembly residual plus the original C2f."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            use_p3_local_residual=True,
            use_p4_dynamic_routing=False,
            **kwargs,
        )


class ScaleDecoupledReassemblyProjectorV38P4(
    ScaleDecoupledReassemblyProjectorV38
):
    """P4 image-conditioned normalized routing plus the original C2f."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            use_p3_local_residual=False,
            use_p4_dynamic_routing=True,
            **kwargs,
        )


class ScaleDecoupledReassemblyProjectorV38Both(
    ScaleDecoupledReassemblyProjectorV38
):
    """Combine P3 local reassembly and P4 dynamic routing."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            use_p3_local_residual=True,
            use_p4_dynamic_routing=True,
            **kwargs,
        )
