# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Scale-decoupled semantic reassembly projector.

The projector turns the single-resolution intermediate features of a ViT into
detector-ready P3/P4/P5 maps.  It keeps scale construction and semantic fusion
separate: intermediate layers are routed per output channel, P3 is rebuilt by
content-aware local reassembly, and P5 is produced by anti-aliased,
phase-adaptive downsampling.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.backbone.projector import LayerNorm


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unable to create GroupNorm for {channels} channels")


class LayerChannelRouter(nn.Module):
    """Learn separate, channel-wise mixtures of ViT layers for P3 and P4."""

    def __init__(self, in_channels: Sequence[int], out_channels: int):
        super().__init__()
        if not in_channels:
            raise ValueError("in_channels must contain at least one feature")

        self.norms = nn.ModuleList([LayerNorm(channels) for channels in in_channels])
        if len(set(in_channels)) == 1:
            self.shared_projection = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
            self.projections = None
        else:
            self.shared_projection = None
            self.projections = nn.ModuleList(
                [nn.Conv2d(channels, out_channels, kernel_size=1) for channels in in_channels]
            )

        # Uniform initialization lets the module start as a neutral layer fusion.
        self.route_logits = nn.Parameter(torch.zeros(2, len(in_channels), out_channels))

    def forward(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(features) != len(self.norms):
            raise ValueError(f"Expected {len(self.norms)} features, got {len(features)}")

        target_size = features[-1].shape[-2:]
        projected = []
        for index, (feature, norm) in enumerate(zip(features, self.norms)):
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            feature = norm(feature)
            if self.shared_projection is not None:
                projected.append(self.shared_projection(feature))
            else:
                projected.append(self.projections[index](feature))

        stacked = torch.stack(projected, dim=1)
        weights = self.route_logits.float().softmax(dim=1).to(stacked.dtype)
        routed = torch.einsum("blchw,slc->bschw", stacked, weights)
        return routed[:, 0], routed[:, 1]


class DirectionalDetailStem(nn.Module):
    """Extract a cheap stride-8 directional guide directly from the image."""

    def __init__(self, out_channels: int):
        super().__init__()
        if out_channels < 8 or out_channels % 4 != 0:
            raise ValueError("detail_channels must be at least 8 and divisible by 4")

        hidden1 = max(out_channels // 2, 8)
        hidden2 = max(3 * out_channels // 4, 8)
        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden1, kernel_size=3, stride=2, padding=1, bias=False),
            _group_norm(hidden1),
            nn.GELU(),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, stride=2, padding=1, bias=False),
            _group_norm(hidden2),
            nn.GELU(),
            nn.Conv2d(hidden2, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            _group_norm(out_channels),
            nn.GELU(),
        )

        branch_channels = out_channels // 4
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(out_channels, branch_channels, kernel_size=1, bias=False),
                nn.Conv2d(out_channels, branch_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
                nn.Conv2d(out_channels, branch_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
                nn.Conv2d(out_channels, branch_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            ]
        )
        self.fuse = nn.Sequential(
            _group_norm(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            image = image.masked_fill(mask[:, None], 0)
        base = self.stem(image)
        directional = torch.cat([branch(base) for branch in self.branches], dim=1)
        return base + self.fuse(directional)


def _bilinear_phase_prior(kernel_size: int) -> torch.Tensor:
    if kernel_size != 3:
        raise ValueError("The current semantic reassembly implementation expects a 3x3 kernel")

    prior = torch.full((4, kernel_size * kernel_size), -20.0)
    phase_axes = (
        ((-1, 0.25), (0, 0.75)),
        ((0, 0.75), (1, 0.25)),
    )
    phase = 0
    for y_choices in phase_axes:
        for x_choices in phase_axes:
            for y_offset, y_weight in y_choices:
                for x_offset, x_weight in x_choices:
                    index = (y_offset + 1) * kernel_size + x_offset + 1
                    prior[phase, index] = math.log(y_weight * x_weight)
            phase += 1
    return prior


class SemanticLocalReassembly(nn.Module):
    """Reassemble each P4 token into four P3 children using local semantic weights."""

    def __init__(self, channels: int, context_channels: int, use_directional_guide: bool):
        super().__init__()
        self.kernel_size = 3
        self.use_directional_guide = use_directional_guide
        self.context_projection = nn.Sequential(
            nn.Conv2d(channels, context_channels, kernel_size=1, bias=False),
            _group_norm(context_channels),
            nn.GELU(),
        )
        predictor_channels = context_channels * (2 if use_directional_guide else 1)
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(predictor_channels, context_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(context_channels),
            nn.GELU(),
            nn.Conv2d(context_channels, self.kernel_size**2, kernel_size=1),
        )
        # A tiny non-zero initialization preserves the bilinear prior while allowing
        # gradients to reach the semantic and directional context paths on step one.
        nn.init.normal_(self.weight_predictor[-1].weight, std=1.0e-3)
        nn.init.zeros_(self.weight_predictor[-1].bias)

        self.phase_prior = nn.Parameter(_bilinear_phase_prior(self.kernel_size))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        source: torch.Tensor,
        detail_guide: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, channels, height, width = source.shape
        output_size = (height * 2, width * 2)
        context = F.interpolate(
            self.context_projection(source), size=output_size, mode="bilinear", align_corners=False
        )
        if self.use_directional_guide:
            if detail_guide is None:
                raise ValueError("Directional detail guide is enabled but no guide was provided")
            detail_guide = F.interpolate(detail_guide, size=output_size, mode="bilinear", align_corners=False)
            context = torch.cat((context, detail_guide), dim=1)

        high_resolution_logits = self.weight_predictor(context)
        logits = F.pixel_unshuffle(high_resolution_logits, downscale_factor=2)
        logits = logits.reshape(batch, self.kernel_size**2, 4, height, width).permute(0, 2, 1, 3, 4)
        logits = logits + self.phase_prior[None, :, :, None, None].to(logits.dtype)

        padding = self.kernel_size // 2
        padded_source = F.pad(source, (padding, padding, padding, padding), mode="replicate")
        patches = F.unfold(padded_source, kernel_size=self.kernel_size)
        patches = patches.reshape(batch, channels, self.kernel_size**2, height, width)

        if mask is not None:
            low_mask = F.interpolate(mask[:, None].float(), size=(height, width), mode="nearest").to(torch.bool)
            padded_mask = F.pad(low_mask.float(), (padding, padding, padding, padding), value=1.0)
            invalid = F.unfold(padded_mask, kernel_size=self.kernel_size)
            invalid = invalid.reshape(batch, 1, self.kernel_size**2, height, width).to(torch.bool)
            logits = logits.masked_fill(invalid, -1.0e4)
        else:
            low_mask = None

        weights = logits.float().softmax(dim=2).to(source.dtype)
        children = torch.einsum("bckhw,bpkhw->bcphw", patches, weights)
        children = children.reshape(batch, channels * 4, height, width)
        local = F.pixel_shuffle(children, upscale_factor=2)
        bilinear = F.interpolate(source, size=output_size, mode="bilinear", align_corners=False)
        output = bilinear + torch.tanh(self.residual_scale) * (local - bilinear)

        if low_mask is not None:
            high_mask = F.interpolate(low_mask.float(), size=output_size, mode="nearest").to(torch.bool)
            output = output.masked_fill(high_mask, 0)
        return output


class LightRefineBlock(nn.Module):
    """A small residual spatial/channel refinement block."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = LayerNorm(channels)
        self.in_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.activation = nn.GELU()
        self.out_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), 1.0e-3))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        residual = self.in_projection(self.norm(feature))
        residual = self.out_projection(self.activation(self.depthwise(residual)))
        return feature + self.layer_scale * residual


class StridedProjection(nn.Module):
    """Cheap P5 fallback used by the phase-downsampling ablation."""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, groups=channels)
        self.norm = LayerNorm(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.norm(self.depthwise(feature)))


class AdaptivePhaseDownsample(nn.Module):
    """Fuse anti-aliased low-pass sampling with learned 2x2 phase selection."""

    def __init__(self, channels: int):
        super().__init__()
        self.phase_logits = nn.Conv2d(channels * 4, channels * 4, kernel_size=1, groups=channels)
        nn.init.zeros_(self.phase_logits.weight)
        nn.init.zeros_(self.phase_logits.bias)
        self.gate = nn.Conv2d(1, 1, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

        kernel_1d = torch.tensor([1.0, 2.0, 1.0])
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        self.register_buffer("blur_kernel", kernel_2d / kernel_2d.sum(), persistent=False)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        pad_height = height % 2
        pad_width = width % 2
        if pad_height or pad_width:
            feature = F.pad(feature, (0, pad_width, 0, pad_height), mode="replicate")

        kernel = self.blur_kernel.to(dtype=feature.dtype)
        kernel = kernel[None, None].expand(channels, 1, -1, -1)
        blur_input = F.pad(feature, (1, 1, 1, 1), mode="replicate")
        low_pass = F.conv2d(blur_input, kernel, groups=channels)[:, :, ::2, ::2]

        phases_flat = F.pixel_unshuffle(feature, downscale_factor=2)
        phase_weights = self.phase_logits(phases_flat)
        phase_weights = phase_weights.reshape(batch, channels, 4, *phases_flat.shape[-2:])
        phase_weights = phase_weights.float().softmax(dim=2).to(feature.dtype)
        phases = phases_flat.reshape(batch, channels, 4, *phases_flat.shape[-2:])
        phase_sample = (phases * phase_weights).sum(dim=2)

        local_average = F.avg_pool2d(F.pad(feature, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
        high_frequency = (feature - local_average).abs()
        high_frequency = F.avg_pool2d(high_frequency.mean(dim=1, keepdim=True), kernel_size=2, stride=2)
        gate = torch.sigmoid(self.gate(high_frequency))
        return low_pass * (1.0 - gate) + phase_sample * gate


class ScaleDecoupledReassemblyProjector(nn.Module):
    """Build P3/P4/P5 with scale-specific operators and shared ViT semantics."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        levels: Sequence[str],
        detail_channels: int = 32,
        use_local_reassembly: bool = True,
        use_directional_guide: bool = True,
        use_phase_downsample: bool = True,
    ):
        super().__init__()
        unsupported = set(levels) - {"P3", "P4", "P5"}
        if unsupported:
            raise ValueError(f"SDSR Projector does not support levels: {sorted(unsupported)}")
        if not levels:
            raise ValueError("levels must contain at least one pyramid level")

        self.levels = list(levels)
        self.use_local_reassembly = use_local_reassembly and "P3" in self.levels
        self.use_directional_guide = use_directional_guide and self.use_local_reassembly
        self.use_phase_downsample = use_phase_downsample

        self.router = LayerChannelRouter(in_channels, out_channels)
        self.detail_stem = DirectionalDetailStem(detail_channels) if self.use_directional_guide else None
        self.local_reassembly = (
            SemanticLocalReassembly(out_channels, detail_channels, self.use_directional_guide)
            if self.use_local_reassembly
            else None
        )
        if "P5" in self.levels:
            self.p5_downsample = (
                AdaptivePhaseDownsample(out_channels) if use_phase_downsample else StridedProjection(out_channels)
            )
        else:
            self.p5_downsample = None
        self.refine = nn.ModuleDict({level: LightRefineBlock(out_channels) for level in self.levels})

    def forward(
        self,
        features: Sequence[torch.Tensor],
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        p3_source, p4 = self.router(features)
        outputs = {}

        if "P3" in self.levels:
            if self.local_reassembly is not None:
                guide = None
                if self.detail_stem is not None:
                    if image is None:
                        raise ValueError("SDSR directional guide requires the input image")
                    guide = self.detail_stem(image, mask)
                p3 = self.local_reassembly(p3_source, guide, mask)
            else:
                p3 = F.interpolate(p3_source, scale_factor=2.0, mode="bilinear", align_corners=False)
            outputs["P3"] = self.refine["P3"](p3)

        if "P4" in self.levels:
            outputs["P4"] = self.refine["P4"](p4)

        if "P5" in self.levels:
            p5 = self.p5_downsample(p4)
            outputs["P5"] = self.refine["P5"](p5)

        return [outputs[level] for level in self.levels]
