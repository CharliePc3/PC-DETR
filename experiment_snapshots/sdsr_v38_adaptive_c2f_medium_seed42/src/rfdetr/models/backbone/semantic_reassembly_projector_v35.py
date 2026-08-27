# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Single-kernel two-basis P5 sampling for efficient semantic reassembly."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from rfdetr.models.backbone.semantic_reassembly_projector_v34 import (
    PackedSeparableTwoBasisP5Fusion,
    ScaleDecoupledReassemblyProjectorV34,
)


class FusedKernelTwoBasisP5Fusion(PackedSeparableTwoBasisP5Fusion):
    """Evaluate fixed low-pass and learned detail bases in one grouped kernel."""

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.primary is not None:
            return super().forward(features)
        if len(features) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {len(features)}"
            )

        packed = torch.cat(features, dim=1)
        primary_kernel = self.blur_kernel.to(dtype=packed.dtype).expand(
            self.total_channels, 1, -1, -1
        )
        paired_kernel = torch.stack(
            (primary_kernel, self.detail.weight), dim=1
        ).reshape(2 * self.total_channels, 1, 3, 3)
        sampled = F.conv2d(
            packed,
            paired_kernel,
            stride=2,
            padding=1,
            groups=self.total_channels,
        )

        batch_size, _, height, width = sampled.shape
        sampled = sampled.reshape(
            batch_size, self.total_channels, 2, height, width
        )
        primary = sampled[:, :, 0].reshape(
            batch_size, self.num_features, self.channels, height, width
        )
        detail = sampled[:, :, 1].reshape(
            batch_size, self.num_features, self.channels, height, width
        )
        deep_detail = self.semantic_mixer(
            detail[:, self.grouped_feature_index].contiguous()
        )
        detail_parts = list(detail.unbind(dim=1))
        detail_parts[self.grouped_feature_index] = deep_detail
        detail = torch.stack(detail_parts, dim=1)

        primary = self._featurewise_norm(
            primary, self.primary_weight, self.primary_bias
        )
        detail = self._featurewise_norm(
            detail, self.detail_weight, self.detail_bias
        )
        detail = torch.tanh(self.detail_scale).to(detail.dtype) * detail
        sampled = torch.stack((primary, detail), dim=2).reshape(
            batch_size,
            2 * self.total_channels,
            height,
            width,
        )
        return self.output_norm(self.fusion(sampled))


class ScaleDecoupledReassemblyProjectorV35(
    ScaleDecoupledReassemblyProjectorV34
):
    """Use the v34 representation with one fused P5 sampling kernel."""

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
        detail_groups: int = 16,
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
            detail_groups=detail_groups,
        )
        if "P5" in self.levels:
            source = self.branches["P5"]
            with torch.random.fork_rng(devices=[]):
                fused = FusedKernelTwoBasisP5Fusion(
                    in_channels,
                    out_channels,
                    num_blocks=num_blocks,
                    anti_alias=use_phase_downsample,
                    maximum_groups=detail_groups,
                    grouped_feature_index=-1,
                )
            fused.load_state_dict(source.state_dict())
            self.branches["P5"] = fused
