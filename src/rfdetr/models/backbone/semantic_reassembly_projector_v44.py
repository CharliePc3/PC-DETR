# ------------------------------------------------------------------------
# RF-DETR-DINOv3
# ------------------------------------------------------------------------

"""Alternative CSP fusion blocks for the SDSR-v40 scale branches."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn

from rfdetr.models.backbone.projector import Bottleneck, ConvX, LayerNorm
from rfdetr.models.backbone.semantic_reassembly_projector_v40 import (
    ScaleDecoupledReassemblyProjectorV40P4Learnable,
)


class C3k(nn.Module):
    """Compact C3 block with configurable spatial kernels."""

    def __init__(self, channels: int, num_blocks: int = 2):
        super().__init__()
        hidden_channels = max(1, channels // 2)
        self.cv1 = ConvX(
            channels, hidden_channels, kernel=1, layer_norm=True, act="silu"
        )
        self.cv2 = ConvX(
            channels, hidden_channels, kernel=1, layer_norm=True, act="silu"
        )
        self.blocks = nn.Sequential(
            *[
                Bottleneck(
                    hidden_channels,
                    hidden_channels,
                    shortcut=True,
                    k=(3, 3),
                    e=1.0,
                    act="silu",
                    layer_norm=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.cv3 = ConvX(
            2 * hidden_channels,
            channels,
            kernel=1,
            layer_norm=True,
            act="silu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.blocks(self.cv1(x)), self.cv2(x)), dim=1))


class C3k2Fusion(nn.Module):
    """C2f shell whose iterative path uses nested C3k blocks."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        self.hidden_channels = max(1, out_channels // 2)
        self.cv1 = ConvX(
            in_channels,
            2 * self.hidden_channels,
            kernel=1,
            layer_norm=True,
            act="silu",
        )
        self.blocks = nn.ModuleList(
            [C3k(self.hidden_channels, num_blocks=2) for _ in range(num_blocks)]
        )
        self.cv2 = ConvX(
            (2 + num_blocks) * self.hidden_channels,
            out_channels,
            kernel=1,
            layer_norm=True,
            act="silu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = list(
            self.cv1(x).split(
                (self.hidden_channels, self.hidden_channels), dim=1
            )
        )
        outputs.extend(block(outputs[-1]) for block in self.blocks)
        return self.cv2(torch.cat(outputs, dim=1))


class RepStyleBlock(nn.Module):
    """Parallel 3x3/1x1 residual block adapted to channel LayerNorm."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv3 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.norm3 = LayerNorm(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm1 = LayerNorm(channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.norm3(self.conv3(x)) + self.norm1(self.conv1(x)))


class RepC3Fusion(nn.Module):
    """Low-concatenation RepC3-style fusion with a half-width hidden path."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        hidden_channels = max(1, out_channels // 2)
        self.cv1 = ConvX(
            in_channels,
            hidden_channels,
            kernel=1,
            layer_norm=True,
            act="silu",
        )
        self.cv2 = ConvX(
            in_channels,
            hidden_channels,
            kernel=1,
            layer_norm=True,
            act="silu",
        )
        self.blocks = nn.Sequential(
            *[RepStyleBlock(hidden_channels) for _ in range(num_blocks)]
        )
        self.cv3 = ConvX(
            hidden_channels,
            out_channels,
            kernel=1,
            layer_norm=True,
            act="silu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(self.blocks(self.cv1(x)) + self.cv2(x))


def _replace_scale_fusions(
    projector: ScaleDecoupledReassemblyProjectorV40P4Learnable,
    in_channels: Sequence[int],
    out_channels: int,
    num_blocks: int,
    block_factory: Callable[[int, int, int], nn.Module],
) -> None:
    fused_channels = {
        "P3": sum(channels // 2 for channels in in_channels),
        "P4": sum(in_channels),
        "P5": 2 * sum(in_channels),
    }
    if "P3" in projector.branches:
        projector.branches["P3"].fusion = block_factory(
            fused_channels["P3"], out_channels, num_blocks
        )
    if "P4" in projector.branches:
        projector.branches["P4"].base.fusion = block_factory(
            fused_channels["P4"], out_channels, num_blocks
        )
    if "P5" in projector.branches:
        projector.branches["P5"].fusion = block_factory(
            fused_channels["P5"], out_channels, num_blocks
        )


class ScaleDecoupledReassemblyProjectorV44(
    ScaleDecoupledReassemblyProjectorV40P4Learnable
):
    """Replace all three v40 C2f fusion blocks with one alternative family."""

    fusion_factory: Callable[[int, int, int], nn.Module]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_channels = tuple(args[0] if args else kwargs["in_channels"])
        out_channels = int(args[1] if len(args) > 1 else kwargs["out_channels"])
        num_blocks = int(kwargs.get("num_blocks", 3))
        rng_state = torch.get_rng_state()
        try:
            _replace_scale_fusions(
                self,
                in_channels=in_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                block_factory=self.fusion_factory,
            )
        finally:
            torch.set_rng_state(rng_state)


class ScaleDecoupledReassemblyProjectorV44RepC3(
    ScaleDecoupledReassemblyProjectorV44
):
    fusion_factory = RepC3Fusion


class ScaleDecoupledReassemblyProjectorV44C3k2(
    ScaleDecoupledReassemblyProjectorV44
):
    fusion_factory = C3k2Fusion
