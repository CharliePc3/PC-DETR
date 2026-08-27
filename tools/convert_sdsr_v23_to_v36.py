#!/usr/bin/env python3

"""Convert an SDSR-v23 checkpoint to the functionally equivalent SDSR-v36."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from rfdetr.models.backbone.semantic_reassembly_projector_v23 import (
    ScaleDecoupledReassemblyProjectorV23,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v36 import (
    ScaleDecoupledReassemblyProjectorV36,
)


PROJECTOR_PREFIX = "backbone.0.projector."
P5_PREFIX = PROJECTOR_PREFIX + "branches.P5."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def stack_layer_values(
    state: dict[str, torch.Tensor], suffix: str
) -> torch.Tensor:
    values = [
        state[f"{P5_PREFIX}downsampling.{index}.{suffix}"]
        for index in range(4)
    ]
    return torch.stack(values, dim=0)


def convert_model_state(
    source: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    state = copy.copy(source)
    detail_weights = [
        source[f"{P5_PREFIX}downsampling.{index}.detail.weight"]
        for index in range(4)
    ]
    state[f"{P5_PREFIX}shallow_detail.weight"] = torch.cat(
        detail_weights[:3], dim=0
    )
    state[f"{P5_PREFIX}deep_detail.weight"] = detail_weights[3]
    state[f"{P5_PREFIX}primary_weight"] = stack_layer_values(
        source, "primary_norm.weight"
    )
    state[f"{P5_PREFIX}primary_bias"] = stack_layer_values(
        source, "primary_norm.bias"
    )
    state[f"{P5_PREFIX}detail_weight"] = stack_layer_values(
        source, "detail_norm.weight"
    )
    state[f"{P5_PREFIX}detail_bias"] = stack_layer_values(
        source, "detail_norm.bias"
    )
    scales = [
        source[f"{P5_PREFIX}downsampling.{index}.detail_scale"]
        for index in range(4)
    ]
    state[f"{P5_PREFIX}detail_scale"] = torch.stack(scales, dim=1)

    for key in list(state):
        if key.startswith(f"{P5_PREFIX}downsampling."):
            del state[key]
    return state


def projector_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix(PROJECTOR_PREFIX): value
        for key, value in state.items()
        if key.startswith(PROJECTOR_PREFIX)
    }


def verify_conversion(
    source: dict[str, torch.Tensor],
    converted: dict[str, torch.Tensor],
) -> float:
    kwargs = {
        "in_channels": [384] * 4,
        "out_channels": 256,
        "levels": ["P3", "P4", "P5"],
    }
    v23 = ScaleDecoupledReassemblyProjectorV23(**kwargs).eval()
    v36 = ScaleDecoupledReassemblyProjectorV36(**kwargs).eval()
    v23.load_state_dict(projector_state(source), strict=True)
    v36.load_state_dict(projector_state(converted), strict=True)
    generator = torch.Generator().manual_seed(2026)
    features = [torch.randn(1, 384, 7, 9, generator=generator) for _ in range(4)]
    with torch.inference_mode():
        expected = v23(features)
        actual = v36(features)
    maximum_difference = max(
        float((left - right).abs().max())
        for left, right in zip(expected, actual)
    )
    if maximum_difference > 2.0e-5:
        raise RuntimeError(
            f"Converted projector differs from v23 by {maximum_difference:.6g}"
        )
    return maximum_difference


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    source_type = getattr(checkpoint.get("args"), "projector_type", None)
    if source_type != "sdsr_v23":
        raise ValueError(f"Expected an sdsr_v23 checkpoint, got {source_type!r}")

    source = checkpoint["model"]
    converted = convert_model_state(source)
    maximum_difference = None
    if not args.skip_verify:
        maximum_difference = verify_conversion(source, converted)

    checkpoint["model"] = converted
    checkpoint["args"].projector_type = "sdsr_v36"
    checkpoint.pop("optimizer", None)
    checkpoint.pop("lr_scheduler", None)
    checkpoint["conversion"] = {
        "source_projector_type": "sdsr_v23",
        "target_projector_type": "sdsr_v36",
        "functionally_equivalent": True,
        "maximum_verification_difference": maximum_difference,
        "training_state_removed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(
        f"Converted {args.input} -> {args.output}; "
        f"max_diff={maximum_difference}"
    )


if __name__ == "__main__":
    main()
