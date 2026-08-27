#!/usr/bin/env python3

"""Benchmark aligned MultiScaleProjector and efficient SDSR forward paths."""

from __future__ import annotations

import argparse

import torch

from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.backbone.semantic_reassembly_projector_v23 import (
    ScaleDecoupledReassemblyProjectorV23,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v34 import (
    ScaleDecoupledReassemblyProjectorV34,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v35 import (
    ScaleDecoupledReassemblyProjectorV35,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v36 import (
    ScaleDecoupledReassemblyProjectorV36,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    return parser.parse_args()


def benchmark(
    model: torch.nn.Module,
    features: list[torch.Tensor],
    warmup: int,
    iterations: int,
) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            model(features)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            model(features)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the latency benchmark")
    dtype = getattr(torch, args.dtype)
    features = [
        torch.randn(
            args.batch_size,
            384,
            args.grid_size,
            args.grid_size,
            device="cuda",
            dtype=dtype,
        )
        for _ in range(4)
    ]
    models = {
        "multiscale": MultiScaleProjector(
            [384] * 4,
            256,
            [2.0, 1.0, 0.5],
            layer_norm=True,
        ),
        "sdsr_v23": ScaleDecoupledReassemblyProjectorV23(
            [384] * 4,
            256,
            ["P3", "P4", "P5"],
        ),
        "sdsr_v34": ScaleDecoupledReassemblyProjectorV34(
            [384] * 4,
            256,
            ["P3", "P4", "P5"],
        ),
        "sdsr_v35": ScaleDecoupledReassemblyProjectorV35(
            [384] * 4,
            256,
            ["P3", "P4", "P5"],
        ),
        "sdsr_v36": ScaleDecoupledReassemblyProjectorV36(
            [384] * 4,
            256,
            ["P3", "P4", "P5"],
        ),
    }
    for name, model in models.items():
        model = model.to(device="cuda", dtype=dtype).eval()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        latency_ms = benchmark(model, features, args.warmup, args.iterations)
        print(f"{name}\tparameters={parameters}\tlatency_ms={latency_ms:.6f}")


if __name__ == "__main__":
    main()
