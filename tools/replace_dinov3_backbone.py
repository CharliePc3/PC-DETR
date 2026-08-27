#!/usr/bin/env python3
"""Replace the DINOv3 encoder inside an RF-DETR-DINOv3 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


DEFAULT_PREFIX = "backbone.0.encoder.encoder."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a dictionary checkpoint: {path}")
    for key in ("model", "state_dict", "teacher"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            return nested
    return checkpoint


def main() -> None:
    args = parse_args()
    detector = torch.load(args.detector, map_location="cpu", weights_only=False)
    model_state = detector.get("model")
    if not isinstance(model_state, dict):
        raise KeyError(f"Detector checkpoint has no model state: {args.detector}")
    backbone_state = load_state_dict(args.backbone)

    detector_keys = {
        name.removeprefix(args.prefix)
        for name in model_state
        if name.startswith(args.prefix)
    }
    missing = sorted(detector_keys - backbone_state.keys())
    unexpected = sorted(backbone_state.keys() - detector_keys)
    if missing or unexpected:
        raise KeyError(
            f"Backbone mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    for name, value in backbone_state.items():
        detector_name = args.prefix + name
        if model_state[detector_name].shape != value.shape:
            raise ValueError(
                f"Shape mismatch for {name}: {tuple(model_state[detector_name].shape)} "
                f"!= {tuple(value.shape)}"
            )
        model_state[detector_name] = value

    detector.pop("optimizer", None)
    detector.pop("lr_scheduler", None)
    detector["posthoc_backbone_source"] = str(args.backbone.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(detector, args.output)
    print(f"Replaced {len(backbone_state)} tensors and saved {args.output}")


if __name__ == "__main__":
    main()
