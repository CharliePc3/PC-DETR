#!/usr/bin/env python3
"""Extract a raw DINOv3 encoder state dict from a detector checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


DEFAULT_PREFIX = "backbone.0.encoder.encoder."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    extracted = {
        name.removeprefix(args.prefix): value
        for name, value in state_dict.items()
        if name.startswith(args.prefix)
    }
    if not extracted:
        raise KeyError(f"No parameters found with prefix {args.prefix!r} in {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(extracted, args.output)
    print(f"Saved {len(extracted)} DINOv3 tensors to {args.output}")


if __name__ == "__main__":
    main()
