#!/usr/bin/env python3
"""Run a real DINOv3 forward/backward smoke test for the EoMT-style control."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("DINOV3_REPO_DIR", "/data/cpc/root/project/DINOv3")
os.environ.setdefault("DINOV3_WEIGHTS_DIR", str(PROJECT_ROOT / "weights" / "dinov3"))

from rfdetr.models.backbone.dinov3 import (  # noqa: E402
    SIZE_TO_MODEL,
    _load_dinov3_backbone_builder,
    _load_state_dict,
    _resolve_weights_path,
)
from rfdetr.models.eomt import EncoderOnlyMaskTransformer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("EoMT-style DINOv3 smoke test")
    parser.add_argument("--model-size", default="small", choices=tuple(SIZE_TO_MODEL))
    parser.add_argument("--pretrained-encoder", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--num-query-blocks", type=int, default=2)
    parser.add_argument("--mask-dim", type=int, default=128)
    parser.add_argument("--masked-attention", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = SIZE_TO_MODEL[args.model_size]
    builder = _load_dinov3_backbone_builder(model_name)
    encoder = builder(pretrained=False)
    weights_path = _resolve_weights_path(model_name, args.pretrained_encoder)
    encoder.load_state_dict(_load_state_dict(weights_path), strict=True)

    model = EncoderOnlyMaskTransformer(
        encoder,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        num_query_blocks=args.num_query_blocks,
        mask_dim=args.mask_dim,
        freeze_image_prefix=True,
        masked_attention=args.masked_attention,
    ).to(args.device)
    model.train()
    images = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=args.device,
    )
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast = torch.autocast("cpu", enabled=False)

    with autocast:
        output = model(images)
        synthetic_loss = (
            output["pred_logits"].float().square().mean()
            + output["pred_masks"].float().square().mean()
        )
    synthetic_loss.backward()

    result = {
        "model": model_name,
        "image_shape": list(images.shape),
        "pred_logits_shape": list(output["pred_logits"].shape),
        "pred_masks_shape": list(output["pred_masks"].shape),
        "aux_outputs": len(output["aux_outputs"]),
        "masked_attention": args.masked_attention,
        "masking_probabilities": model.masking_probabilities,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "query_gradient_norm": float(model.segmentation_queries.grad.float().norm()),
        "peak_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated() / 1024**2)
            if args.device.startswith("cuda")
            else None
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
