"""Benchmark detector forwards with RF-DETR's 200 ms latency buffer.

This is a local PyTorch comparison, not a reproduction of the paper's
TensorRT/T4 numbers. Each model is loaded by its native environment and emits
the same JSON schema so the results can be compared without mixing timing
protocols.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=("rfdetr-dinov3", "deimv2", "rfdetr"), required=True
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Required for DEIMv2.")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--buffer-ms", type=float, default=200.0)
    parser.add_argument("--precision", choices=("both", "fp32", "fp16"), default="both")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_rfdetr_dinov3(checkpoint_path: Path, resolution: int | None):
    src = ROOT / "src"
    sys.path.insert(0, str(src))
    os.environ.setdefault("DINOV3_REPO_DIR", "/data/cpc/root/project/DINOv3")
    os.environ.setdefault("DINOV3_WEIGHTS_DIR", str(ROOT / "weights" / "dinov3"))

    from benchmark_detector import build_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, config = build_model(checkpoint)
    native_resolution = int(config["resolution"])
    if resolution is not None and resolution != native_resolution:
        raise ValueError(
            "RF-DETR-DINOv3 checkpoints must be rebuilt with the matching saved "
            f"resolution; requested {resolution}, checkpoint uses {native_resolution}."
        )
    model.eval()
    return model, native_resolution, {"checkpoint_config": config, "weights": "model"}


def load_rfdetr(checkpoint_path: Path, resolution: int | None):
    from rfdetr import RFDETRMedium

    native_resolution = 576
    input_resolution = native_resolution if resolution is None else resolution
    detector = RFDETRMedium(
        pretrain_weights=str(checkpoint_path),
        device="cpu",
    )
    model = detector.model.model.eval()
    return model, input_resolution, {
        "variant": "RF-DETR Medium",
        "weights": "official checkpoint",
        "native_resolution": native_resolution,
        "input_resolution_override": input_resolution != native_resolution,
    }


def load_deimv2(checkpoint_path: Path, config_path: Path | None, resolution: int | None):
    if config_path is None:
        raise ValueError("--config is required for DEIMv2.")
    deim_root = config_path.resolve().parents[2]
    sys.path.insert(0, str(deim_root))

    from engine.core import YAMLConfig

    cfg = YAMLConfig(str(config_path))
    model = cfg.model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("ema") and checkpoint["ema"].get("module"):
        state_dict = checkpoint["ema"]["module"]
        weight_source = "ema.module"
    else:
        state_dict = checkpoint["model"]
        weight_source = "model"
    model.load_state_dict(state_dict, strict=True)
    model = model.deploy()

    native_resolution = int(cfg.yaml_cfg.get("eval_spatial_size", [640, 640])[0])
    resolution = native_resolution if resolution is None else resolution
    if resolution != native_resolution:
        raise ValueError(
            f"This DEIMv2 deploy config is fixed at {native_resolution}; got {resolution}."
        )
    return model, resolution, {
        "config": str(config_path.resolve()),
        "weights": weight_source,
    }


def load_model(args: argparse.Namespace):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    if args.model == "rfdetr-dinov3":
        loaded = load_rfdetr_dinov3(checkpoint_path, args.resolution)
    elif args.model == "deimv2":
        loaded = load_deimv2(checkpoint_path, config_path, args.resolution)
    else:
        loaded = load_rfdetr(checkpoint_path, args.resolution)
    model, resolution, metadata = loaded
    metadata["checkpoint"] = str(checkpoint_path)
    return model, resolution, metadata


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def buffered_latency(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
    buffer_ms: float,
    amp: bool,
) -> dict[str, float]:
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp
        else nullcontext()
    )
    buffer_seconds = buffer_ms / 1000.0

    with torch.inference_mode():
        for _ in range(warmup):
            with autocast():
                model(image)
            synchronize(device)
            time.sleep(buffer_seconds)

        latencies = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast():
                model(image)
            end.record()
            end.synchronize()
            latencies.append(start.elapsed_time(end))
            time.sleep(buffer_seconds)

    values = np.asarray(latencies, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std()),
        "min_ms": float(values.min()),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "max_ms": float(values.max()),
    }


def count_flops(model: torch.nn.Module, image: torch.Tensor) -> tuple[int, dict]:
    counter = FlopCounterMode(display=False)
    # ModuleTracker's hooks require an ordinary autograd graph on PyTorch 2.11.
    # The graph is discarded immediately after this single tracing forward.
    with counter:
        model(image)
    raw_counts = counter.get_flop_counts()
    serializable_counts = {
        str(module): {str(operation): int(value) for operation, value in operations.items()}
        for module, operations in raw_counts.items()
    }
    return counter.get_total_flops(), serializable_counts


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.warmup < 1 or args.repeats < 1 or args.buffer_ms < 0:
        raise ValueError("warmup/repeats must be positive and buffer-ms non-negative.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model, resolution, model_metadata = load_model(args)
    model.to(device).eval()
    image = torch.randn(1, 3, resolution, resolution, device=device)
    parameters = sum(parameter.numel() for parameter in model.parameters())

    total_flops, flop_counts = count_flops(model, image)
    synchronize(device)

    timing = {}
    if args.precision in ("both", "fp32"):
        timing["fp32_tf32"] = buffered_latency(
            model, image, device, args.warmup, args.repeats, args.buffer_ms, amp=False
        )
    if args.precision in ("both", "fp16"):
        timing["amp_fp16"] = buffered_latency(
            model, image, device, args.warmup, args.repeats, args.buffer_ms, amp=True
        )

    result = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "model": args.model,
        "model_metadata": model_metadata,
        "parameters": parameters,
        "resolution": resolution,
        "gflops": total_flops / 1e9,
        "gflops_fma_as_one": total_flops / 2e9,
        "flop_convention": (
            "gflops uses torch.utils.flop_counter with a fused multiply-add counted "
            "as two FLOPs; gflops_fma_as_one divides that result by two to match "
            "the convention used by the local RF-DETR benchmark utility"
        ),
        "flop_counts": flop_counts,
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "batch_size": 1,
            "mode": "PyTorch eager",
            "postprocess_included": False,
            "buffer_ms_between_forwards": args.buffer_ms,
            "buffer_included_in_latency": False,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "timing": timing,
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
