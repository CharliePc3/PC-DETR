import argparse
import json
import os
import sys
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("DINOV3_REPO_DIR", "/data/cpc/root/project/DINOv3")
os.environ.setdefault("DINOV3_WEIGHTS_DIR", str(PROJECT_ROOT / "weights" / "dinov3"))

from rfdetr.util.benchmark import flop_count, get_shape
from rfdetr_dinov3 import RFDETRDINOv3


MIB = 1024**2


def sdpa_flop_jit(inputs, outputs):
    query_shape = get_shape(inputs[0])
    key_shape = get_shape(inputs[1])
    value_shape = get_shape(inputs[2])
    if len(query_shape) < 3 or len(key_shape) != len(query_shape) or len(value_shape) != len(query_shape):
        raise ValueError(
            "scaled_dot_product_attention expects query/key/value tensors with matching ranks, "
            f"got {query_shape}, {key_shape}, and {value_shape}."
        )

    batch_heads = int(np.prod(query_shape[:-2]))
    query_tokens, query_dim = query_shape[-2:]
    key_tokens = key_shape[-2]
    value_dim = value_shape[-1]
    qk_flops = batch_heads * query_tokens * key_tokens * query_dim
    av_flops = batch_heads * query_tokens * key_tokens * value_dim
    return Counter({"scaled_dot_product_attention": qk_flops + av_flops})


def parse_args():
    parser = argparse.ArgumentParser("Benchmark an RF-DETR-DINOv3 detector checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def checkpoint_arg(checkpoint_args, name, default):
    if checkpoint_args is None:
        return default
    if isinstance(checkpoint_args, dict):
        return checkpoint_args.get(name, default)
    return getattr(checkpoint_args, name, default)


def build_model(checkpoint):
    saved_args = checkpoint.get("args")
    config = {
        "encoder": checkpoint_arg(saved_args, "encoder", "dinov3_small"),
        "resolution": checkpoint_arg(saved_args, "resolution", 576),
        "dec_layers": checkpoint_arg(saved_args, "dec_layers", 4),
        "num_queries": checkpoint_arg(saved_args, "num_queries", 300),
        "num_select": checkpoint_arg(saved_args, "num_select", 300),
        "group_detr": checkpoint_arg(saved_args, "group_detr", 13),
        "out_feature_indexes": checkpoint_arg(saved_args, "out_feature_indexes", [2, 5, 8, 11]),
        "projector_scale": checkpoint_arg(saved_args, "projector_scale", ["P4"]),
        "positional_encoding_size": checkpoint_arg(saved_args, "positional_encoding_size", 36),
        "use_cdn": checkpoint_arg(saved_args, "use_cdn", False),
        "dn_number": checkpoint_arg(saved_args, "dn_number", 50),
        "dn_label_noise_scale": checkpoint_arg(saved_args, "dn_label_noise_scale", 0.5),
        "dn_box_noise_scale": checkpoint_arg(saved_args, "dn_box_noise_scale", 0.6),
        "dn_negative": checkpoint_arg(saved_args, "dn_negative", True),
        "register_border_tokens": checkpoint_arg(saved_args, "register_border_tokens", 0),
        "register_fill": checkpoint_arg(saved_args, "register_fill", "randn"),
        "register_noise_std": checkpoint_arg(saved_args, "register_noise_std", 1.0),
        "feature_adapter": checkpoint_arg(saved_args, "feature_adapter", "none"),
        "feature_adapter_init_scale": checkpoint_arg(saved_args, "feature_adapter_init_scale", 1.0),
    }
    detector = RFDETRDINOv3(pretrain_weights=None, **config)
    model = detector.model.model
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    return model, config


def latency_and_memory(model, inputs, device, warmup, repeats, amp):
    context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp
        else nullcontext()
    )

    with torch.inference_mode():
        for _ in range(warmup):
            with context():
                outputs = model(inputs)
        del outputs
        torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        memory_before = torch.cuda.memory_allocated(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        for start, end in zip(starts, ends):
            start.record()
            with context():
                outputs = model(inputs)
            end.record()
        torch.cuda.synchronize(device)
        latencies = np.asarray([start.elapsed_time(end) for start, end in zip(starts, ends)])
        peak_memory = torch.cuda.max_memory_allocated(device)
        del outputs

    return {
        "mean_ms": float(latencies.mean()),
        "std_ms": float(latencies.std()),
        "min_ms": float(latencies.min()),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "max_ms": float(latencies.max()),
        "fps": float(1000.0 / latencies.mean()),
        "memory_before_mib": float(memory_before / MIB),
        "peak_memory_mib": float(peak_memory / MIB),
        "incremental_peak_memory_mib": float((peak_memory - memory_before) / MIB),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("warmup and repeats must be positive.")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, config = build_model(checkpoint)
    model.eval().to(device)

    resolution = int(config["resolution"])
    image = torch.randn(3, resolution, resolution, device=device)
    inputs = [image]
    parameters = sum(parameter.numel() for parameter in model.parameters())

    with torch.inference_mode():
        detailed_flops = dict(
            flop_count(
                model,
                (inputs,),
                customized_ops={"aten::scaled_dot_product_attention": sdpa_flop_jit},
            )
        )

    sdpa_gflops = float(detailed_flops.get("scaled_dot_product_attention", 0.0))
    total_gflops = float(sum(detailed_flops.values()))

    results = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "config": config,
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "batch_size": 1,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "pytorch_mode": "eager",
            "postprocess_included": False,
            "tf32_enabled": True,
        },
        "parameters": parameters,
        "gflops": total_gflops,
        "repo_supported_gflops_without_sdpa": total_gflops - sdpa_gflops,
        "sdpa_gflops": sdpa_gflops,
        "flop_convention": "One multiply-add is counted as one FLOP; SDPA QK^T and attention-V are included; grid_sampler remains ignored as in the RF-DETR utility.",
        "detailed_gflops": detailed_flops,
        "fp32_tf32": latency_and_memory(model, inputs, device, args.warmup, args.repeats, amp=False),
        "amp_fp16": latency_and_memory(model, inputs, device, args.warmup, args.repeats, amp=True),
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
