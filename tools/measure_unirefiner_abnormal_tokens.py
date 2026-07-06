#!/usr/bin/env python3
"""Measure UniRefiner-style abnormal token ratios for DINOv3 variants.

The script reports FP, GP, AH, and union abnormal-token ratios. It counts only
image-region patch tokens in the denominator. Optional register borders can
participate in backbone self-attention but are excluded from the statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import analyze_dinov3_tokens as diag


UNIREFINER_ROOT = Path("/data/cpc/root/project/UniRefiner")
if str(UNIREFINER_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIREFINER_ROOT))

from unirefiner.spurious_filtering.attention_hijack import analyze_attention_hijacking  # noqa: E402
from unirefiner.spurious_filtering.fp_gp import analyze_fp_gp_similarity  # noqa: E402
from unirefiner.method.crops import compose_crop_with_background, roi_align_feature_map, sample_random_crop_boxes  # noqa: E402


def parse_variant(value: str) -> tuple[str, str | None]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Variant must be formatted as label:path, use label: for base weights.")
    label, path = value.split(":", 1)
    if not label:
        raise argparse.ArgumentTypeError("Variant label cannot be empty.")
    return label, path or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Measure UniRefiner-style abnormal token ratios")
    parser.add_argument("--data-root", default="/data/cpc/root/dataset/COCO")
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--ann-file", default=None)
    parser.add_argument("--image-root", default=None, help="Recursive image folder. Overrides COCO record loading.")
    parser.add_argument(
        "--background-image",
        default="/data/cpc/root/project/UniRefiner/assets/backgrounds/fixed_reference.png",
    )
    parser.add_argument("--output-dir", default="output/token_analysis/abnormal_ratio_compare")
    parser.add_argument("--encoder", default="dinov3_small", choices=tuple(f"dinov3_{k}" for k in diag.SIZE_TO_MODEL))
    parser.add_argument("--variant", action="append", type=parse_variant, required=True)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--visualize", type=int, default=12)
    parser.add_argument(
        "--paper-style",
        action="store_true",
        help="Use UniRefiner crop/composite foreground-token filtering instead of full-image filtering.",
    )
    parser.add_argument("--num-proposals", type=int, default=3)
    parser.add_argument("--crop-scale-min", type=float, default=0.25)
    parser.add_argument("--crop-scale-max", type=float, default=0.5)
    parser.add_argument("--crop-bg-ratio", type=float, default=0.35)
    parser.add_argument("--register-border", type=int, default=0)
    parser.add_argument("--register-fill", choices=("zero", "rand", "randn"), default="zero")
    parser.add_argument("--fp-gp-sigma", type=float, default=1.0)
    parser.add_argument("--gp-exclude-radius", type=int, default=1)
    parser.add_argument("--disable-ah", action="store_true")
    parser.add_argument("--ah-layer-start", type=int, default=10)
    parser.add_argument("--ah-layer-end", type=int, default=-1, help="-1 means the last transformer block.")
    parser.add_argument("--ah-sigma", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


class PlainImageRecord:
    def __init__(self, image_id: int, file_name: str, path: Path):
        self.image_id = image_id
        self.file_name = file_name
        self.path = path


def load_plain_image_records(image_root: Path) -> list[PlainImageRecord]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(path for path in image_root.rglob("*") if path.suffix.lower() in extensions)
    if not paths:
        raise RuntimeError(f"No images found under {image_root}")
    return [PlainImageRecord(idx, str(path.relative_to(image_root)), path) for idx, path in enumerate(paths)]


def build_model(encoder: str, pretrained_encoder: str | None, device: torch.device) -> torch.nn.Module:
    model_args = argparse.Namespace(encoder=encoder, pretrained_encoder=pretrained_encoder)
    return diag.build_dinov3_encoder(model_args).to(device)


def registered_image_token_mask(
    image_size: int,
    patch_size: int,
    register_border: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, int, int, int]:
    image_h = image_size // patch_size
    image_w = image_size // patch_size
    full_h = image_h + 2 * register_border
    full_w = image_w + 2 * register_border
    mask = torch.zeros((full_h, full_w), dtype=torch.bool, device=device)
    mask[register_border : register_border + image_h, register_border : register_border + image_w] = True
    return mask.reshape(-1), image_h, image_w, full_h, full_w


def compute_gp_clean_mask(
    image_tokens: torch.Tensor,
    grid_h: int,
    grid_w: int,
    sigma: float,
    exclude_radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = F.normalize(image_tokens.float(), dim=-1)
    similarity = tokens @ tokens.transpose(0, 1)
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(grid_h, device=tokens.device),
            torch.arange(grid_w, device=tokens.device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    distance = (coords[:, None, :] - coords[None, :, :]).abs().amax(dim=-1)
    similarity = similarity.masked_fill(distance <= exclude_radius, -1.0)
    max_similarity = similarity.max(dim=1).values
    threshold = max_similarity.mean() + sigma * max_similarity.std()
    return max_similarity < threshold, max_similarity


def register_ah_hooks(model: torch.nn.Module, cache: dict) -> list[torch.utils.hooks.RemovableHandle]:
    handles = []

    def make_hook(layer_id: int):
        def hook(_module, _inputs, output):
            if not bool(cache.get("record", False)):
                return
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            q, k, _v = tensor.split(tensor.shape[-1] // 3, dim=-1)
            skip_prefix = 1 + int(getattr(model, "n_storage_tokens", 0))
            cache[f"{layer_id}_q"] = q[:, skip_prefix:, :].detach()
            cache[f"{layer_id}_k"] = k[:, skip_prefix:, :].detach()

        return hook

    for layer_id, block in enumerate(model.blocks):
        handles.append(block.attn.qkv.register_forward_hook(make_hook(layer_id)))
    return handles


def remove_hooks(handles: list[torch.utils.hooks.RemovableHandle]) -> None:
    while handles:
        handles.pop().remove()


@torch.no_grad()
def extract_tokens_and_ah_cache(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    patch_size: int,
    register_border: int,
    register_fill: str,
    enable_ah: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict, int, int, int, int]:
    registered_tensor, _register_mask = diag.surround_tensor_with_registers(
        image_tensor,
        patch_size=patch_size,
        register_border=register_border,
        register_fill=register_fill,
    )
    image_mask, image_h, image_w, full_h, full_w = registered_image_token_mask(
        image_tensor.shape[-1],
        patch_size,
        register_border,
        image_tensor.device,
    )
    cache = {"record": False}
    handles = []
    try:
        if enable_ah:
            handles = register_ah_hooks(model, cache)
            cache["record"] = True
        features = model.forward_features(registered_tensor)
        cache["record"] = False
    finally:
        remove_hooks(handles)
    full_tokens = F.normalize(features["x_norm_patchtokens"][0].float(), dim=-1)
    image_tokens = full_tokens[image_mask]
    return full_tokens, image_tokens, cache, image_h, image_w, full_h, full_w


@torch.no_grad()
def encode_dense_with_cache(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    enable_ah: bool,
) -> tuple[torch.Tensor, dict]:
    cache = {"record": False}
    handles = []
    try:
        if enable_ah:
            handles = register_ah_hooks(model, cache)
            cache["record"] = True
        features = model.forward_features(image_tensor)
        cache["record"] = False
    finally:
        remove_hooks(handles)
    tokens = F.normalize(features["x_norm_patchtokens"].float(), dim=-1)
    return tokens, cache


@torch.no_grad()
def evaluate_paper_style_image(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    background_tensor: torch.Tensor,
    args: argparse.Namespace,
    enable_ah: bool,
    ah_start: int,
    ah_end: int,
) -> tuple[dict, dict[str, torch.Tensor], int, int]:
    batch_size = 1
    patch_size = int(model.patch_size)
    crop_boxes = sample_random_crop_boxes(
        scale=(args.crop_scale_min, args.crop_scale_max),
        ratio=(2 / 3, 3 / 2),
        num_boxes=args.num_proposals * batch_size,
    )
    crop_boxes = torch.from_numpy(crop_boxes).to(device=image_tensor.device, dtype=torch.float32).reshape(
        batch_size,
        -1,
        4,
    )
    crop_images = roi_align_feature_map(image_tensor.float(), crop_boxes, size=args.resolution)
    composited_crop_images, foreground_token_mask = compose_crop_with_background(
        crop_images,
        background_tensor,
        ratio=args.crop_bg_ratio,
        patch_size=patch_size,
    )

    composited_tokens, cache = encode_dense_with_cache(model, composited_crop_images, enable_ah)
    _, _, feature_dim = composited_tokens.shape
    teacher_background_tokens = composited_tokens[~foreground_token_mask].reshape(
        args.num_proposals,
        -1,
        feature_dim,
    )
    teacher_foreground_tokens = composited_tokens[foreground_token_mask].reshape(
        args.num_proposals,
        -1,
        feature_dim,
    )
    background_tokens, _ = encode_dense_with_cache(model, background_tensor[:1], False)

    fp_clean = analyze_fp_gp_similarity(
        teacher_foreground_tokens,
        background_tokens,
        thres_sigma=args.fp_gp_sigma,
    )
    gp_clean = analyze_fp_gp_similarity(
        teacher_foreground_tokens,
        teacher_background_tokens,
        thres_sigma=args.fp_gp_sigma,
    )
    pre_ah_clean = fp_clean & gp_clean
    if enable_ah:
        ah_clean = analyze_attention_hijacking(
            cache,
            layer_start=ah_start,
            layer_end=ah_end,
            thres_sigma=args.ah_sigma,
            foreground_mask=foreground_token_mask,
            rejected_mask=~pre_ah_clean,
        )
    else:
        ah_clean = torch.ones_like(pre_ah_clean)

    fp_abnormal = ~fp_clean
    gp_abnormal = ~gp_clean
    ah_abnormal = ~ah_clean
    abnormal = fp_abnormal | gp_abnormal | ah_abnormal

    fp_scores = (teacher_foreground_tokens @ background_tokens.expand(args.num_proposals, -1, -1).transpose(1, 2)).max(dim=2).values
    gp_scores = (teacher_foreground_tokens @ teacher_background_tokens.transpose(1, 2)).max(dim=2).values
    fp_p50, fp_p95 = quantiles(fp_scores.reshape(-1))
    gp_p50, gp_p95 = quantiles(gp_scores.reshape(-1))
    result = {
        "fp_ratio": float(fp_abnormal.float().mean().item()),
        "gp_ratio": float(gp_abnormal.float().mean().item()),
        "ah_ratio": float(ah_abnormal.float().mean().item()),
        "abnormal_ratio": float(abnormal.float().mean().item()),
        "clean_ratio": float((~abnormal).float().mean().item()),
        "fp_inter_max_mean": float(fp_scores.mean().item()),
        "fp_inter_max_p50": fp_p50,
        "fp_inter_max_p95": fp_p95,
        "gp_intra_max_mean": float(gp_scores.mean().item()),
        "gp_intra_max_p50": gp_p50,
        "gp_intra_max_p95": gp_p95,
    }
    masks = {
        "fp": fp_abnormal.reshape(-1, fp_abnormal.shape[-1]),
        "gp": gp_abnormal.reshape(-1, gp_abnormal.shape[-1]),
        "ah": ah_abnormal.reshape(-1, ah_abnormal.shape[-1]),
        "abnormal": abnormal.reshape(-1, abnormal.shape[-1]),
    }
    side = int(math.sqrt(masks["abnormal"].shape[-1]))
    return result, masks, side, side


def mask_to_panel(mask: torch.Tensor, grid_h: int, grid_w: int, size: int, title: str) -> Image.Image:
    heat = diag.score_to_heatmap(mask.float(), grid_h, grid_w, size)
    return diag.draw_title(heat, title)


def save_mask_visualization(
    output_path: Path,
    image: Image.Image,
    fp_mask: torch.Tensor,
    gp_mask: torch.Tensor,
    ah_mask: torch.Tensor,
    abnormal_mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    size: int,
    title: str,
) -> None:
    original = diag.draw_title(image.convert("RGB").resize((size, size), Image.BICUBIC), title)
    panels = [
        original,
        mask_to_panel(fp_mask, grid_h, grid_w, size, "FP abnormal"),
        mask_to_panel(gp_mask, grid_h, grid_w, size, "GP abnormal"),
        mask_to_panel(ah_mask, grid_h, grid_w, size, "AH abnormal"),
        mask_to_panel(abnormal_mask, grid_h, grid_w, size, "FP | GP | AH"),
    ]
    cols = 3
    rows = 2
    canvas = Image.new("RGB", (cols * size, rows * size), "black")
    for idx, panel in enumerate(panels):
        canvas.paste(panel, ((idx % cols) * size, (idx // cols) * size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def mean(values: list[float]) -> float:
    valid = [v for v in values if math.isfinite(v)]
    return float(sum(valid) / max(len(valid), 1))


def summarize(records: list[dict]) -> dict:
    keys = [
        "fp_ratio",
        "gp_ratio",
        "ah_ratio",
        "abnormal_ratio",
        "clean_ratio",
        "fp_inter_max_mean",
        "fp_inter_max_p95",
        "gp_intra_max_mean",
        "gp_intra_max_p95",
    ]
    summary = {"num_images": len(records)}
    for key in keys:
        summary[key] = mean([float(record[key]) for record in records])
    return summary


def quantiles(values: torch.Tensor) -> tuple[float, float]:
    q = torch.quantile(values.float(), values.new_tensor([0.5, 0.95]))
    return float(q[0].item()), float(q[1].item())


@torch.no_grad()
def evaluate_variant(
    label: str,
    pretrained_encoder: str | None,
    args: argparse.Namespace,
    records,
    device: torch.device,
) -> dict:
    model = build_model(args.encoder, pretrained_encoder, device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    patch_size = int(model.patch_size)
    image_dir = Path(args.data_root) / args.split
    enable_ah = not args.disable_ah
    layer_count = len(model.blocks)
    ah_start = max(0, min(args.ah_layer_start, layer_count - 1))
    ah_end = layer_count - 1 if args.ah_layer_end < 0 else args.ah_layer_end
    ah_end = max(ah_start, min(ah_end, layer_count - 1))

    per_image = []
    variant_dir = Path(args.output_dir) / label
    examples_dir = variant_dir / "examples"
    background_tensor = None
    if args.paper_style:
        background_tensor = diag.image_to_tensor(Image.open(args.background_image).convert("RGB"), args.resolution, device)

    for idx, record in enumerate(records):
        image_path = getattr(record, "path", None)
        if image_path is None:
            image_path = image_dir / record.file_name
            if not image_path.exists():
                image_path = Path(args.data_root) / record.file_name
        image = Image.open(image_path).convert("RGB")
        image_tensor = diag.image_to_tensor(image, args.resolution, device)

        if args.paper_style:
            result, masks, image_h, image_w = evaluate_paper_style_image(
                model,
                image_tensor,
                background_tensor,
                args,
                enable_ah,
                ah_start,
                ah_end,
            )
            fp_abnormal = masks["fp"][0]
            gp_abnormal = masks["gp"][0]
            ah_abnormal = masks["ah"][0]
            abnormal = masks["abnormal"][0]
        else:
            reference = records[(idx + 1) % len(records)]
            reference_path = getattr(reference, "path", None)
            if reference_path is None:
                reference_path = image_dir / reference.file_name
                if not reference_path.exists():
                    reference_path = Path(args.data_root) / reference.file_name
            reference_image = Image.open(reference_path).convert("RGB")
            reference_tensor = diag.image_to_tensor(reference_image, args.resolution, device)

            full_tokens, image_tokens, cache, image_h, image_w, _full_h, _full_w = extract_tokens_and_ah_cache(
                model,
                image_tensor,
                patch_size,
                args.register_border,
                args.register_fill,
                enable_ah,
            )
            _reference_full_tokens, reference_image_tokens, _reference_cache, *_ = extract_tokens_and_ah_cache(
                model,
                reference_tensor,
                patch_size,
                args.register_border,
                args.register_fill,
                False,
            )

            fp_clean = analyze_fp_gp_similarity(
                image_tokens.unsqueeze(0),
                reference_image_tokens.unsqueeze(0),
                thres_sigma=args.fp_gp_sigma,
            )[0]
            gp_clean, gp_scores = compute_gp_clean_mask(
                image_tokens,
                image_h,
                image_w,
                sigma=args.fp_gp_sigma,
                exclude_radius=args.gp_exclude_radius,
            )
            fp_scores = (image_tokens @ reference_image_tokens.transpose(0, 1)).max(dim=1).values

            if enable_ah:
                full_image_mask, *_ = registered_image_token_mask(
                    args.resolution,
                    patch_size,
                    args.register_border,
                    device,
                )
                pre_ah_clean = (fp_clean & gp_clean).unsqueeze(0)
                ah_clean = analyze_attention_hijacking(
                    cache,
                    layer_start=ah_start,
                    layer_end=ah_end,
                    thres_sigma=args.ah_sigma,
                    foreground_mask=full_image_mask.unsqueeze(0),
                    rejected_mask=~pre_ah_clean,
                )[0]
            else:
                ah_clean = torch.ones_like(fp_clean)

            fp_abnormal = ~fp_clean
            gp_abnormal = ~gp_clean
            ah_abnormal = ~ah_clean
            abnormal = fp_abnormal | gp_abnormal | ah_abnormal
            fp_p50, fp_p95 = quantiles(fp_scores)
            gp_p50, gp_p95 = quantiles(gp_scores)
            result = {
                "fp_ratio": float(fp_abnormal.float().mean().item()),
                "gp_ratio": float(gp_abnormal.float().mean().item()),
                "ah_ratio": float(ah_abnormal.float().mean().item()),
                "abnormal_ratio": float(abnormal.float().mean().item()),
                "clean_ratio": float((~abnormal).float().mean().item()),
                "fp_inter_max_mean": float(fp_scores.mean().item()),
                "fp_inter_max_p50": fp_p50,
                "fp_inter_max_p95": fp_p95,
                "gp_intra_max_mean": float(gp_scores.mean().item()),
                "gp_intra_max_p50": gp_p50,
                "gp_intra_max_p95": gp_p95,
            }
        result.update({"image_id": int(record.image_id), "file_name": record.file_name})
        per_image.append(result)

        if idx < args.visualize:
            save_mask_visualization(
                examples_dir / f"{idx:04d}_{record.image_id}_abnormal.jpg",
                image,
                fp_abnormal,
                gp_abnormal,
                ah_abnormal,
                abnormal,
                image_h,
                image_w,
                args.resolution,
                f"{label}: {record.image_id}",
            )

        if (idx + 1) % 25 == 0 or idx + 1 == len(records):
            print(f"[{label}] {idx + 1}/{len(records)}", flush=True)

    summary = summarize(per_image)
    output = {
        "variant": label,
        "pretrained_encoder": pretrained_encoder,
        "config": {
            "encoder": args.encoder,
            "resolution": args.resolution,
            "num_images": len(records),
            "register_border": args.register_border,
            "register_fill": args.register_fill,
            "paper_style": args.paper_style,
            "num_proposals": args.num_proposals,
            "crop_scale": [args.crop_scale_min, args.crop_scale_max],
            "crop_bg_ratio": args.crop_bg_ratio,
            "background_image": args.background_image,
            "fp_gp_sigma": args.fp_gp_sigma,
            "gp_exclude_radius": args.gp_exclude_radius,
            "disable_ah": args.disable_ah,
            "ah_layer_start": ah_start,
            "ah_layer_end": ah_end,
            "ah_sigma": args.ah_sigma,
        },
        "summary": summary,
        "per_image": per_image,
    }
    variant_dir.mkdir(parents=True, exist_ok=True)
    with (variant_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image_root:
        records = load_plain_image_records(Path(args.image_root))
    else:
        ann_file = Path(args.ann_file) if args.ann_file else diag.infer_ann_file(Path(args.data_root), args.split)
        records = diag.load_coco_records(Path(args.data_root), args.split, ann_file)
    if args.shuffle:
        random.shuffle(records)
    records = records[: args.num_images]
    if len(records) < 2:
        raise RuntimeError("Need at least two images to compute inter-image FP ratio.")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    summaries = []
    for label, path in args.variant:
        summary = evaluate_variant(label, path, args, records, device)
        summaries.append({"variant": label, "pretrained_encoder": path or "", **summary})

    fieldnames = [
        "variant",
        "pretrained_encoder",
        "num_images",
        "fp_ratio",
        "gp_ratio",
        "ah_ratio",
        "abnormal_ratio",
        "clean_ratio",
        "fp_inter_max_mean",
        "fp_inter_max_p95",
        "gp_intra_max_mean",
        "gp_intra_max_p95",
    ]
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"Saved abnormal-token ratio comparison to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
