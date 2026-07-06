#!/usr/bin/env python3
"""Analyze DINOv3 dense-token foreground alignment and spurious-token proxies.

The script is intentionally independent from RF-DETR training. It computes:

* CLS-patch cosine score and COCO-style Point-in-Box.
* LAST-ViT-style frequency-stability vote score and Point-in-Box.
* A lightweight UniRefiner-style FP/GP proxy ratio.

The metrics are diagnostic only; they are meant to compare backbones or
refinement variants under the same settings.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TVF


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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    bboxes_xywh: list[tuple[float, float, float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("DINOv3 token diagnostics")
    parser.add_argument("--data-root", default="/data/cpc/root/dataset/COCO")
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--ann-file", default=None)
    parser.add_argument("--output-dir", default="output/token_analysis/dinov3_coco_val")
    parser.add_argument("--encoder", default="dinov3_small", choices=tuple(f"dinov3_{k}" for k in SIZE_TO_MODEL))
    parser.add_argument("--pretrained-encoder", default=None)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--visualize", type=int, default=24)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--coverage-topk", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--size-coverage-topk", type=int, default=100)
    parser.add_argument("--last-topk", type=int, default=1)
    parser.add_argument("--fp-gp-threshold", type=float, default=0.5)
    parser.add_argument("--fp-gp-sigma", type=float, default=None)
    parser.add_argument("--fixed-fp-threshold", type=float, default=None)
    parser.add_argument("--fixed-gp-threshold", type=float, default=None)
    parser.add_argument("--gp-exclude-radius", type=int, default=1)
    parser.add_argument("--reference-offset", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--pca-visualize", action="store_true", help="Save UniRefiner-style dense-token PCA panels.")
    parser.add_argument("--pca-register-border", type=int, default=1, help="Register border width in patch tokens.")
    parser.add_argument("--pca-register-fill", choices=("zero", "rand", "randn"), default="zero")
    parser.add_argument(
        "--hist-from-metrics",
        default=None,
        help="Regenerate histogram images from an existing metrics.json without rerunning DINOv3.",
    )
    return parser.parse_args()


def infer_ann_file(data_root: Path, split: str) -> Path:
    candidates = [
        data_root / "annotations" / f"instances_{split}.json",
        data_root / f"instances_{split}.json",
        data_root / split / "_annotations.coco.json",
        data_root / "annotations.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not infer annotation file under {data_root} for split={split}")


def load_coco_records(data_root: Path, split: str, ann_file: Path) -> list[ImageRecord]:
    with ann_file.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images = {int(img["id"]): img for img in coco.get("images", [])}
    boxes_by_image: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        bbox = ann.get("bbox", None)
        if bbox is None or len(bbox) != 4:
            continue
        x, y, w, h = [float(v) for v in bbox]
        if w <= 1 or h <= 1:
            continue
        boxes_by_image[int(ann["image_id"])].append((x, y, w, h))

    records = []
    for image_id, image in images.items():
        boxes = boxes_by_image.get(image_id, [])
        if not boxes:
            continue
        file_name = image["file_name"]
        image_path = data_root / split / file_name
        if not image_path.exists():
            image_path = data_root / file_name
        if not image_path.exists():
            continue
        records.append(
            ImageRecord(
                image_id=image_id,
                file_name=file_name,
                width=int(image["width"]),
                height=int(image["height"]),
                bboxes_xywh=boxes,
            )
        )
    return records


def build_dinov3_encoder(args: argparse.Namespace) -> torch.nn.Module:
    size = args.encoder.split("_", 1)[1]
    model_name = SIZE_TO_MODEL[size]
    builder = _load_dinov3_backbone_builder(model_name)
    model = builder(pretrained=False)
    weights_path = _resolve_weights_path(model_name, args.pretrained_encoder)
    state_dict = _load_state_dict(weights_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def image_to_tensor(image: Image.Image, resolution: int, device: torch.device) -> torch.Tensor:
    image = image.convert("RGB").resize((resolution, resolution), Image.BICUBIC)
    tensor = TVF.to_tensor(image)
    tensor = TVF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor.unsqueeze(0).to(device=device, non_blocking=True)


def boxes_to_resized_xyxy(
    boxes_xywh: Iterable[tuple[float, float, float, float]],
    original_width: int,
    original_height: int,
    resolution: int,
) -> torch.Tensor:
    scale_x = resolution / float(original_width)
    scale_y = resolution / float(original_height)
    boxes = []
    for x, y, w, h in boxes_xywh:
        boxes.append((x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y))
    return torch.tensor(boxes, dtype=torch.float32)


def patch_centers(grid_h: int, grid_w: int, patch_size: int, device: torch.device) -> torch.Tensor:
    y = (torch.arange(grid_h, device=device, dtype=torch.float32) + 0.5) * patch_size
    x = (torch.arange(grid_w, device=device, dtype=torch.float32) + 0.5) * patch_size
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def point_in_any_box(points_xy: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
    inside = point_in_boxes(points_xy, boxes_xyxy)
    return inside.any(dim=1)


def point_in_boxes(points_xy: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return torch.zeros((points_xy.shape[0], 0), dtype=torch.bool, device=points_xy.device)
    boxes_xyxy = boxes_xyxy.to(device=points_xy.device)
    px = points_xy[:, 0:1]
    py = points_xy[:, 1:2]
    return (
        (px >= boxes_xyxy[:, 0])
        & (px <= boxes_xyxy[:, 2])
        & (py >= boxes_xyxy[:, 1])
        & (py <= boxes_xyxy[:, 3])
    )


def box_size_masks(
    boxes_xywh: Iterable[tuple[float, float, float, float]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    areas = torch.tensor([w * h for _, _, w, h in boxes_xywh], dtype=torch.float32, device=device)
    return {
        "small": areas < 32**2,
        "medium": (areas >= 32**2) & (areas < 96**2),
        "large": areas >= 96**2,
    }


def masked_mean_bool(values: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        values = values[mask]
    if values.numel() == 0:
        return float("nan")
    return float(values.float().mean().item())


def masked_ratio(values: torch.Tensor, mask: torch.Tensor) -> float:
    values = values[mask]
    if values.numel() == 0:
        return float("nan")
    return float(values.float().mean().item())


def score_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    if values.numel() == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan")}
    qs = torch.quantile(values, values.new_tensor([0.5, 0.9, 0.95, 0.99]))
    return {
        "mean": float(values.mean().item()),
        "p50": float(qs[0].item()),
        "p90": float(qs[1].item()),
        "p95": float(qs[2].item()),
        "p99": float(qs[3].item()),
    }


def add_prefixed_stats(result: dict, prefix: str, values: torch.Tensor) -> None:
    for key, value in score_stats(values).items():
        result[f"{prefix}_{key}"] = value


def box_coverage(
    ranked_indices: torch.Tensor,
    inside_by_box: torch.Tensor,
    box_mask: torch.Tensor | None = None,
) -> float:
    if inside_by_box.numel() == 0:
        return float("nan")
    covered = inside_by_box[ranked_indices].any(dim=0)
    return masked_mean_bool(covered, box_mask)


def per_box_max_score(score: torch.Tensor, inside_by_box: torch.Tensor) -> torch.Tensor:
    if inside_by_box.shape[1] == 0:
        return score.new_empty((0,))
    expanded = score[:, None].expand_as(inside_by_box).float()
    neg_inf = torch.full_like(expanded, -torch.inf)
    values = torch.where(inside_by_box, expanded, neg_inf).max(dim=0).values
    return torch.where(torch.isfinite(values), values, torch.full_like(values, float("nan")))


def finite_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        values = values[mask]
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return float("nan")
    return float(values.mean().item())


def finite_median(values: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        values = values[mask]
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return float("nan")
    return float(values.median().item())


def add_multi_object_metrics(
    result: dict,
    prefix: str,
    score: torch.Tensor,
    coverage_topk: list[int],
    size_coverage_topk: int,
    inside_by_box: torch.Tensor,
    size_masks: dict[str, torch.Tensor],
) -> None:
    ranked = torch.argsort(score, descending=True)
    for k in coverage_topk:
        topk = ranked[: min(k, ranked.numel())]
        result[f"{prefix}_box_coverage_top{k}"] = box_coverage(topk, inside_by_box)

    topk = ranked[: min(size_coverage_topk, ranked.numel())]
    for size_name, size_mask in size_masks.items():
        result[f"{prefix}_{size_name}_box_coverage_top{size_coverage_topk}"] = box_coverage(
            topk,
            inside_by_box,
            size_mask,
        )

    box_max = per_box_max_score(score, inside_by_box)
    result[f"{prefix}_per_box_max_mean"] = finite_mean(box_max)
    result[f"{prefix}_per_box_max_p50"] = finite_median(box_max)
    for size_name, size_mask in size_masks.items():
        result[f"{prefix}_{size_name}_per_box_max_p50"] = finite_median(box_max, size_mask)


def foreground_background_ratio(score: torch.Tensor, foreground_mask: torch.Tensor) -> float:
    score = score.float()
    if foreground_mask.any():
        fg = score[foreground_mask].mean()
    else:
        fg = score.new_tensor(float("nan"))
    if (~foreground_mask).any():
        bg = score[~foreground_mask].mean()
    else:
        bg = score.new_tensor(float("nan"))
    return float((fg / (bg.abs() + 1e-6)).item())


def normalize_01(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    min_v = values.min()
    max_v = values.max()
    return (values - min_v) / (max_v - min_v + 1e-6)


def compute_last_vote_score(
    patch_tokens: torch.Tensor,
    topk: int,
    sigma: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scalar vote score per patch and selected indices.

    patch_tokens: [N, C]. The LAST-ViT selection is channel-wise, so the
    selected indices have shape [K, C].
    """

    original_dtype = patch_tokens.dtype
    work_tokens = patch_tokens.float()
    hidden_dim = work_tokens.shape[-1]
    if sigma is None:
        sigma = math.sqrt(hidden_dim)
    positions = torch.arange(
        -hidden_dim // 2 + 1,
        hidden_dim // 2 + 1,
        device=work_tokens.device,
        dtype=work_tokens.dtype,
    )
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    kernel = kernel / kernel.max()
    spectrum = torch.fft.fft(work_tokens, dim=-1)
    spectrum = torch.fft.fftshift(spectrum, dim=-1)
    spectrum = spectrum * kernel.view(1, -1)
    spectrum = torch.fft.ifftshift(spectrum, dim=-1)
    low_pass = torch.fft.ifft(spectrum, dim=-1).real.to(dtype=original_dtype)

    stability = patch_tokens / (low_pass - patch_tokens).abs().clamp_min(1e-6)
    k = min(topk, patch_tokens.shape[0])
    _, selected = torch.topk(stability, k=k, dim=0, largest=True)
    votes = torch.zeros(patch_tokens.shape[0], device=patch_tokens.device, dtype=torch.float32)
    votes.scatter_add_(0, selected.reshape(-1), torch.ones(selected.numel(), device=patch_tokens.device))
    return votes, selected


def fp_gp_proxy_masks(
    patch_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    grid_h: int,
    grid_w: int,
    threshold: float,
    sigma: float | None,
    exclude_radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute simplified UniRefiner-style FP/GP proxy masks.

    FP proxy: token is too similar to unrelated reference-image tokens.
    GP proxy: token is too similar to distant tokens in the same image.
    """

    patch_tokens = F.normalize(patch_tokens.float(), dim=-1)
    reference_tokens = F.normalize(reference_tokens.float(), dim=-1)

    inter_max = (patch_tokens @ reference_tokens.T).max(dim=1).values
    intra = patch_tokens @ patch_tokens.T
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(grid_h, device=patch_tokens.device),
            torch.arange(grid_w, device=patch_tokens.device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    distance = (coords[:, None, :] - coords[None, :, :]).abs().amax(dim=-1)
    intra = intra.masked_fill(distance <= exclude_radius, -1.0)
    intra_max = intra.max(dim=1).values

    if sigma is not None:
        fp_threshold = inter_max.mean() + sigma * inter_max.std()
        gp_threshold = intra_max.mean() + sigma * intra_max.std()
    else:
        fp_threshold = torch.as_tensor(threshold, device=patch_tokens.device)
        gp_threshold = torch.as_tensor(threshold, device=patch_tokens.device)

    fp_mask = inter_max >= fp_threshold
    gp_mask = intra_max >= gp_threshold
    return fp_mask, gp_mask, fp_mask | gp_mask, inter_max, intra_max


def score_to_heatmap(score: torch.Tensor, grid_h: int, grid_w: int, size: int) -> Image.Image:
    score = normalize_01(score.detach().cpu()).reshape(grid_h, grid_w)
    score_img = Image.fromarray((score.numpy() * 255).astype("uint8"), mode="L")
    score_img = score_img.resize((size, size), Image.BICUBIC)
    # Blue-to-red heat map without requiring matplotlib.
    heat = Image.new("RGB", (size, size))
    pix = heat.load()
    src = score_img.load()
    for y in range(size):
        for x in range(size):
            v = src[x, y] / 255.0
            r = int(255 * min(max(1.5 * v - 0.25, 0.0), 1.0))
            g = int(255 * min(max(1.5 - abs(2.0 * v - 1.0) * 1.5, 0.0), 1.0))
            b = int(255 * min(max(1.25 - 1.5 * v, 0.0), 1.0))
            pix[x, y] = (r, g, b)
    return heat


def surround_tensor_with_registers(
    image_tensor: torch.Tensor,
    patch_size: int,
    register_border: int,
    register_fill: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if register_border <= 0:
        token_h = image_tensor.shape[-2] // patch_size
        token_w = image_tensor.shape[-1] // patch_size
        mask = torch.zeros((1, token_h * token_w), dtype=torch.bool, device=image_tensor.device)
        return image_tensor, mask

    batch_size, channels, height, width = image_tensor.shape
    pad = register_border * patch_size
    out_h = height + 2 * pad
    out_w = width + 2 * pad
    if register_fill == "zero":
        canvas = torch.zeros((batch_size, channels, out_h, out_w), dtype=image_tensor.dtype, device=image_tensor.device)
    elif register_fill == "randn":
        canvas = torch.randn((batch_size, channels, out_h, out_w), dtype=image_tensor.dtype, device=image_tensor.device)
    else:
        canvas = torch.rand((batch_size, channels, out_h, out_w), dtype=image_tensor.dtype, device=image_tensor.device)
    canvas[:, :, pad : pad + height, pad : pad + width] = image_tensor

    token_h = out_h // patch_size
    token_w = out_w // patch_size
    mask = torch.zeros((1, token_h, token_w), dtype=torch.bool, device=image_tensor.device)
    mask[:, :register_border, :] = True
    mask[:, -register_border:, :] = True
    mask[:, register_border:-register_border, :register_border] = True
    mask[:, register_border:-register_border, -register_border:] = True
    return canvas, mask.reshape(1, -1)


def compute_pca_metadata(features: torch.Tensor) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    features = features.float()
    center = features.mean(dim=0, keepdim=True)
    centered = features - center
    if features.shape[0] < 2:
        eigenvectors = torch.eye(features.shape[1], dtype=features.dtype, device=features.device)
    else:
        covariance = centered.T @ centered / max(features.shape[0] - 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvectors = eigenvectors[:, torch.argsort(eigenvalues, descending=True)]
    projected = centered @ eigenvectors[:, :3]
    return {
        "center": center,
        "eigenvectors": eigenvectors,
        "value_range": (
            torch.quantile(projected, 0.01, dim=0),
            torch.quantile(projected, 0.99, dim=0),
        ),
    }


def pca_tokens_to_image(
    features: torch.Tensor,
    grid_h: int,
    grid_w: int,
    output_size: int,
    metadata: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
) -> Image.Image:
    features = features.float()
    center = metadata["center"].to(device=features.device, dtype=features.dtype)
    eigenvectors = metadata["eigenvectors"].to(device=features.device, dtype=features.dtype)
    low, high = metadata["value_range"]
    low = low.to(device=features.device, dtype=features.dtype).view(1, 3)
    high = high.to(device=features.device, dtype=features.dtype).view(1, 3)
    projected = (features - center) @ eigenvectors[:, :3]
    rgb = (projected - low) / (high - low + 1e-8)
    rgb = torch.clamp(rgb, 0.0, 1.0)
    array = (rgb.reshape(grid_h, grid_w, 3).detach().cpu() * 255).to(torch.uint8).numpy()
    image = Image.fromarray(array, mode="RGB")
    return image.resize((output_size, output_size), Image.NEAREST)


def register_mask_to_image(mask: torch.Tensor, grid_h: int, grid_w: int, output_size: int) -> Image.Image:
    mask_img = mask.reshape(grid_h, grid_w).detach().cpu().to(torch.uint8) * 255
    image = Image.fromarray(mask_img.numpy(), mode="L").resize((output_size, output_size), Image.NEAREST)
    return Image.merge("RGB", (image, image, image))


@torch.no_grad()
def save_pca_visualization(
    output_path: Path,
    encoder: torch.nn.Module,
    image: Image.Image,
    image_tensor: torch.Tensor,
    patch_tokens: torch.Tensor,
    grid_h: int,
    grid_w: int,
    patch_size: int,
    resolution: int,
    register_border: int,
    register_fill: str,
    title: str,
) -> None:
    registered_tensor, register_mask = surround_tensor_with_registers(
        image_tensor,
        patch_size=patch_size,
        register_border=register_border,
        register_fill=register_fill,
    )
    registered_features = encoder.forward_features(registered_tensor)
    registered_tokens = F.normalize(registered_features["x_norm_patchtokens"][0].float(), dim=-1)
    reg_h = registered_tensor.shape[-2] // patch_size
    reg_w = registered_tensor.shape[-1] // patch_size

    metadata = compute_pca_metadata(registered_tokens)
    no_register_pca = pca_tokens_to_image(patch_tokens, grid_h, grid_w, resolution, metadata)
    register_pca = pca_tokens_to_image(registered_tokens, reg_h, reg_w, resolution, metadata)
    register_mask_img = register_mask_to_image(register_mask[0], reg_h, reg_w, resolution)
    original = image.convert("RGB").resize((resolution, resolution), Image.BICUBIC)

    panels = [
        draw_title(original, title),
        draw_title(no_register_pca, "PCA: image tokens only"),
        draw_title(register_pca, f"PCA: +{register_border} {register_fill} register border"),
        draw_title(register_mask_img, "register-token mask"),
    ]
    canvas = Image.new("RGB", (resolution * 2, resolution * 2))
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (resolution, 0))
    canvas.paste(panels[2], (0, resolution))
    canvas.paste(panels[3], (resolution, resolution))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def overlay_heatmap(image: Image.Image, heat: Image.Image, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGB").resize(heat.size, Image.BICUBIC)
    return Image.blend(base, heat, alpha=alpha)


def draw_boxes(image: Image.Image, boxes_xyxy: torch.Tensor, color: str = "lime") -> Image.Image:
    draw = ImageDraw.Draw(image)
    for box in boxes_xyxy.cpu().tolist():
        draw.rectangle(box, outline=color, width=2)
    return image


def draw_title(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = None
    draw.rectangle((0, 0, image.width, 22), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255), font=font)
    return image


def save_visualization(
    output_path: Path,
    image: Image.Image,
    boxes_xyxy: torch.Tensor,
    patch_score: torch.Tensor,
    last_vote: torch.Tensor,
    fp_gp_mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    resolution: int,
    title: str,
) -> None:
    score_panel = draw_boxes(
        draw_title(overlay_heatmap(image, score_to_heatmap(patch_score, grid_h, grid_w, resolution)), "CLS patch score"),
        boxes_xyxy,
    )
    last_panel = draw_boxes(
        draw_title(overlay_heatmap(image, score_to_heatmap(last_vote, grid_h, grid_w, resolution)), "LAST stability votes"),
        boxes_xyxy,
    )
    spurious_panel = draw_boxes(
        draw_title(overlay_heatmap(image, score_to_heatmap(fp_gp_mask.float(), grid_h, grid_w, resolution)), "FP-GP proxy mask"),
        boxes_xyxy,
    )
    original_panel = draw_boxes(draw_title(image.convert("RGB").resize((resolution, resolution)), title), boxes_xyxy)
    canvas = Image.new("RGB", (resolution * 2, resolution * 2))
    canvas.paste(original_panel, (0, 0))
    canvas.paste(score_panel, (resolution, 0))
    canvas.paste(last_panel, (0, resolution))
    canvas.paste(spurious_panel, (resolution, resolution))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def mean(values: list[float]) -> float:
    valid = [v for v in values if math.isfinite(v)]
    return float(sum(valid) / max(len(valid), 1))


def summarize(records: list[dict]) -> dict:
    summary = {
        "num_images": len(records),
        "patch_score_pib_any": mean([r["patch_score_pib_any"] for r in records]),
        "patch_score_topk_pib_any": mean([r["patch_score_topk_pib_any"] for r in records]),
        "patch_score_fg_bg_ratio": mean([r["patch_score_fg_bg_ratio"] for r in records]),
        "last_vote_pib_any": mean([r["last_vote_pib_any"] for r in records]),
        "last_vote_topk_pib_any": mean([r["last_vote_topk_pib_any"] for r in records]),
        "last_vote_fg_bg_ratio": mean([r["last_vote_fg_bg_ratio"] for r in records]),
        "fp_proxy_ratio": mean([r["fp_proxy_ratio"] for r in records]),
        "gp_proxy_ratio": mean([r["gp_proxy_ratio"] for r in records]),
        "fp_gp_proxy_ratio": mean([r["fp_gp_proxy_ratio"] for r in records]),
    }
    metric_prefixes = (
        "box_with_patch_center_ratio",
        "small_box_with_patch_center_ratio",
        "medium_box_with_patch_center_ratio",
        "large_box_with_patch_center_ratio",
        "patch_score_box_coverage_",
        "patch_score_small_box_coverage_",
        "patch_score_medium_box_coverage_",
        "patch_score_large_box_coverage_",
        "patch_score_per_box_",
        "patch_score_small_per_box_",
        "patch_score_medium_per_box_",
        "patch_score_large_per_box_",
        "last_vote_box_coverage_",
        "last_vote_small_box_coverage_",
        "last_vote_medium_box_coverage_",
        "last_vote_large_box_coverage_",
        "last_vote_per_box_",
        "last_vote_small_per_box_",
        "last_vote_medium_per_box_",
        "last_vote_large_per_box_",
        "fp_inter_max_",
        "gp_intra_max_",
        "fp_proxy_ratio_",
        "gp_proxy_ratio_",
        "fp_gp_proxy_ratio_",
        "fixed_fp_proxy_ratio",
        "fixed_gp_proxy_ratio",
        "fixed_fp_gp_proxy_ratio",
    )
    for key in sorted(records[0].keys()) if records else []:
        if key in summary:
            continue
        if key.startswith(metric_prefixes):
            summary[key] = mean([float(r[key]) for r in records])
    return summary


def save_histogram(records: list[dict], output_path: Path, key: str, bins: int = 20) -> None:
    values = [
        float(r[key])
        for r in records
        if key in r and math.isfinite(float(r[key]))
    ]
    width, height = 720, 360
    margin = 48
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    if not values:
        canvas.save(output_path)
        return
    min_v, max_v = min(values), max(values)
    if abs(max_v - min_v) < 1e-9:
        max_v = min_v + 1.0
    counts = [0 for _ in range(bins)]
    for value in values:
        idx = min(int((value - min_v) / (max_v - min_v) * bins), bins - 1)
        counts[idx] += 1
    max_count = max(counts) or 1
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    for i, count in enumerate(counts):
        x0 = margin + i * plot_w / bins
        x1 = margin + (i + 1) * plot_w / bins - 2
        bar_h = plot_h * count / max_count
        y0 = height - margin - bar_h
        draw.rectangle((x0, y0, x1, height - margin), fill=(79, 129, 189))
    draw.line((margin, margin, margin, height - margin), fill="black")
    draw.line((margin, height - margin, width - margin, height - margin), fill="black")
    draw.text((margin, 14), f"{key} histogram", fill="black")
    draw.text((margin, height - margin + 8), f"{min_v:.3f}", fill="black")
    draw.text((width - margin - 60, height - margin + 8), f"{max_v:.3f}", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_all_histograms(records: list[dict], output_dir: Path) -> None:
    save_histogram(records, output_dir / "fp_gp_proxy_ratio_hist.png", "fp_gp_proxy_ratio")
    save_histogram(records, output_dir / "patch_score_fg_bg_ratio_hist.png", "patch_score_fg_bg_ratio")
    save_histogram(records, output_dir / "last_vote_fg_bg_ratio_hist.png", "last_vote_fg_bg_ratio")


COMPACT_SUMMARY_KEYS = (
    "num_images",
    "fixed_fp_gp_proxy_ratio",
    "fixed_fp_proxy_ratio",
    "fixed_gp_proxy_ratio",
    "fixed_fp_gp_proxy_ratio_fg",
    "fixed_fp_gp_proxy_ratio_bg",
    "global_fp_inter_max_mean",
    "global_fp_inter_max_p95",
    "global_gp_intra_max_mean",
    "global_gp_intra_max_p95",
    "patch_score_box_coverage_top100",
    "patch_score_small_box_coverage_top100",
    "patch_score_medium_box_coverage_top100",
    "patch_score_large_box_coverage_top100",
    "last_vote_box_coverage_top100",
    "last_vote_small_box_coverage_top100",
    "last_vote_medium_box_coverage_top100",
    "last_vote_large_box_coverage_top100",
    "patch_score_fg_bg_ratio",
    "last_vote_fg_bg_ratio",
)


def save_compact_summary(summary: dict, output_dir: Path) -> None:
    compact = {key: summary[key] for key in COMPACT_SUMMARY_KEYS if key in summary}
    with (output_dir / "summary_compact.json").open("w", encoding="utf-8") as handle:
        json.dump(compact, handle, indent=2, ensure_ascii=False)
    with (output_dir / "summary_table.csv").open("w", encoding="utf-8") as handle:
        handle.write(",".join(compact.keys()) + "\n")
        handle.write(",".join(str(compact[key]) for key in compact) + "\n")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.hist_from_metrics:
        metrics_path = Path(args.hist_from_metrics)
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        save_all_histograms(metrics.get("per_image", []), output_dir)
        if isinstance(metrics.get("summary"), dict):
            save_compact_summary(metrics["summary"], output_dir)
        print(f"Saved histograms to {output_dir}", flush=True)
        return

    data_root = Path(args.data_root)
    ann_file = Path(args.ann_file) if args.ann_file else infer_ann_file(data_root, args.split)

    records = load_coco_records(data_root, args.split, ann_file)
    if args.shuffle:
        random.shuffle(records)
    records = records[: args.num_images]
    if len(records) < 2:
        raise RuntimeError("Need at least two annotated images to compute reference-image FP proxy.")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    encoder = build_dinov3_encoder(args).to(device)
    patch_size = int(encoder.patch_size)
    grid_h = args.resolution // patch_size
    grid_w = args.resolution // patch_size
    centers = patch_centers(grid_h, grid_w, patch_size, device=device)

    per_image = []
    global_inter_scores = []
    global_intra_scores = []
    examples_dir = output_dir / "examples"
    pca_examples_dir = output_dir / "pca_examples"
    image_dir = data_root / args.split

    for idx, record in enumerate(records):
        image_path = image_dir / record.file_name
        if not image_path.exists():
            image_path = data_root / record.file_name
        image = Image.open(image_path).convert("RGB")
        image_tensor = image_to_tensor(image, args.resolution, device)

        reference = records[(idx + args.reference_offset) % len(records)]
        reference_path = image_dir / reference.file_name
        if not reference_path.exists():
            reference_path = data_root / reference.file_name
        reference_image = Image.open(reference_path).convert("RGB")
        reference_tensor = image_to_tensor(reference_image, args.resolution, device)

        features = encoder.forward_features(image_tensor)
        reference_features = encoder.forward_features(reference_tensor)
        cls_token = F.normalize(features["x_norm_clstoken"][0].float(), dim=-1)
        patch_tokens = F.normalize(features["x_norm_patchtokens"][0].float(), dim=-1)
        reference_tokens = F.normalize(reference_features["x_norm_patchtokens"][0].float(), dim=-1)

        patch_score = patch_tokens @ cls_token
        last_vote, _ = compute_last_vote_score(patch_tokens, topk=args.last_topk)
        fp_mask, gp_mask, fp_gp_mask, inter_max, intra_max = fp_gp_proxy_masks(
            patch_tokens,
            reference_tokens,
            grid_h,
            grid_w,
            threshold=args.fp_gp_threshold,
            sigma=args.fp_gp_sigma,
            exclude_radius=args.gp_exclude_radius,
        )

        boxes_xyxy = boxes_to_resized_xyxy(record.bboxes_xywh, record.width, record.height, args.resolution)
        inside_by_box = point_in_boxes(centers, boxes_xyxy)
        fg_mask = inside_by_box.any(dim=1)
        bg_mask = ~fg_mask
        size_masks = box_size_masks(record.bboxes_xywh, device=device)
        box_has_patch_center = inside_by_box.any(dim=0)

        score_argmax = int(patch_score.argmax().item())
        score_topk = torch.topk(patch_score, k=min(args.topk, patch_score.numel())).indices
        vote_argmax = int(last_vote.argmax().item())
        vote_topk = torch.topk(last_vote, k=min(args.topk, last_vote.numel())).indices

        result = {
            "image_id": record.image_id,
            "file_name": record.file_name,
            "num_boxes": len(record.bboxes_xywh),
            "box_with_patch_center_ratio": masked_mean_bool(box_has_patch_center),
            "small_box_with_patch_center_ratio": masked_mean_bool(box_has_patch_center, size_masks["small"]),
            "medium_box_with_patch_center_ratio": masked_mean_bool(box_has_patch_center, size_masks["medium"]),
            "large_box_with_patch_center_ratio": masked_mean_bool(box_has_patch_center, size_masks["large"]),
            "patch_score_pib_any": float(fg_mask[score_argmax].item()),
            "patch_score_topk_pib_any": float(fg_mask[score_topk].any().item()),
            "patch_score_fg_bg_ratio": foreground_background_ratio(patch_score, fg_mask),
            "last_vote_pib_any": float(fg_mask[vote_argmax].item()),
            "last_vote_topk_pib_any": float(fg_mask[vote_topk].any().item()),
            "last_vote_fg_bg_ratio": foreground_background_ratio(last_vote, fg_mask),
            "fp_proxy_ratio": float(fp_mask.float().mean().item()),
            "gp_proxy_ratio": float(gp_mask.float().mean().item()),
            "fp_gp_proxy_ratio": float(fp_gp_mask.float().mean().item()),
            "fp_proxy_ratio_fg": masked_ratio(fp_mask, fg_mask),
            "fp_proxy_ratio_bg": masked_ratio(fp_mask, bg_mask),
            "gp_proxy_ratio_fg": masked_ratio(gp_mask, fg_mask),
            "gp_proxy_ratio_bg": masked_ratio(gp_mask, bg_mask),
            "fp_gp_proxy_ratio_fg": masked_ratio(fp_gp_mask, fg_mask),
            "fp_gp_proxy_ratio_bg": masked_ratio(fp_gp_mask, bg_mask),
        }
        add_prefixed_stats(result, "fp_inter_max", inter_max)
        add_prefixed_stats(result, "gp_intra_max", intra_max)
        if args.fixed_fp_threshold is not None:
            fixed_fp_mask = inter_max >= args.fixed_fp_threshold
            result["fixed_fp_proxy_ratio"] = float(fixed_fp_mask.float().mean().item())
            result["fixed_fp_proxy_ratio_fg"] = masked_ratio(fixed_fp_mask, fg_mask)
            result["fixed_fp_proxy_ratio_bg"] = masked_ratio(fixed_fp_mask, bg_mask)
        if args.fixed_gp_threshold is not None:
            fixed_gp_mask = intra_max >= args.fixed_gp_threshold
            result["fixed_gp_proxy_ratio"] = float(fixed_gp_mask.float().mean().item())
            result["fixed_gp_proxy_ratio_fg"] = masked_ratio(fixed_gp_mask, fg_mask)
            result["fixed_gp_proxy_ratio_bg"] = masked_ratio(fixed_gp_mask, bg_mask)
        if args.fixed_fp_threshold is not None and args.fixed_gp_threshold is not None:
            fixed_fp_gp_mask = (inter_max >= args.fixed_fp_threshold) | (intra_max >= args.fixed_gp_threshold)
            result["fixed_fp_gp_proxy_ratio"] = float(fixed_fp_gp_mask.float().mean().item())
            result["fixed_fp_gp_proxy_ratio_fg"] = masked_ratio(fixed_fp_gp_mask, fg_mask)
            result["fixed_fp_gp_proxy_ratio_bg"] = masked_ratio(fixed_fp_gp_mask, bg_mask)
        add_multi_object_metrics(
            result,
            "patch_score",
            patch_score,
            args.coverage_topk,
            args.size_coverage_topk,
            inside_by_box,
            size_masks,
        )
        add_multi_object_metrics(
            result,
            "last_vote",
            last_vote,
            args.coverage_topk,
            args.size_coverage_topk,
            inside_by_box,
            size_masks,
        )
        per_image.append(result)
        global_inter_scores.append(inter_max.detach().cpu())
        global_intra_scores.append(intra_max.detach().cpu())

        if idx < args.visualize:
            save_visualization(
                examples_dir / f"{idx:04d}_{record.image_id}.jpg",
                image,
                boxes_xyxy,
                patch_score,
                last_vote,
                fp_gp_mask,
                grid_h,
                grid_w,
                args.resolution,
                f"{record.image_id} boxes={len(record.bboxes_xywh)}",
            )
            if args.pca_visualize:
                save_pca_visualization(
                    pca_examples_dir / f"{idx:04d}_{record.image_id}_pca.jpg",
                    encoder,
                    image,
                    image_tensor,
                    patch_tokens,
                    grid_h,
                    grid_w,
                    patch_size,
                    args.resolution,
                    args.pca_register_border,
                    args.pca_register_fill,
                    f"{record.image_id} boxes={len(record.bboxes_xywh)}",
                )

        if (idx + 1) % 25 == 0 or idx + 1 == len(records):
            print(f"[{idx + 1}/{len(records)}] processed", flush=True)

    summary = summarize(per_image)
    if global_inter_scores:
        add_prefixed_stats(summary, "global_fp_inter_max", torch.cat(global_inter_scores))
    if global_intra_scores:
        add_prefixed_stats(summary, "global_gp_intra_max", torch.cat(global_intra_scores))
    output = {
        "config": {
            "data_root": str(data_root),
            "split": args.split,
            "ann_file": str(ann_file),
            "encoder": args.encoder,
            "resolution": args.resolution,
            "patch_size": patch_size,
            "grid": [grid_h, grid_w],
            "num_images": len(records),
            "topk": args.topk,
            "coverage_topk": args.coverage_topk,
            "size_coverage_topk": args.size_coverage_topk,
            "last_topk": args.last_topk,
            "fp_gp_threshold": args.fp_gp_threshold,
            "fp_gp_sigma": args.fp_gp_sigma,
            "fixed_fp_threshold": args.fixed_fp_threshold,
            "fixed_gp_threshold": args.fixed_gp_threshold,
            "gp_exclude_radius": args.gp_exclude_radius,
            "pca_visualize": args.pca_visualize,
            "pca_register_border": args.pca_register_border,
            "pca_register_fill": args.pca_register_fill,
        },
        "summary": summary,
        "per_image": per_image,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    save_compact_summary(summary, output_dir)

    save_all_histograms(per_image, output_dir)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved diagnostics to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
