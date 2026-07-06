#!/usr/bin/env python3
"""Visualize ADE20K DINOv3 dense features before and after refinement.

The montage is intended for qualitative comparison of frozen backbone features:
image, ADE20K label, shared-PCA RGB, feature-norm heatmap, and cosine change
relative to the base encoder.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
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
DEFAULT_VARIANTS = [
    ("base", None),
    (
        "author_dinov2recipe",
        "/data/cpc/root/project/UniRefiner/outputs/"
        "dinov3_small_640_unirefiner_coco_author_dinov2recipe_bs8_2e/checkpoints/model_final.pt",
    ),
    (
        "balanced_zero3e",
        "/data/cpc/root/project/UniRefiner/outputs/"
        "dinov3_small_640_unirefiner_coco_balanced_zero3e/checkpoints/model_final.pt",
    ),
    (
        "randn3e",
        "/data/cpc/root/project/UniRefiner/outputs/"
        "dinov3_small_640_unirefiner_coco_train3e_randn_v3/checkpoints/model_final.pt",
    ),
]


@dataclass(frozen=True)
class Variant:
    label: str
    checkpoint: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ADE20K DINOv3 feature PCA and heatmap visualization")
    parser.add_argument("--data-root", default="/data/cpc/root/dataset/ADE20K/ADEChallengeData2016")
    parser.add_argument("--output-dir", default="output/ade20k_feature_visuals/dinov3_small_refine_compare")
    parser.add_argument("--encoder", default="dinov3_small", choices=tuple(f"dinov3_{k}" for k in SIZE_TO_MODEL))
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 1, 4, 10, 11, 15])
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Variant as label:path. Use label: for the default DINOv3 checkpoint.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_variants(values: list[str] | None) -> list[Variant]:
    if not values:
        return [Variant(label, ckpt) for label, ckpt in DEFAULT_VARIANTS]
    variants = []
    for value in values:
        label, sep, checkpoint = value.partition(":")
        if not sep or not label:
            raise ValueError(f"Invalid --variant '{value}', expected label:path")
        variants.append(Variant(label, checkpoint or None))
    return variants


def ade_label_to_train_id(mask: Image.Image) -> np.ndarray:
    array = np.array(mask, dtype=np.int64)
    target = np.full_like(array, 255, dtype=np.int64)
    valid = (array >= 1) & (array <= 150)
    target[valid] = array[valid] - 1
    return target


def colorize_mask(mask: np.ndarray, num_classes: int = 150) -> Image.Image:
    rng = np.random.default_rng(0)
    palette = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < num_classes)
    out[valid] = palette[mask[valid]]
    return Image.fromarray(out)


def normalize_image_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TVF.to_tensor(image)
    return TVF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def resize_and_center_crop(
    image: Image.Image,
    mask: Image.Image,
    image_size: int,
    crop_size: int,
) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    if width < height:
        new_w = image_size
        new_h = int(round(height * image_size / width))
    else:
        new_h = image_size
        new_w = int(round(width * image_size / height))
    image = image.resize((new_w, new_h), Image.BICUBIC)
    mask = mask.resize((new_w, new_h), Image.NEAREST)
    pad_w = max(crop_size - new_w, 0)
    pad_h = max(crop_size - new_h, 0)
    if pad_w or pad_h:
        image = TVF.pad(image, [0, 0, pad_w, pad_h], fill=0)
        mask = TVF.pad(mask, [0, 0, pad_w, pad_h], fill=0)
        new_w, new_h = image.size
    left = (new_w - crop_size) // 2
    top = (new_h - crop_size) // 2
    return (
        image.crop((left, top, left + crop_size, top + crop_size)),
        mask.crop((left, top, left + crop_size, top + crop_size)),
    )


def load_samples(data_root: Path, indices: list[int], image_size: int, crop_size: int):
    image_dir = data_root / "images" / "validation"
    ann_dir = data_root / "annotations" / "validation"
    samples = []
    all_images = sorted(image_dir.glob("*.jpg"))
    for index in indices:
        image_path = all_images[index]
        mask_path = ann_dir / f"{image_path.stem}.png"
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        image, mask = resize_and_center_crop(image, mask, image_size, crop_size)
        tensor = normalize_image_tensor(image)
        target = ade_label_to_train_id(mask)
        samples.append((image_path.stem, image, target, tensor))
    return samples


def build_encoder(encoder_name: str, checkpoint: str | None, device: torch.device) -> torch.nn.Module:
    size = encoder_name.split("_", 1)[1]
    model_name = SIZE_TO_MODEL[size]
    builder = _load_dinov3_backbone_builder(model_name)
    model = builder(pretrained=False)
    weights_path = _resolve_weights_path(model_name, checkpoint)
    state_dict = _load_state_dict(weights_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def extract_feature(model: torch.nn.Module, image: torch.Tensor, layer: int, device: torch.device) -> torch.Tensor:
    features = model.get_intermediate_layers(
        image[None].to(device),
        n=[layer],
        reshape=True,
        return_class_token=False,
    )
    feature = features[0] if isinstance(features, (tuple, list)) else features
    return feature[0].float().cpu()


def to_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def percentile_normalize(values: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    low = np.percentile(values, lo)
    high = np.percentile(values, hi)
    return np.clip((values - low) / max(high - low, 1e-6), 0.0, 1.0)


def pca_rgb(features: list[torch.Tensor]) -> list[Image.Image]:
    tokens = []
    shapes = []
    for feature in features:
        c, h, w = feature.shape
        shapes.append((h, w))
        tokens.append(feature.permute(1, 2, 0).reshape(-1, c))
    matrix = torch.cat(tokens, dim=0)
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    projected = matrix @ vh[:3].T
    projected_np = projected.numpy()
    projected_np = percentile_normalize(projected_np, 1.0, 99.0)

    images = []
    offset = 0
    for h, w in shapes:
        count = h * w
        rgb = projected_np[offset : offset + count].reshape(h, w, 3)
        offset += count
        images.append(Image.fromarray(to_uint8(rgb), mode="RGB"))
    return images


def heatmap_color(values: np.ndarray) -> Image.Image:
    x = np.clip(values, 0.0, 1.0)
    stops = np.array(
        [
            [0.05, 0.05, 0.35],
            [0.10, 0.45, 0.95],
            [0.20, 0.80, 0.70],
            [0.95, 0.90, 0.25],
            [0.90, 0.10, 0.10],
        ],
        dtype=np.float32,
    )
    position = x * (len(stops) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    weight = position - lower
    rgb = stops[lower] * (1.0 - weight[..., None]) + stops[upper] * weight[..., None]
    return Image.fromarray(to_uint8(rgb), mode="RGB")


def feature_norm_heatmaps(features: list[torch.Tensor]) -> list[Image.Image]:
    norms = [feature.norm(dim=0).numpy() for feature in features]
    stacked = np.concatenate([norm.reshape(-1) for norm in norms])
    low = np.percentile(stacked, 1.0)
    high = np.percentile(stacked, 99.0)
    return [heatmap_color(np.clip((norm - low) / max(high - low, 1e-6), 0.0, 1.0)) for norm in norms]


def change_heatmaps(features: list[torch.Tensor]) -> list[Image.Image]:
    base = F.normalize(features[0], dim=0)
    changes = []
    for feature in features:
        current = F.normalize(feature, dim=0)
        change = (1.0 - (base * current).sum(dim=0)).clamp_min(0.0).numpy()
        changes.append(change)
    stacked = np.concatenate([change.reshape(-1) for change in changes[1:]]) if len(changes) > 1 else changes[0].reshape(-1)
    high = np.percentile(stacked, 99.0) if stacked.size else 1.0
    return [heatmap_color(np.clip(change / max(high, 1e-6), 0.0, 1.0)) for change in changes]


def add_title(image: Image.Image, title: str, height: int = 26) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title, fill=(0, 0, 0))
    canvas.paste(image, (0, height))
    return canvas


def make_montage(
    sample_name: str,
    image: Image.Image,
    target: np.ndarray,
    variants: list[Variant],
    features: list[torch.Tensor],
    output_dir: Path,
) -> None:
    pca_images = pca_rgb(features)
    norm_images = feature_norm_heatmaps(features)
    change_images = change_heatmaps(features)
    gt = colorize_mask(target)
    size = image.size

    rows = []
    for variant, pca, norm, change in zip(variants, pca_images, norm_images, change_images):
        cols = [
            add_title(image.copy(), f"{variant.label}: image"),
            add_title(gt.copy(), "ADE20K GT"),
            add_title(pca.resize(size, Image.BILINEAR), "shared PCA RGB"),
            add_title(norm.resize(size, Image.BILINEAR), "feature norm"),
            add_title(change.resize(size, Image.BILINEAR), "change vs base"),
        ]
        row = Image.new("RGB", (sum(col.width for col in cols), max(col.height for col in cols)), "white")
        x = 0
        for col in cols:
            row.paste(col, (x, 0))
            x += col.width
        rows.append(row)

    montage = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        montage.paste(row, (0, y))
        y += row.height
    montage.save(output_dir / f"{sample_name}_feature_compare.jpg", quality=92)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = parse_variants(args.variant)
    samples = load_samples(Path(args.data_root).expanduser(), args.samples, args.image_size, args.crop_size)

    encoders = []
    for variant in variants:
        print(f"loading {variant.label}: {variant.checkpoint}", flush=True)
        encoders.append(build_encoder(args.encoder, variant.checkpoint, device))

    for sample_name, image, target, tensor in samples:
        print(f"visualizing {sample_name}", flush=True)
        features = [extract_feature(encoder, tensor, args.layer, device) for encoder in encoders]
        make_montage(sample_name, image, target, variants, features, output_dir)

    print(f"saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
