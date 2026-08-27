#!/usr/bin/env python3
"""Small COCO one-shot segmentation probe using frozen DINOv3 features.

The probe builds semantic episodes from COCO instance masks: one train image is
used as the annotated reference for a category, and validation images containing
that category are targets.  It is intended to compare raw and positionally
debiased features, not to reproduce the full INSID3 benchmark protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
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
from rfdetr.models.insid3 import (  # noqa: E402
    DinoV3InContextSegmenter,
    agglomerative_cluster_labels,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("DINOv3 INSID3-style COCO one-shot probe")
    parser.add_argument(
        "--data-root",
        default="/data/cpc/root/dataset/COCO_RFDETR_TEST/medium",
    )
    parser.add_argument("--output-dir", default="output/insid3_coco_probe/dinov3_small")
    parser.add_argument("--model-size", default="small", choices=tuple(SIZE_TO_MODEL))
    parser.add_argument("--pretrained-encoder", default=None)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["person", "car", "dog", "cat", "bicycle"],
    )
    parser.add_argument("--targets-per-category", type=int, default=6)
    parser.add_argument("--svd-components", type=int, nargs="+", default=[0, 32, 64, 128])
    parser.add_argument("--cluster-similarity", type=float, default=0.6)
    parser.add_argument("--merge-threshold", type=float, default=0.2)
    parser.add_argument("--min-mask-fraction", type=float, default=0.01)
    parser.add_argument("--max-mask-fraction", type=float, default=0.75)
    parser.add_argument("--visualize-per-category", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_encoder(args: argparse.Namespace) -> torch.nn.Module:
    model_name = SIZE_TO_MODEL[args.model_size]
    builder = _load_dinov3_backbone_builder(model_name)
    encoder = builder(pretrained=False)
    weights_path = _resolve_weights_path(model_name, args.pretrained_encoder)
    encoder.load_state_dict(_load_state_dict(weights_path), strict=True)
    return encoder.eval()


def category_union_mask(coco: COCO, image_id: int, category_id: int) -> np.ndarray:
    image_info = coco.loadImgs([image_id])[0]
    mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
    annotation_ids = coco.getAnnIds(
        imgIds=[image_id], catIds=[category_id], iscrowd=False
    )
    for annotation in coco.loadAnns(annotation_ids):
        mask |= coco.annToMask(annotation).astype(np.uint8)
    return mask


def mask_fraction(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else 0.0


def choose_reference(
    coco: COCO,
    category_id: int,
    min_fraction: float,
    max_fraction: float,
) -> tuple[int, np.ndarray]:
    candidates = []
    for image_id in coco.getImgIds(catIds=[category_id]):
        mask = category_union_mask(coco, image_id, category_id)
        fraction = mask_fraction(mask)
        if min_fraction <= fraction <= max_fraction:
            candidates.append((abs(fraction - 0.2), image_id, mask))
    if not candidates:
        raise RuntimeError(f"No suitable reference found for category id {category_id}.")
    _, image_id, mask = min(candidates, key=lambda item: (item[0], item[1]))
    return image_id, mask


def choose_targets(
    coco: COCO,
    category_id: int,
    count: int,
    min_fraction: float,
    max_fraction: float,
    rng: random.Random,
) -> list[tuple[int, np.ndarray]]:
    candidates = []
    for image_id in coco.getImgIds(catIds=[category_id]):
        mask = category_union_mask(coco, image_id, category_id)
        fraction = mask_fraction(mask)
        if min_fraction <= fraction <= max_fraction:
            candidates.append((image_id, mask))
    rng.shuffle(candidates)
    return candidates[:count]


def load_square_image(
    image_path: Path,
    image_size: int,
) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    tensor = TVF.normalize(TVF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
    return tensor, image


def resize_mask(mask: np.ndarray, image_size: int) -> torch.Tensor:
    mask_image = Image.fromarray(mask * 255)
    resized = mask_image.resize(
        (image_size, image_size), Image.Resampling.NEAREST
    )
    return torch.from_numpy(np.asarray(resized, dtype=np.uint8).copy()) > 127


def intersection_over_union(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.bool()
    target = target.bool()
    intersection = (prediction & target).sum().item()
    union = (prediction | target).sum().item()
    return float(intersection / union) if union else 1.0


def upsample_patch_mask(mask: torch.Tensor, image_size: int) -> torch.Tensor:
    upsampled = F.interpolate(
        mask[None, None].float(),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return upsampled > 0.5


def overlay(image: Image.Image, mask: torch.Tensor, color: tuple[int, int, int]) -> Image.Image:
    base = np.asarray(image, dtype=np.uint8).copy()
    mask_array = mask.cpu().numpy().astype(bool)
    color_array = np.asarray(color, dtype=np.float32)
    base[mask_array] = (0.55 * base[mask_array] + 0.45 * color_array).astype(np.uint8)
    return Image.fromarray(base)


def save_visualization(
    path: Path,
    reference_image: Image.Image,
    reference_mask: torch.Tensor,
    target_image: Image.Image,
    target_mask: torch.Tensor,
    prediction: torch.Tensor,
) -> None:
    panels = [
        overlay(reference_image, reference_mask, (40, 220, 80)),
        target_image,
        overlay(target_image, target_mask, (40, 220, 80)),
        overlay(target_image, prediction, (240, 70, 50)),
    ]
    canvas = Image.new("RGB", (reference_image.width * len(panels), reference_image.height))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * reference_image.width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def main() -> None:
    args = parse_args()
    if args.image_size % 16:
        raise ValueError("--image-size must be divisible by the DINOv3 patch size (16).")
    if args.targets_per_category < 1:
        raise ValueError("--targets-per-category must be positive.")
    if not 0 < args.min_mask_fraction < args.max_mask_fraction <= 1:
        raise ValueError("mask fraction bounds must satisfy 0 < min < max <= 1.")
    components = sorted(set(args.svd_components))
    if not components or components[0] < 0:
        raise ValueError("--svd-components must contain non-negative integers.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root).expanduser().resolve()
    train_coco = COCO(str(data_root / "annotations" / "instances_train2017.json"))
    val_coco = COCO(str(data_root / "annotations" / "instances_val2017.json"))

    encoder = build_encoder(args).to(device)
    segmenter = DinoV3InContextSegmenter(encoder).to(device)
    max_components = max(components)
    positional_basis = segmenter.build_positional_basis(
        (args.image_size, args.image_size),
        max_components,
        device=device,
    )
    if max_components > positional_basis.shape[1]:
        raise ValueError(
            f"Requested {max_components} debias components, but the {args.model_size} "
            f"encoder supports at most {positional_basis.shape[1]}."
        )

    category_by_name = {
        category["name"]: category for category in train_coco.loadCats(train_coco.getCatIds())
    }
    rows = []
    rng = random.Random(args.seed)

    for category_name in args.categories:
        if category_name not in category_by_name:
            raise ValueError(f"Unknown COCO category: {category_name}")
        category_id = int(category_by_name[category_name]["id"])
        reference_id, reference_mask_np = choose_reference(
            train_coco,
            category_id,
            args.min_mask_fraction,
            args.max_mask_fraction,
        )
        targets = choose_targets(
            val_coco,
            category_id,
            args.targets_per_category,
            args.min_mask_fraction,
            args.max_mask_fraction,
            rng,
        )
        if not targets:
            raise RuntimeError(f"No target images found for {category_name}.")

        reference_path = data_root / "train2017" / f"{reference_id:012d}.jpg"
        reference_tensor, reference_image = load_square_image(
            reference_path, args.image_size
        )
        reference_mask = resize_mask(reference_mask_np, args.image_size)

        for target_index, (target_id, target_mask_np) in enumerate(targets):
            target_path = data_root / "val2017" / f"{target_id:012d}.jpg"
            target_tensor, target_image = load_square_image(target_path, args.image_size)
            target_mask = resize_mask(target_mask_np, args.image_size)

            batch = torch.stack([reference_tensor, target_tensor]).to(device)
            features = segmenter.extract_features(batch)
            reference_features, target_features = features[0], features[1]
            target_flat = target_features.flatten(1).T
            cluster_labels = agglomerative_cluster_labels(
                target_flat, args.cluster_similarity
            )

            for component_count in components:
                result = segmenter.predict_from_features(
                    reference_features,
                    reference_mask.to(device),
                    target_features,
                    positional_basis=positional_basis,
                    num_debias_components=component_count,
                    cluster_similarity_threshold=args.cluster_similarity,
                    merge_threshold=args.merge_threshold,
                    cluster_labels=cluster_labels,
                )
                prediction = upsample_patch_mask(result.mask, args.image_size).cpu()
                iou = intersection_over_union(prediction, target_mask)
                rows.append(
                    {
                        "category": category_name,
                        "category_id": category_id,
                        "reference_id": reference_id,
                        "target_id": target_id,
                        "target_index": target_index,
                        "svd_components": component_count,
                        "iou": iou,
                        "num_clusters": result.num_clusters,
                        "seed_cluster": result.seed_cluster,
                        "candidate_patches": result.num_candidate_patches,
                    }
                )
                print(
                    f"category={category_name} target={target_id} svd={component_count} "
                    f"iou={iou:.4f} clusters={result.num_clusters} "
                    f"candidates={result.num_candidate_patches}",
                    flush=True,
                )

                if target_index < args.visualize_per_category:
                    save_visualization(
                        output_dir
                        / "visualizations"
                        / f"{category_name}_{target_id}_svd{component_count}.jpg",
                        reference_image,
                        reference_mask,
                        target_image,
                        target_mask,
                        prediction,
                    )

    episode_csv = output_dir / "episodes.csv"
    with episode_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["svd_components"], row["category"])].append(row["iou"])
    per_category = [
        {
            "svd_components": components_count,
            "category": category,
            "mean_iou": float(np.mean(values)),
            "episodes": len(values),
        }
        for (components_count, category), values in sorted(grouped.items())
    ]
    overall = []
    for component_count in components:
        values = [row["iou"] for row in rows if row["svd_components"] == component_count]
        overall.append(
            {
                "svd_components": component_count,
                "mean_iou": float(np.mean(values)),
                "std_iou": float(np.std(values)),
                "episodes": len(values),
            }
        )

    summary = {
        "scope": (
            "Small diagnostic on COCO semantic episodes; this is not the official "
            "INSID3 fold protocol."
        ),
        "args": vars(args),
        "weights": str(_resolve_weights_path(SIZE_TO_MODEL[args.model_size], args.pretrained_encoder)),
        "positional_basis_shape": list(positional_basis.shape),
        "overall": overall,
        "per_category": per_category,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(overall, indent=2), flush=True)
    print(output_dir, flush=True)


if __name__ == "__main__":
    main()

