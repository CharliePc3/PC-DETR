#!/usr/bin/env python3
"""LazyStrike-style offline refinement for DINOv3 backbones.

This script refines only the ViT encoder weights. It does not add a module to
RF-DETR inference. The refined checkpoint can be loaded later with
`--pretrained-encoder`.

The objective is intentionally conservative:

* box-balanced CLS aggregation: every annotated object contributes one local
  prototype, so large objects do not dominate the CLS target;
* soft object coverage: each box should contain at least one high-response
  patch under CLS-patch cosine;
* horizontal-flip CLS consistency;
* frozen-teacher dense-token preservation to reduce representation drift.

Optional detection-aware dense refinement extends the objective to intermediate
patch features used by RF-DETR. It uses soft patch/box overlap, box-balanced
cross-view consistency, and multi-level frozen-teacher preservation. These
options default to disabled so existing LazyStrike recipes remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TVF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

os.environ.setdefault("DINOV3_REPO_DIR", "/data/cpc/root/project/DINOv3")
os.environ.setdefault("DINOV3_WEIGHTS_DIR", str(PROJECT_ROOT / "weights" / "dinov3"))

import analyze_dinov3_tokens as diag  # noqa: E402
from rfdetr.models.backbone.dinov3 import SIZE_TO_MODEL  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class CocoImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    bboxes_xywh: list[tuple[float, float, float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("LazyStrike-style DINOv3 refinement")
    parser.add_argument("--data-root", default="/data/cpc/root/dataset/COCO")
    parser.add_argument("--split", default="train2017")
    parser.add_argument("--ann-file", default=None)
    parser.add_argument("--output-dir", default="output/lazystrike_refine/dinov3_small_coco5k_v1")
    parser.add_argument("--encoder", default="dinov3_small", choices=tuple(f"dinov3_{k}" for k in SIZE_TO_MODEL))
    parser.add_argument("--pretrained-encoder", default=None)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--train-last-blocks", type=int, default=4)
    parser.add_argument("--train-block-indexes", type=int, nargs="+", default=None)
    parser.add_argument("--train-block-lr-scales", type=float, nargs="+", default=None)
    parser.add_argument("--freeze-final-norm", action="store_true")
    parser.add_argument("--max-boxes-per-image", type=int, default=12)
    parser.add_argument("--min-box-patches", type=int, default=1)
    parser.add_argument("--lazy-topk", type=int, default=1)
    parser.add_argument("--lazy-target-weight", type=float, default=0.35)
    parser.add_argument("--cover-margin", type=float, default=0.20)
    parser.add_argument("--cover-temperature", type=float, default=0.10)
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--lambda-cover", type=float, default=0.5)
    parser.add_argument("--lambda-consistency", type=float, default=0.25)
    parser.add_argument("--lambda-distill", type=float, default=0.2)
    parser.add_argument("--dense-layers", type=int, nargs="+", default=None)
    parser.add_argument("--dense-layer-weights", type=float, nargs="+", default=None)
    parser.add_argument("--dense-object-layer-weights", type=float, nargs="+", default=None)
    parser.add_argument("--lambda-dense-object", type=float, default=0.0)
    parser.add_argument("--lambda-dense-global", type=float, default=0.0)
    parser.add_argument("--dense-min-overlap", type=float, default=0.0)
    parser.add_argument("--layer-lr-decay", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, model: torch.nn.Module) -> None:
    num_blocks = len(model.blocks)
    if args.train_block_indexes is not None:
        if args.train_block_indexes != sorted(set(args.train_block_indexes)):
            raise ValueError("--train-block-indexes must be unique and in ascending order.")
        if args.train_block_indexes[0] < 0 or args.train_block_indexes[-1] >= num_blocks:
            raise ValueError(f"--train-block-indexes must be in [0, {num_blocks - 1}].")
    if args.train_block_lr_scales is not None:
        if args.train_block_indexes is None:
            raise ValueError("--train-block-lr-scales requires --train-block-indexes.")
        if len(args.train_block_lr_scales) != len(args.train_block_indexes):
            raise ValueError(
                "--train-block-lr-scales must match --train-block-indexes."
            )
        if any(scale <= 0 for scale in args.train_block_lr_scales):
            raise ValueError("--train-block-lr-scales values must be positive.")
    if not 0 < args.layer_lr_decay <= 1:
        raise ValueError("--layer-lr-decay must be in (0, 1].")
    if not 0 <= args.dense_min_overlap < 1:
        raise ValueError("--dense-min-overlap must be in [0, 1).")
    if args.lambda_dense_object < 0 or args.lambda_dense_global < 0:
        raise ValueError("Dense loss weights must be non-negative.")

    dense_enabled = args.lambda_dense_object > 0 or args.lambda_dense_global > 0
    if dense_enabled and not args.dense_layers:
        raise ValueError("--dense-layers is required when a dense loss is enabled.")
    if not args.dense_layers:
        if args.dense_layer_weights is not None or args.dense_object_layer_weights is not None:
            raise ValueError("Dense layer weights require --dense-layers.")
        return

    if args.dense_layers != sorted(set(args.dense_layers)):
        raise ValueError("--dense-layers must be unique and in ascending order.")
    if args.dense_layers[0] < 0 or args.dense_layers[-1] >= num_blocks:
        raise ValueError(f"--dense-layers must be in [0, {num_blocks - 1}].")
    if args.dense_layers[-1] != num_blocks - 1:
        raise ValueError("The final dense layer must be the encoder's last block for CLS losses.")
    if args.dense_layer_weights is None:
        args.dense_layer_weights = [1.0 / len(args.dense_layers)] * len(args.dense_layers)
    elif len(args.dense_layer_weights) != len(args.dense_layers):
        raise ValueError("--dense-layer-weights must match --dense-layers.")
    if any(weight < 0 for weight in args.dense_layer_weights) or sum(args.dense_layer_weights) <= 0:
        raise ValueError("Dense layer weights must be non-negative with a positive sum.")
    weight_sum = sum(args.dense_layer_weights)
    args.dense_layer_weights = [weight / weight_sum for weight in args.dense_layer_weights]

    if args.dense_object_layer_weights is None:
        args.dense_object_layer_weights = list(args.dense_layer_weights)
    elif len(args.dense_object_layer_weights) != len(args.dense_layers):
        raise ValueError("--dense-object-layer-weights must match --dense-layers.")
    if any(weight < 0 for weight in args.dense_object_layer_weights) or sum(
        args.dense_object_layer_weights
    ) <= 0:
        raise ValueError("Dense object layer weights must be non-negative with a positive sum.")
    object_weight_sum = sum(args.dense_object_layer_weights)
    args.dense_object_layer_weights = [
        weight / object_weight_sum for weight in args.dense_object_layer_weights
    ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_coco_records(data_root: Path, split: str, ann_file: Path) -> list[CocoImageRecord]:
    with ann_file.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images = {int(img["id"]): img for img in coco.get("images", [])}
    boxes_by_image: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
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
        if not (data_root / split / file_name).exists() and not (data_root / file_name).exists():
            continue
        records.append(
            CocoImageRecord(
                image_id=image_id,
                file_name=file_name,
                width=int(image["width"]),
                height=int(image["height"]),
                bboxes_xywh=boxes,
            )
        )
    return records


class CocoBoxDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        ann_file: Path,
        resolution: int,
        num_images: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        self.data_root = data_root
        self.split = split
        self.resolution = resolution
        records = load_coco_records(data_root, split, ann_file)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(records)
        else:
            records = sorted(records, key=lambda record: record.image_id)
        self.records = records[:num_images] if num_images > 0 else records
        if not self.records:
            raise RuntimeError("No valid COCO images with boxes were found.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image_path = self.data_root / self.split / record.file_name
        if not image_path.exists():
            image_path = self.data_root / record.file_name
        image = Image.open(image_path).convert("RGB").resize((self.resolution, self.resolution), Image.BICUBIC)
        tensor = TVF.to_tensor(image)
        tensor = TVF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        boxes = diag.boxes_to_resized_xyxy(record.bboxes_xywh, record.width, record.height, self.resolution)
        return {
            "image": tensor,
            "boxes": boxes,
            "image_id": record.image_id,
            "file_name": record.file_name,
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "boxes": [item["boxes"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "file_names": [item["file_name"] for item in batch],
    }


def build_encoder(encoder: str, pretrained_encoder: str | None, device: torch.device) -> torch.nn.Module:
    model_args = argparse.Namespace(encoder=encoder, pretrained_encoder=pretrained_encoder)
    return diag.build_dinov3_encoder(model_args).to(device)


def set_trainable_layers(
    model: torch.nn.Module,
    train_last_blocks: int,
    train_block_indexes: list[int] | None,
    freeze_final_norm: bool,
) -> list[str]:
    for param in model.parameters():
        param.requires_grad = False

    trainable_names = []
    if train_block_indexes is not None:
        block_indexes = train_block_indexes
    else:
        blocks = list(model.blocks)
        block_indexes = list(range(max(0, len(blocks) - train_last_blocks), len(blocks)))

    for block_idx in block_indexes:
        for param in model.blocks[block_idx].parameters():
            param.requires_grad = True
        trainable_names.append(f"blocks.{block_idx}")

    for module_name in (() if freeze_final_norm else ("norm", "fc_norm")):
        if hasattr(model, module_name):
            module = getattr(model, module_name)
            for param in module.parameters():
                param.requires_grad = True
            trainable_names.append(module_name)

    if not any(param.requires_grad for param in model.parameters()):
        raise RuntimeError("No trainable parameters. Configure trainable encoder blocks.")
    return trainable_names


def build_optimizer_param_groups(model: torch.nn.Module, args: argparse.Namespace) -> list[dict]:
    """Apply explicit block scales or layer-wise decay to trainable parameters."""
    explicit_scales = None
    if args.train_block_lr_scales is not None:
        explicit_scales = dict(zip(args.train_block_indexes, args.train_block_lr_scales))

    if args.layer_lr_decay == 1.0 and explicit_scales is None:
        return [
            {
                "params": [param for param in model.parameters() if param.requires_grad],
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            }
        ]

    last_block_idx = len(model.blocks) - 1
    groups: dict[float, list[torch.nn.Parameter]] = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = args.lr
        if name.startswith("blocks."):
            block_idx = int(name.split(".")[1])
            if explicit_scales is not None:
                lr *= explicit_scales[block_idx]
            else:
                lr *= args.layer_lr_decay ** (last_block_idx - block_idx)
        groups[lr].append(param)
    return [
        {"params": params, "lr": lr, "weight_decay": args.weight_decay}
        for lr, params in sorted(groups.items())
    ]


def extract_multilevel_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    layers: list[int],
) -> dict[int, dict[str, torch.Tensor]]:
    outputs = model.get_intermediate_layers(
        images,
        n=layers,
        reshape=False,
        return_class_token=True,
    )
    return {
        layer: {
            "x_norm_patchtokens": patch_tokens,
            "x_norm_clstoken": cls_token,
        }
        for layer, (patch_tokens, cls_token) in zip(layers, outputs)
    }


def flip_patch_tokens_back(tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    batch_size, num_patches, channels = tokens.shape
    if num_patches != grid_h * grid_w:
        raise ValueError(f"Expected {grid_h * grid_w} patch tokens, got {num_patches}.")
    grid = tokens.reshape(batch_size, grid_h, grid_w, channels)
    return torch.flip(grid, dims=[2]).reshape(batch_size, num_patches, channels)


def patch_boxes_xyxy(
    grid_h: int,
    grid_w: int,
    patch_size: int,
    device: torch.device,
) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(grid_h, device=device),
        torch.arange(grid_w, device=device),
        indexing="ij",
    )
    x1 = xs.reshape(-1).float() * patch_size
    y1 = ys.reshape(-1).float() * patch_size
    return torch.stack((x1, y1, x1 + patch_size, y1 + patch_size), dim=-1)


def select_size_balanced_boxes(boxes: torch.Tensor, max_boxes: int) -> torch.Tensor:
    if boxes.shape[0] <= max_boxes:
        return boxes
    areas = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    order = torch.argsort(areas)
    positions = torch.linspace(0, boxes.shape[0] - 1, max_boxes, device=boxes.device).round().long()
    return boxes[order[positions]]


def soft_patch_box_weights(
    patch_boxes: torch.Tensor,
    boxes: torch.Tensor,
    min_overlap: float,
) -> torch.Tensor:
    """Return normalized patch-area overlap weights with shape [patches, boxes]."""
    if boxes.numel() == 0:
        return patch_boxes.new_zeros((patch_boxes.shape[0], 0))
    inter_x1 = torch.maximum(patch_boxes[:, None, 0], boxes[None, :, 0])
    inter_y1 = torch.maximum(patch_boxes[:, None, 1], boxes[None, :, 1])
    inter_x2 = torch.minimum(patch_boxes[:, None, 2], boxes[None, :, 2])
    inter_y2 = torch.minimum(patch_boxes[:, None, 3], boxes[None, :, 3])
    intersection = (inter_x2 - inter_x1).clamp_min(0) * (inter_y2 - inter_y1).clamp_min(0)
    patch_area = (patch_boxes[:, 2] - patch_boxes[:, 0]) * (patch_boxes[:, 3] - patch_boxes[:, 1])
    weights = intersection / patch_area[:, None].clamp_min(1e-6)
    if min_overlap > 0:
        weights = torch.where(weights >= min_overlap, weights, torch.zeros_like(weights))
    valid = weights.sum(dim=0) > 0
    return weights[:, valid]


def patch_centers_for_model(model: torch.nn.Module, resolution: int, device: torch.device) -> tuple[torch.Tensor, int, int]:
    patch_size = int(model.patch_size)
    grid_h = resolution // patch_size
    grid_w = resolution // patch_size
    centers = diag.patch_centers(grid_h, grid_w, patch_size, device)
    return centers, grid_h, grid_w


def select_boxes(
    boxes: torch.Tensor,
    centers: torch.Tensor,
    max_boxes: int,
    min_box_patches: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    inside = diag.point_in_boxes(centers, boxes)
    valid = inside.sum(dim=0) >= min_box_patches
    if valid.any():
        boxes = boxes[valid]
        inside = inside[:, valid]
    else:
        return boxes.new_zeros((0, 4)), inside[:, :0]

    if boxes.shape[0] > max_boxes:
        areas = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
        keep = torch.argsort(areas, descending=True)[:max_boxes]
        boxes = boxes[keep]
        inside = inside[:, keep]
    return boxes, inside


def logsumexp_box_scores(
    score: torch.Tensor,
    inside_by_box: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    values = []
    for box_idx in range(inside_by_box.shape[1]):
        box_scores = score[inside_by_box[:, box_idx]]
        if box_scores.numel() == 0:
            continue
        values.append(temperature * torch.logsumexp(box_scores.float() / temperature, dim=0))
    if not values:
        return score.new_empty((0,))
    return torch.stack(values)


def box_balanced_prototype(
    tokens: torch.Tensor,
    inside_by_box: torch.Tensor,
) -> torch.Tensor | None:
    prototypes = []
    for box_idx in range(inside_by_box.shape[1]):
        box_tokens = tokens[inside_by_box[:, box_idx]]
        if box_tokens.numel() == 0:
            continue
        prototypes.append(F.normalize(box_tokens.mean(dim=0).float(), dim=-1))
    if not prototypes:
        return None
    return F.normalize(torch.stack(prototypes, dim=0).mean(dim=0), dim=-1)


def lazy_cls_target(tokens: torch.Tensor, topk: int) -> torch.Tensor:
    _votes, _selected, lazy_cls = diag.compute_last_vote_score(tokens, topk=topk)
    return lazy_cls


def compute_losses(
    student_features: dict,
    teacher_features: dict,
    boxes_list: list[torch.Tensor],
    centers: torch.Tensor,
    args: argparse.Namespace,
    distill_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    cls_tokens = F.normalize(student_features["x_norm_clstoken"].float(), dim=-1)
    patch_tokens = F.normalize(student_features["x_norm_patchtokens"].float(), dim=-1)
    teacher_tokens = F.normalize(teacher_features["x_norm_patchtokens"].float(), dim=-1)

    align_losses = []
    cover_losses = []
    valid_images = 0

    for batch_idx, boxes in enumerate(boxes_list):
        boxes = boxes.to(device=patch_tokens.device, non_blocking=True)
        _boxes, inside = select_boxes(boxes, centers, args.max_boxes_per_image, args.min_box_patches)
        if inside.shape[1] == 0:
            continue
        valid_images += 1
        tokens_i = patch_tokens[batch_idx]
        cls_i = cls_tokens[batch_idx]

        box_proto = box_balanced_prototype(tokens_i, inside)
        if box_proto is not None:
            lazy_proto = lazy_cls_target(tokens_i, args.lazy_topk)
            target = F.normalize(
                (1.0 - args.lazy_target_weight) * box_proto + args.lazy_target_weight * lazy_proto,
                dim=-1,
            ).detach()
            align_losses.append(1.0 - torch.sum(cls_i * target))

        patch_score = tokens_i @ cls_i
        box_scores = logsumexp_box_scores(patch_score, inside, args.cover_temperature)
        if box_scores.numel() > 0:
            cover_losses.append(F.softplus(args.cover_margin - box_scores).mean())

    device = patch_tokens.device
    zero = patch_tokens.new_tensor(0.0)
    align_loss = torch.stack(align_losses).mean() if align_losses else zero
    cover_loss = torch.stack(cover_losses).mean() if cover_losses else zero
    distill_loss = (
        distill_override
        if distill_override is not None
        else 1.0 - (patch_tokens * teacher_tokens).sum(dim=-1).mean()
    )

    total = (
        args.lambda_align * align_loss
        + args.lambda_cover * cover_loss
        + args.lambda_distill * distill_loss
    )
    stats = {
        "loss_align": float(align_loss.detach().item()),
        "loss_cover": float(cover_loss.detach().item()),
        "loss_distill": float(distill_loss.detach().item()),
        "valid_images": float(valid_images),
    }
    return total, stats


def compute_multilevel_dense_losses(
    student_levels: dict[int, dict[str, torch.Tensor]],
    teacher_levels: dict[int, dict[str, torch.Tensor]],
    flipped_student_levels: dict[int, dict[str, torch.Tensor]],
    boxes_list: list[torch.Tensor],
    patch_boxes: torch.Tensor,
    grid_h: int,
    grid_w: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    weighted_distill = []
    weighted_object = []
    weighted_global = []
    stats: dict[str, float] = {}

    for layer, layer_weight, object_layer_weight in zip(
        args.dense_layers,
        args.dense_layer_weights,
        args.dense_object_layer_weights,
    ):
        student_tokens = F.normalize(student_levels[layer]["x_norm_patchtokens"].float(), dim=-1)
        teacher_tokens = F.normalize(teacher_levels[layer]["x_norm_patchtokens"].float(), dim=-1)
        flipped_tokens = F.normalize(flipped_student_levels[layer]["x_norm_patchtokens"].float(), dim=-1)
        flipped_tokens = flip_patch_tokens_back(flipped_tokens, grid_h, grid_w)

        distill = 1.0 - (student_tokens * teacher_tokens).sum(dim=-1).mean()
        cross_view_distance = 1.0 - (flipped_tokens * teacher_tokens).sum(dim=-1)
        global_dense = cross_view_distance.mean()

        per_image_object = []
        for batch_idx, boxes in enumerate(boxes_list):
            boxes = boxes.to(device=student_tokens.device, non_blocking=True)
            boxes = select_size_balanced_boxes(boxes, args.max_boxes_per_image)
            weights = soft_patch_box_weights(patch_boxes, boxes, args.dense_min_overlap)
            if weights.shape[1] == 0:
                continue
            normalized_weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
            per_box = (cross_view_distance[batch_idx, :, None] * normalized_weights).sum(dim=0)
            per_image_object.append(per_box.mean())
        object_dense = (
            torch.stack(per_image_object).mean()
            if per_image_object
            else student_tokens.new_tensor(0.0)
        )

        weighted_distill.append(layer_weight * distill)
        weighted_object.append(object_layer_weight * object_dense)
        weighted_global.append(layer_weight * global_dense)
        stats[f"loss_distill_l{layer}"] = float(distill.detach().item())
        stats[f"loss_dense_object_l{layer}"] = float(object_dense.detach().item())
        stats[f"loss_dense_global_l{layer}"] = float(global_dense.detach().item())
        stats[f"teacher_cosine_l{layer}"] = 1.0 - stats[f"loss_distill_l{layer}"]
        stats[f"cross_view_cosine_l{layer}"] = 1.0 - stats[f"loss_dense_global_l{layer}"]

    return (
        torch.stack(weighted_distill).sum(),
        torch.stack(weighted_object).sum(),
        torch.stack(weighted_global).sum(),
        stats,
    )


def compute_flip_consistency_loss(
    model: torch.nn.Module,
    images: torch.Tensor,
    cls_tokens: torch.Tensor,
) -> torch.Tensor:
    flipped_features = model.forward_features(torch.flip(images, dims=[-1]))
    flipped_cls = F.normalize(flipped_features["x_norm_clstoken"].float(), dim=-1)
    cls_tokens = F.normalize(cls_tokens.float(), dim=-1)
    return 1.0 - (cls_tokens * flipped_cls).sum(dim=-1).mean()


def compute_cls_consistency_from_levels(
    student_levels: dict[int, dict[str, torch.Tensor]],
    flipped_student_levels: dict[int, dict[str, torch.Tensor]],
    final_layer: int,
) -> torch.Tensor:
    cls_tokens = F.normalize(student_levels[final_layer]["x_norm_clstoken"].float(), dim=-1)
    flipped_cls = F.normalize(flipped_student_levels[final_layer]["x_norm_clstoken"].float(), dim=-1)
    return 1.0 - (cls_tokens * flipped_cls).sum(dim=-1).mean()


def save_checkpoint(model: torch.nn.Module, output_dir: Path, name: str, args: argparse.Namespace) -> Path:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / name
    torch.save(model.state_dict(), path)
    with (ckpt_dir / "latest.txt").open("w", encoding="utf-8") as handle:
        handle.write(str(path) + "\n")
    with (output_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    return path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    ann_file = Path(args.ann_file) if args.ann_file else infer_ann_file(data_root, args.split)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = CocoBoxDataset(
        data_root=data_root,
        split=args.split,
        ann_file=ann_file,
        resolution=args.resolution,
        num_images=args.num_images,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )

    student = build_encoder(args.encoder, args.pretrained_encoder, device)
    validate_args(args, student)
    teacher = build_encoder(args.encoder, args.pretrained_encoder, device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    trainable_names = set_trainable_layers(
        student,
        args.train_last_blocks,
        args.train_block_indexes,
        args.freeze_final_norm,
    )
    student.train()
    centers, grid_h, grid_w = patch_centers_for_model(student, args.resolution, device)
    patch_boxes = patch_boxes_xyxy(grid_h, grid_w, int(student.patch_size), device)
    dense_enabled = bool(args.dense_layers)
    print(f"Dataset images: {len(dataset)}")
    print(f"Annotation file: {ann_file}")
    print(f"Patch grid: {grid_h}x{grid_w}")
    print(f"Trainable modules: {', '.join(trainable_names)}")
    if dense_enabled:
        print(
            f"Dense layers: {args.dense_layers} distill/global_weights={args.dense_layer_weights} "
            f"object_weights={args.dense_object_layer_weights}"
        )

    trainable_params = [param for param in student.parameters() if param.requires_grad]
    optimizer_groups = build_optimizer_param_groups(student, args)
    optimizer = torch.optim.AdamW(optimizer_groups, lr=args.lr, weight_decay=args.weight_decay)
    print(f"Optimizer LRs: {sorted({group['lr'] for group in optimizer.param_groups})}")
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history_path = output_dir / "history.csv"
    fieldnames = [
        "epoch",
        "step",
        "lr",
        "loss",
        "loss_align",
        "loss_cover",
        "loss_consistency",
        "loss_distill",
        "loss_dense_object",
        "loss_dense_global",
        "valid_images",
        "max_memory_mib",
        "seconds",
    ]
    if dense_enabled:
        for layer in args.dense_layers:
            fieldnames.extend(
                [
                    f"loss_distill_l{layer}",
                    f"loss_dense_object_l{layer}",
                    f"loss_dense_global_l{layer}",
                    f"teacher_cosine_l{layer}",
                    f"cross_view_cosine_l{layer}",
                ]
            )
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

    global_step = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        for batch in loader:
            global_step += 1
            images = batch["images"].to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                if dense_enabled:
                    student_levels = extract_multilevel_features(student, images, args.dense_layers)
                    with torch.no_grad():
                        teacher_levels = extract_multilevel_features(teacher, images, args.dense_layers)
                    flipped_student_levels = extract_multilevel_features(
                        student,
                        torch.flip(images, dims=[-1]),
                        args.dense_layers,
                    )
                    distill_loss, object_dense_loss, global_dense_loss, dense_stats = (
                        compute_multilevel_dense_losses(
                            student_levels,
                            teacher_levels,
                            flipped_student_levels,
                            batch["boxes"],
                            patch_boxes,
                            grid_h,
                            grid_w,
                            args,
                        )
                    )
                    final_layer = args.dense_layers[-1]
                    loss, stats = compute_losses(
                        student_levels[final_layer],
                        teacher_levels[final_layer],
                        batch["boxes"],
                        centers,
                        args,
                        distill_override=distill_loss,
                    )
                    consistency_loss = compute_cls_consistency_from_levels(
                        student_levels,
                        flipped_student_levels,
                        final_layer,
                    )
                    loss = (
                        loss
                        + args.lambda_consistency * consistency_loss
                        + args.lambda_dense_object * object_dense_loss
                        + args.lambda_dense_global * global_dense_loss
                    )
                    stats.update(dense_stats)
                else:
                    student_features = student.forward_features(images)
                    with torch.no_grad():
                        teacher_features = teacher.forward_features(images)
                    loss, stats = compute_losses(student_features, teacher_features, batch["boxes"], centers, args)
                    consistency_loss = compute_flip_consistency_loss(
                        student,
                        images,
                        student_features["x_norm_clstoken"],
                    )
                    object_dense_loss = loss.new_tensor(0.0)
                    global_dense_loss = loss.new_tensor(0.0)
                    loss = loss + args.lambda_consistency * consistency_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            row = {
                "epoch": epoch,
                "step": global_step,
                "lr": max(group["lr"] for group in optimizer.param_groups),
                "loss": float(loss.detach().item()),
                "loss_align": stats["loss_align"],
                "loss_cover": stats["loss_cover"],
                "loss_consistency": float(consistency_loss.detach().item()),
                "loss_distill": stats["loss_distill"],
                "loss_dense_object": float(object_dense_loss.detach().item()),
                "loss_dense_global": float(global_dense_loss.detach().item()),
                "valid_images": stats["valid_images"],
                "max_memory_mib": (
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                    if device.type == "cuda"
                    else 0.0
                ),
                "seconds": time.time() - start_time,
            }
            for field in fieldnames:
                if field in stats:
                    row[field] = stats[field]
            with history_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writerow(row)

            if global_step == 1 or global_step % args.log_every == 0:
                print(
                    "epoch={epoch} step={step} loss={loss:.5f} align={loss_align:.5f} "
                    "cover={loss_cover:.5f} cons={loss_consistency:.5f} distill={loss_distill:.5f} "
                    "dense_obj={loss_dense_object:.5f} dense_global={loss_dense_global:.5f} "
                    "valid={valid_images:.0f} max_mem={max_memory_mib:.0f}MiB".format(**row),
                    flush=True,
                )

            if args.max_steps is not None and global_step >= args.max_steps:
                final_path = save_checkpoint(student, output_dir, "model_final.pt", args)
                print(f"Saved checkpoint: {final_path}")
                return

        if args.save_every_epoch:
            path = save_checkpoint(student, output_dir, f"model_epoch{epoch + 1}.pt", args)
            print(f"Saved epoch checkpoint: {path}", flush=True)

    final_path = save_checkpoint(student, output_dir, "model_final.pt", args)
    print(f"Saved checkpoint: {final_path}")


if __name__ == "__main__":
    main()
