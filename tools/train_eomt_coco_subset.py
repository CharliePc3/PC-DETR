#!/usr/bin/env python3
"""Initial EoMT-style instance-segmentation experiment on a COCO subset.

This deliberately remains separate from RF-DETR while the architectural
question is being tested. It trains learned queries in the final native DINOv3
blocks with class, BCE-mask, and Dice losses, and reports standard COCO mask AP.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
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
from rfdetr.models.eomt import EncoderOnlyMaskTransformer  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train the EoMT-style control on COCO")
    parser.add_argument(
        "--data-root",
        default="/data/cpc/root/dataset/COCO_RFDETR_TEST/overfit",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-size", default="small", choices=tuple(SIZE_TO_MODEL))
    parser.add_argument("--pretrained-encoder", default=None)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-query-blocks", type=int, default=2)
    parser.add_argument("--mask-dim", type=int, default=256)
    parser.add_argument("--masked-attention", action="store_true")
    parser.add_argument("--mask-annealing-power", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-cost", type=float, default=2.0)
    parser.add_argument("--mask-ce-cost", type=float, default=5.0)
    parser.add_argument("--mask-dice-cost", type=float, default=5.0)
    parser.add_argument("--no-object-weight", type=float, default=0.1)
    parser.add_argument("--aux-loss-weight", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--detection-query-checkpoint",
        default=None,
        help="Optional RF-DETR checkpoint whose query_feat.weight initializes encoder queries.",
    )
    parser.add_argument("--detection-query-seed", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", default="val", choices=("train", "val"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class CocoInstanceDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        image_size: int,
        category_to_contiguous: dict[int, int],
        *,
        augment: bool,
    ) -> None:
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.category_to_contiguous = category_to_contiguous
        self.augment = augment
        self.coco = COCO(str(data_root / "annotations" / f"instances_{split}2017.json"))
        self.image_ids = [
            image_id
            for image_id in sorted(self.coco.getImgIds())
            if self.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        ]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        info = self.coco.loadImgs([image_id])[0]
        image = Image.open(
            self.data_root / f"{self.split}2017" / info["file_name"]
        ).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.BICUBIC
        )
        annotations = self.coco.loadAnns(
            self.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        )
        labels = []
        masks = []
        mask_size = self.image_size // 4
        for annotation in annotations:
            category_id = int(annotation["category_id"])
            if category_id not in self.category_to_contiguous:
                continue
            mask = self.coco.annToMask(annotation).astype(np.uint8)
            resized = Image.fromarray(mask * 255).resize(
                (mask_size, mask_size), Image.Resampling.NEAREST
            )
            mask_tensor = torch.from_numpy(
                np.asarray(resized, dtype=np.uint8).copy()
            ) > 127
            if not mask_tensor.any():
                continue
            labels.append(self.category_to_contiguous[category_id])
            masks.append(mask_tensor)

        image_tensor = TVF.to_tensor(image)
        if self.augment and random.random() < 0.5:
            image_tensor = image_tensor.flip(-1)
            masks = [mask.flip(-1) for mask in masks]
        image_tensor = TVF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        target = {
            "labels": torch.tensor(labels, dtype=torch.long),
            "masks": (
                torch.stack(masks)
                if masks
                else torch.zeros(0, mask_size, mask_size, dtype=torch.bool)
            ),
            "image_id": image_id,
            "original_size": (int(info["height"]), int(info["width"])),
        }
        return image_tensor, target


def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def pairwise_mask_costs(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Efficient pairwise BCE and Dice costs for [Q,P] and [T,P]."""

    targets = targets.to(logits.dtype)
    num_points = logits.shape[1]
    ce = F.softplus(logits).mean(dim=1, keepdim=True)
    ce = ce - logits @ targets.T / max(num_points, 1)
    probabilities = logits.sigmoid()
    numerator = 2 * (probabilities @ targets.T)
    denominator = probabilities.sum(dim=1, keepdim=True) + targets.sum(dim=1)[None]
    dice = 1 - (numerator + 1) / (denominator + 1)
    return ce, dice


def hungarian_indices(
    prediction: dict[str, torch.Tensor],
    target: dict,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = target["labels"]
    if labels.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=prediction["pred_logits"].device)
        return empty, empty
    probabilities = prediction["pred_logits"].softmax(dim=-1)
    class_cost = -probabilities[:, labels]
    flat_masks = prediction["pred_masks"].flatten(1)
    flat_targets = target["masks"].to(flat_masks.device).flatten(1)
    ce_cost, dice_cost = pairwise_mask_costs(flat_masks, flat_targets)
    total_cost = (
        args.class_cost * class_cost
        + args.mask_ce_cost * ce_cost
        + args.mask_dice_cost * dice_cost
    )
    source, destination = linear_sum_assignment(total_cost.detach().float().cpu())
    return (
        torch.as_tensor(source, dtype=torch.long, device=flat_masks.device),
        torch.as_tensor(destination, dtype=torch.long, device=flat_masks.device),
    )


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid().flatten(1)
    targets = targets.to(probabilities.dtype).flatten(1)
    numerator = 2 * (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    return (1 - (numerator + 1) / (denominator + 1)).mean()


def prediction_loss(
    batched_prediction: dict[str, torch.Tensor],
    targets: list[dict],
    args: argparse.Namespace,
    no_object_index: int,
) -> dict[str, torch.Tensor]:
    class_losses = []
    ce_losses = []
    dice_losses = []
    class_weights = torch.ones(
        no_object_index + 1,
        device=batched_prediction["pred_logits"].device,
    )
    class_weights[no_object_index] = args.no_object_weight
    for batch_index, target in enumerate(targets):
        prediction = {
            "pred_logits": batched_prediction["pred_logits"][batch_index],
            "pred_masks": batched_prediction["pred_masks"][batch_index],
        }
        target = {
            **target,
            "labels": target["labels"].to(prediction["pred_logits"].device),
            "masks": target["masks"].to(prediction["pred_masks"].device),
        }
        source, destination = hungarian_indices(prediction, target, args)
        query_targets = torch.full(
            (prediction["pred_logits"].shape[0],),
            no_object_index,
            dtype=torch.long,
            device=prediction["pred_logits"].device,
        )
        if source.numel():
            query_targets[source] = target["labels"][destination]
        class_losses.append(
            F.cross_entropy(
                prediction["pred_logits"], query_targets, weight=class_weights
            )
        )
        if source.numel():
            matched_logits = prediction["pred_masks"][source]
            matched_targets = target["masks"][destination].to(matched_logits.device)
            ce_losses.append(
                F.binary_cross_entropy_with_logits(
                    matched_logits, matched_targets.to(matched_logits.dtype)
                )
            )
            dice_losses.append(dice_loss(matched_logits, matched_targets))

    zero = batched_prediction["pred_logits"].sum() * 0
    return {
        "class": torch.stack(class_losses).mean(),
        "mask_ce": torch.stack(ce_losses).mean() if ce_losses else zero,
        "mask_dice": torch.stack(dice_losses).mean() if dice_losses else zero,
    }


def total_loss(
    output: dict,
    targets: list[dict],
    args: argparse.Namespace,
    no_object_index: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    main_losses = prediction_loss(output, targets, args, no_object_index)
    total = (
        args.class_cost * main_losses["class"]
        + args.mask_ce_cost * main_losses["mask_ce"]
        + args.mask_dice_cost * main_losses["mask_dice"]
    )
    for auxiliary in output["aux_outputs"]:
        auxiliary_losses = prediction_loss(
            auxiliary, targets, args, no_object_index
        )
        total = total + args.aux_loss_weight * (
            args.class_cost * auxiliary_losses["class"]
            + args.mask_ce_cost * auxiliary_losses["mask_ce"]
            + args.mask_dice_cost * auxiliary_losses["mask_dice"]
        )
    values = {name: float(value.detach()) for name, value in main_losses.items()}
    values["total"] = float(total.detach())
    return total, values


@torch.no_grad()
def evaluate_coco(
    model: EncoderOnlyMaskTransformer,
    loader: DataLoader,
    coco: COCO,
    contiguous_to_category: list[int],
    args: argparse.Namespace,
) -> list[float]:
    model.eval()
    results = []
    for images, targets in loader:
        images = images.to(args.device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=args.device.startswith("cuda"),
        ):
            output = model(images)
        probabilities = output["pred_logits"].softmax(dim=-1)[..., :-1]
        scores, labels = probabilities.max(dim=-1)
        for batch_index, target in enumerate(targets):
            height, width = target["original_size"]
            order = scores[batch_index].argsort(descending=True)
            order = order[scores[batch_index, order] >= args.score_threshold][:100]
            masks = F.interpolate(
                output["pred_masks"][batch_index, order, None].float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[:, 0] > 0
            for local_index, query_index in enumerate(order):
                mask = masks[local_index].cpu().numpy().astype(np.uint8)
                if not mask.any():
                    continue
                rle = mask_util.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode("ascii")
                results.append(
                    {
                        "image_id": int(target["image_id"]),
                        "category_id": contiguous_to_category[
                            int(labels[batch_index, query_index])
                        ],
                        "segmentation": rle,
                        "score": float(scores[batch_index, query_index]),
                    }
                )
    if not results:
        return [0.0] * 12
    detections = coco.loadRes(results)
    evaluator = COCOeval(coco, detections, "segm")
    evaluator.params.imgIds = loader.dataset.image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator.stats.tolist()


@torch.no_grad()
def evaluate_oracle_masks(
    model: EncoderOnlyMaskTransformer,
    loader: DataLoader,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    """Measure the best one-to-one mask assignment without class/score filtering.

    COCO AP entangles mask shape, query assignment, class prediction, and score
    calibration. This diagnostic deliberately removes the final two factors: it
    assigns queries to ground-truth instances using binary mask IoU only, then
    reports both mask quality and the class accuracy of those mask-selected
    queries.
    """

    model.eval()
    matched_ious: list[torch.Tensor] = []
    class_correct = 0
    true_class_probability = 0.0
    num_ground_truth = 0
    for images, targets in loader:
        images = images.to(args.device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=args.device.startswith("cuda"),
        ):
            output = model(images)
        probabilities = output["pred_logits"].softmax(dim=-1)
        for batch_index, target in enumerate(targets):
            target_masks = target["masks"].to(args.device).flatten(1).bool()
            if target_masks.shape[0] == 0:
                continue
            predicted_masks = output["pred_masks"][batch_index].flatten(1) > 0
            intersections = predicted_masks.float() @ target_masks.float().T
            unions = (
                predicted_masks.sum(dim=1, keepdim=True)
                + target_masks.sum(dim=1)[None]
                - intersections
            )
            ious = intersections / unions.clamp_min(1)
            source, destination = linear_sum_assignment(
                -ious.detach().float().cpu().numpy()
            )
            source = torch.as_tensor(source, dtype=torch.long, device=args.device)
            destination = torch.as_tensor(
                destination, dtype=torch.long, device=args.device
            )
            selected_ious = ious[source, destination]
            matched_ious.append(selected_ious.cpu())

            labels = target["labels"].to(args.device)[destination]
            selected_probabilities = probabilities[batch_index, source]
            class_correct += int(
                (selected_probabilities[:, :-1].argmax(dim=-1) == labels).sum()
            )
            true_class_probability += float(
                selected_probabilities[
                    torch.arange(labels.shape[0], device=args.device), labels
                ].sum()
            )
            num_ground_truth += labels.shape[0]

    if not matched_ious:
        return {
            "num_ground_truth": 0,
            "oracle_mask_iou": 0.0,
            "oracle_mask_recall50": 0.0,
            "oracle_mask_recall75": 0.0,
            "mask_selected_class_accuracy": 0.0,
            "mask_selected_true_class_probability": 0.0,
        }
    ious = torch.cat(matched_ious)
    return {
        "num_ground_truth": num_ground_truth,
        "oracle_mask_iou": float(ious.mean()),
        "oracle_mask_recall50": float((ious >= 0.5).float().mean()),
        "oracle_mask_recall75": float((ious >= 0.75).float().mean()),
        "mask_selected_class_accuracy": class_correct / num_ground_truth,
        "mask_selected_true_class_probability": (
            true_class_probability / num_ground_truth
        ),
    }


def set_training_modes(model: EncoderOnlyMaskTransformer) -> None:
    model.train()
    model.encoder.eval()
    for block in model.encoder.blocks[model.query_start_block :]:
        block.train()
    model.encoder.norm.train()


def main() -> None:
    args = parse_args()
    if args.image_size % 16:
        raise ValueError("image size must be divisible by the DINOv3 patch size.")
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("batch size and epochs must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_coco = COCO(str(data_root / "annotations" / "instances_train2017.json"))
    categories = sorted(train_coco.getCatIds())
    category_to_contiguous = {
        category_id: index for index, category_id in enumerate(categories)
    }
    train_dataset = CocoInstanceDataset(
        data_root,
        "train",
        args.image_size,
        category_to_contiguous,
        augment=True,
    )
    val_dataset = CocoInstanceDataset(
        data_root,
        "val",
        args.image_size,
        category_to_contiguous,
        augment=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    model_name = SIZE_TO_MODEL[args.model_size]
    encoder = _load_dinov3_backbone_builder(model_name)(pretrained=False)
    weights_path = _resolve_weights_path(model_name, args.pretrained_encoder)
    encoder.load_state_dict(_load_state_dict(weights_path), strict=True)
    model = EncoderOnlyMaskTransformer(
        encoder,
        num_classes=len(categories),
        num_queries=args.num_queries,
        num_query_blocks=args.num_query_blocks,
        mask_dim=args.mask_dim,
        freeze_image_prefix=True,
        masked_attention=args.masked_attention,
        mask_annealing_power=args.mask_annealing_power,
    ).to(args.device)
    query_initialization = None
    if args.detection_query_checkpoint and not args.resume:
        detector_checkpoint_path = (
            Path(args.detection_query_checkpoint).expanduser().resolve()
        )
        detector_checkpoint = torch.load(
            detector_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        detector_state = detector_checkpoint.get("model", detector_checkpoint)
        if "query_feat.weight" not in detector_state:
            raise KeyError(
                f"{detector_checkpoint_path} does not contain query_feat.weight."
            )
        query_initialization = model.initialize_queries_from_detection(
            detector_state["query_feat.weight"],
            seed=args.detection_query_seed,
        )
        query_initialization["checkpoint"] = str(detector_checkpoint_path)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    if args.resume:
        resume_checkpoint = torch.load(
            Path(args.resume).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        if not args.eval_only and "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1
    no_object_index = len(categories)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        json.dumps(
            {
                "train_images": len(train_dataset),
                "val_images": len(val_dataset),
                "categories": len(categories),
                "trainable_parameters": trainable_parameters,
                "weights": str(weights_path),
                "query_initialization": query_initialization,
            }
        ),
        flush=True,
    )

    if args.eval_only:
        evaluation_dataset = CocoInstanceDataset(
            data_root,
            args.eval_split,
            args.image_size,
            category_to_contiguous,
            augment=False,
        )
        evaluation_loader = DataLoader(
            evaluation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
        )
        stats = evaluate_coco(
            model,
            evaluation_loader,
            evaluation_dataset.coco,
            categories,
            args,
        )
        oracle = evaluate_oracle_masks(model, evaluation_loader, args)
        print(
            json.dumps(
                {
                    "split": args.eval_split,
                    "mask_ap": stats[0],
                    "mask_ap50": stats[1],
                    "mask_ap75": stats[2],
                    "mask_aps": stats[3],
                    "mask_apm": stats[4],
                    "mask_apl": stats[5],
                    **oracle,
                }
            ),
            flush=True,
        )
        return

    best_ap = -1.0
    metrics_path = output_dir / "metrics.jsonl"
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.masked_attention:
            progress = epoch / max(args.epochs - 1, 1)
            model.set_mask_annealing_progress(progress)
        set_training_modes(model)
        running = {"total": 0.0, "class": 0.0, "mask_ce": 0.0, "mask_dice": 0.0}
        for step, (images, targets) in enumerate(train_loader):
            images = images.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=args.device.startswith("cuda"),
            ):
                output = model(images)
                loss, values = total_loss(output, targets, args, no_object_index)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                1.0,
            )
            optimizer.step()
            for name in running:
                running[name] += values[name]
            if step % 25 == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step}/{len(train_loader)} "
                    f"loss={values['total']:.4f} ce={values['mask_ce']:.4f} "
                    f"dice={values['mask_dice']:.4f}",
                    flush=True,
                )
        stats = evaluate_coco(
            model,
            val_loader,
            val_dataset.coco,
            categories,
            args,
        )
        record = {
            "epoch": epoch + 1,
            "train": {
                name: value / len(train_loader) for name, value in running.items()
            },
            "mask_ap": stats[0],
            "mask_ap50": stats[1],
            "mask_ap75": stats[2],
            "mask_aps": stats[3],
            "mask_apm": stats[4],
            "mask_apl": stats[5],
            "elapsed_seconds": time.time() - start_time,
            "masking_probabilities": (
                list(model.masking_probabilities)
                if args.masked_attention
                else []
            ),
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "categories": categories,
            "metrics": record,
        }
        torch.save(state, output_dir / "checkpoint.pth")
        if record["mask_ap"] > best_ap:
            best_ap = record["mask_ap"]
            torch.save(state, output_dir / "checkpoint_best.pth")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
