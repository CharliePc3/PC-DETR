#!/usr/bin/env python3
"""CPU smoke checks for Dense O2O batch augmentations."""

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr.datasets.dense_o2o import DenseO2OAugmenter


def make_batch():
    images = torch.stack([torch.full((3, 64, 64), index / 4) for index in range(4)])
    masks = torch.zeros((4, 64, 64), dtype=torch.bool)
    targets = []
    for index in range(4):
        targets.append(
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32),
                "labels": torch.tensor([index], dtype=torch.int64),
                "area": torch.tensor([256.0]),
                "iscrowd": torch.tensor([0]),
                "image_id": torch.tensor([index]),
                "orig_size": torch.tensor([64, 64]),
                "size": torch.tensor([64, 64]),
            }
        )
    return images, masks, targets


def assert_valid(images, targets):
    assert torch.isfinite(images).all()
    for target in targets:
        assert len(target["boxes"]) == len(target["labels"]) == len(target["area"])
        assert torch.isfinite(target["boxes"]).all()
        assert (target["boxes"] >= 0).all() and (target["boxes"] <= 1).all()


def main():
    images, masks, targets = make_batch()
    original_images = images.clone()
    original_boxes = [target["boxes"].clone() for target in targets]

    mosaic = DenseO2OAugmenter(
        mode="image",
        start_epoch=0,
        image_stop_epoch=1,
        mosaic_prob=1,
        mixup_prob=0,
        seed=11,
    )
    mosaic_images, _, mosaic_targets, mosaic_stats = mosaic(images, masks, targets, epoch=0, step=0)
    assert mosaic_stats["dense_o2o_mosaic_images"] == 4
    assert mosaic_stats["dense_o2o_objects_before"] == 4
    assert mosaic_stats["dense_o2o_objects_after"] == 16
    assert all(len(target["boxes"]) == 4 for target in mosaic_targets)
    assert_valid(mosaic_images, mosaic_targets)

    repeated_images, _, repeated_targets, repeated_stats = mosaic(images, masks, targets, epoch=0, step=0)
    assert torch.equal(mosaic_images, repeated_images)
    assert repeated_stats == mosaic_stats
    assert all(torch.equal(a["boxes"], b["boxes"]) for a, b in zip(mosaic_targets, repeated_targets))

    mixup = DenseO2OAugmenter(
        mode="image",
        start_epoch=0,
        image_stop_epoch=1,
        mosaic_prob=0,
        mixup_prob=1,
        seed=12,
    )
    mixup_images, _, mixup_targets, mixup_stats = mixup(images, masks, targets, epoch=0, step=0)
    assert mixup_stats["dense_o2o_mixup"] == 1
    assert mixup_stats["dense_o2o_objects_after"] == 8
    assert all(len(target["boxes"]) == 2 for target in mixup_targets)
    assert_valid(mixup_images, mixup_targets)

    copyblend = DenseO2OAugmenter(
        mode="enhanced",
        start_epoch=0,
        image_stop_epoch=1,
        copyblend_stop_epoch=2,
        mosaic_prob=0,
        mixup_prob=0,
        copyblend_prob=1,
        copyblend_area_threshold=1,
        copyblend_num_objects=2,
        seed=13,
    )
    blended_images, _, blended_targets, blended_stats = copyblend(images, masks, targets, epoch=0, step=0)
    assert blended_stats["dense_o2o_copyblend_objects"] == 8
    assert blended_stats["dense_o2o_objects_after"] == 12
    assert all(len(target["boxes"]) == 3 for target in blended_targets)
    assert_valid(blended_images, blended_targets)

    inactive_images, _, inactive_targets, inactive_stats = copyblend(images, masks, targets, epoch=2, step=0)
    assert inactive_stats["dense_o2o_active"] == 0
    assert torch.equal(inactive_images, images)
    assert all(torch.equal(a["boxes"], b["boxes"]) for a, b in zip(inactive_targets, targets))

    assert torch.equal(images, original_images)
    assert all(torch.equal(target["boxes"], boxes) for target, boxes in zip(targets, original_boxes))
    print("Dense O2O smoke checks passed")


if __name__ == "__main__":
    main()
