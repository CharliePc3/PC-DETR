#!/usr/bin/env python3
"""CPU smoke checks for budgeted scale-adaptive auxiliary matching."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr.models.lwdetr import SetCriterion
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def legacy_match(matcher, outputs, targets, group_detr):
    """Reference implementation of the pre-SA grouped Hungarian path."""
    batch_size, num_queries = outputs["pred_logits"].shape[:2]
    logits = outputs["pred_logits"].flatten(0, 1)
    probabilities = logits.sigmoid()
    boxes = outputs["pred_boxes"].flatten(0, 1)
    target_labels = torch.cat([target["labels"] for target in targets])
    target_boxes = torch.cat([target["boxes"] for target in targets])

    alpha = 0.25
    gamma = 2.0
    negative_cost = (1 - alpha) * probabilities.pow(gamma) * (-F.logsigmoid(-logits))
    positive_cost = alpha * (1 - probabilities).pow(gamma) * (-F.logsigmoid(logits))
    class_cost = positive_cost[:, target_labels] - negative_cost[:, target_labels]
    bbox_cost = torch.cdist(boxes, target_boxes, p=1)
    giou_cost = -generalized_box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(target_boxes))
    cost = (
        matcher.cost_bbox * bbox_cost + matcher.cost_class * class_cost + matcher.cost_giou * giou_cost
    ).view(batch_size, num_queries, -1).float().cpu()

    sizes = [len(target["boxes"]) for target in targets]
    per_group_queries = num_queries // group_detr
    indices = []
    for group_index, group_cost in enumerate(cost.split(per_group_queries, dim=1)):
        group_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(group_cost.split(sizes, -1))]
        if group_index == 0:
            indices = group_indices
        else:
            indices = [
                (
                    np.concatenate([old[0], new[0] + per_group_queries * group_index]),
                    np.concatenate([old[1], new[1]]),
                )
                for old, new in zip(indices, group_indices)
            ]
    return [(torch.as_tensor(src), torch.as_tensor(tgt)) for src, tgt in indices]


def make_inputs():
    torch.manual_seed(7)
    outputs = {
        "pred_logits": torch.randn(1, 60, 3),
        "pred_boxes": torch.rand(1, 60, 4),
    }
    outputs["pred_boxes"][..., 2:] = outputs["pred_boxes"][..., 2:] * 0.25 + 0.01
    target = {
        "labels": torch.tensor([0, 1, 2]),
        "boxes": torch.tensor(
            [
                [0.2, 0.2, 20 / 640, 20 / 640],
                [0.5, 0.5, 60 / 640, 60 / 640],
                [0.8, 0.8, 120 / 640, 120 / 640],
            ],
            dtype=torch.float32,
        ),
        "size": torch.tensor([640, 640]),
        "image_id": torch.tensor([17]),
    }
    return outputs, [target]


def pair_set(indices):
    return set(zip(indices[0][0].tolist(), indices[0][1].tolist()))


def main():
    matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)
    outputs, targets = make_inputs()

    expected = legacy_match(matcher, outputs, targets, group_detr=6)
    strict = matcher(outputs, targets, group_detr=6)
    assert all(torch.equal(a, b) and torch.equal(c, d) for (a, c), (b, d) in zip(strict, expected))

    # Exercise the target-offset path with uneven target counts in a real batch.
    batched_outputs = {
        "pred_logits": torch.cat([outputs["pred_logits"], outputs["pred_logits"] * 0.7], dim=0),
        "pred_boxes": torch.cat([outputs["pred_boxes"], outputs["pred_boxes"].flip(1)], dim=0),
    }
    second_target = {
        key: value[:2].clone() if key in ("labels", "boxes") else value.clone()
        for key, value in targets[0].items()
    }
    second_target["image_id"] = torch.tensor([23])
    batched_targets = [targets[0], second_target]
    batched_expected = legacy_match(matcher, batched_outputs, batched_targets, group_detr=6)
    batched_strict = matcher(batched_outputs, batched_targets, group_detr=6)
    assert all(
        torch.equal(a, b) and torch.equal(c, d)
        for (a, c), (b, d) in zip(batched_strict, batched_expected)
    )

    adaptive, info = matcher(
        outputs,
        targets,
        group_detr=6,
        sa_total_budgets=(6, 7, 9),
        sa_area_thresholds=(32**2, 96**2),
        sa_layer_index=1,
        return_match_info=True,
    )
    assert len(strict[0][0]) == 18
    assert len(adaptive[0][0]) == 22
    assert len(set(adaptive[0][0].tolist())) == 22
    assert pair_set(strict).issubset(pair_set(adaptive))
    assert sum(src < 10 for src in adaptive[0][0].tolist()) == 3
    assert info == {
        "sa_requested": 4,
        "sa_matched": 4,
        "sa_matched_small": 0,
        "sa_matched_medium": 1,
        "sa_matched_large": 3,
    }

    criterion = SetCriterion(
        num_classes=3,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2},
        focal_alpha=0.25,
        losses=["labels", "boxes", "cardinality"],
        group_detr=6,
        ia_bce_loss=True,
        use_budgeted_sa=True,
        sa_start_epoch=2,
        sa_stop_epoch=20,
        sa_total_budgets=(6, 7, 9),
    )
    criterion.train()
    criterion_outputs = {
        **outputs,
        "aux_outputs": [
            {key: value.clone() for key, value in outputs.items()},
            {key: value.clone() for key, value in outputs.items()},
            {key: value.clone() for key, value in outputs.items()},
        ],
    }
    criterion.set_epoch(2)
    active_losses = criterion(criterion_outputs, targets)
    assert active_losses["sa_active"].item() == 1
    assert active_losses["sa_matched"].item() == 12
    assert all(torch.isfinite(value).all() for value in active_losses.values())

    criterion.set_epoch(20)
    inactive_losses = criterion(criterion_outputs, targets)
    assert inactive_losses["sa_active"].item() == 0
    assert inactive_losses["sa_matched"].item() == 0
    assert all(torch.isfinite(value).all() for value in inactive_losses.values())

    print("budgeted SA smoke checks passed")


if __name__ == "__main__":
    main()
