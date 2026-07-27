# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from rfdetr.models.segmentation_head import point_sample
from rfdetr.util.box_ops import batch_dice_loss, batch_sigmoid_ce_loss, box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha: float = 0.25,
        use_pos_only: bool = False,
        use_position_modulated_cost: bool = False,
        mask_point_sample_ratio: int = 16,
        cost_mask_ce: float = 1,
        cost_mask_dice: float = 1,
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"
        self.focal_alpha = focal_alpha
        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.cost_mask_ce = cost_mask_ce
        self.cost_mask_dice = cost_mask_dice

    @staticmethod
    def _target_pixel_areas(target):
        boxes = target["boxes"].detach().float().cpu()
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.float32)

        size = target.get("size", target.get("orig_size"))
        if size is None:
            raise ValueError("Budgeted SA matching requires target['size'] or target['orig_size'].")
        height, width = size.detach().flatten().cpu().tolist()[:2]
        return boxes[:, 2].clamp(min=0) * float(width) * boxes[:, 3].clamp(min=0) * float(height)

    @classmethod
    def _build_sa_slots(
        cls,
        target,
        group_detr,
        total_budgets,
        area_thresholds,
        layer_index,
    ):
        """Assign extra GT slots deterministically while preserving base O2O matches."""
        slots = [[] for _ in range(group_detr)]
        if group_detr < 1 or len(target["boxes"]) == 0:
            return slots, [0, 0, 0]

        small_threshold, large_threshold = area_thresholds
        areas = cls._target_pixel_areas(target)
        image_id = target.get("image_id", 0)
        if isinstance(image_id, torch.Tensor):
            image_id = int(image_id.detach().flatten()[0].cpu()) if image_id.numel() else 0
        else:
            image_id = int(image_id)

        requested_by_scale = [0, 0, 0]
        # With grouped DETR, group 0 is used for inference and remains strict O2O.
        # With a single group, auxiliary decoder layers reuse unmatched queries for
        # SA positives while the final decoder output remains strict O2O.
        first_sa_group = 0 if group_detr == 1 else 1
        num_sa_groups = group_detr - first_sa_group
        for target_index, area in enumerate(areas.tolist()):
            if area < small_threshold:
                scale_index = 0
            elif area < large_threshold:
                scale_index = 1
            else:
                scale_index = 2

            extra_count = max(int(total_budgets[scale_index]) - group_detr, 0)
            requested_by_scale[scale_index] += extra_count
            if extra_count == 0:
                continue

            # The hash balances extra slots without introducing RNG state.
            start = (image_id * 1000003 + target_index * 9176 + layer_index * 131) % num_sa_groups
            for extra_index in range(extra_count):
                group_index = first_sa_group + (start + extra_index) % num_sa_groups
                slots[group_index].append((target_index, scale_index))

        return slots, requested_by_scale

    @torch.no_grad()
    def forward(
        self,
        outputs,
        targets,
        group_detr=1,
        sa_total_budgets=None,
        sa_area_thresholds=(32**2, 96**2),
        sa_layer_index=0,
        return_match_info=False,
    ):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "masks": Tensor of dim [num_target_boxes, H, W] containing the target mask coordinates
            group_detr: Number of groups used for matching.
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        flat_pred_logits = outputs["pred_logits"].flatten(0, 1)
        out_prob = flat_pred_logits.sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        masks_present = "masks" in targets[0]

        # Compute the giou cost between boxes
        giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -giou

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0

        # neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # we refactor these with logsigmoid for numerical stability
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-F.logsigmoid(-flat_pred_logits))
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-F.logsigmoid(flat_pred_logits))
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        if masks_present:
            # Resize predicted masks to target mask size if needed
            # if out_masks.shape[-2:] != tgt_masks.shape[-2:]:
            #     # out_masks = F.interpolate(out_masks.unsqueeze(1), size=tgt_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            #     tgt_masks = F.interpolate(tgt_masks.unsqueeze(1).float(), size=out_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

            # # Flatten masks
            # pred_masks_logits = out_masks.flatten(1)  # [P, HW]
            # tgt_masks_flat = tgt_masks.flatten(1).float()  # [T, HW]

            tgt_masks = torch.cat([v["masks"] for v in targets])

            if isinstance(outputs["pred_masks"], torch.Tensor):
                out_masks = outputs["pred_masks"].flatten(0, 1)

                num_points = out_masks.shape[-2] * out_masks.shape[-1] // self.mask_point_sample_ratio

                point_coords = torch.rand(1, num_points, 2, device=out_masks.device)
                pred_masks_logits = point_sample(
                    out_masks.unsqueeze(1), point_coords.repeat(out_masks.shape[0], 1, 1), align_corners=False
                ).squeeze(1)
            else:
                # pred_masks_logits = outputs["sparse_matcher_mask_logits"].flatten(0, 1)
                # point_coords = outputs["matcher_sample_coords"]
                spatial_features = outputs["pred_masks"]["spatial_features"]
                query_features = outputs["pred_masks"]["query_features"]
                bias = outputs["pred_masks"]["bias"]

                num_points = spatial_features.shape[-2] * spatial_features.shape[-1] // self.mask_point_sample_ratio
                point_coords = torch.rand(1, num_points, 2, device=spatial_features.device)
                pred_masks_logits = point_sample(
                    spatial_features, point_coords.repeat(spatial_features.shape[0], 1, 1), align_corners=False
                )
                # print(f"pred_masks_logits.shape: {pred_masks_logits.shape}")
                pred_masks_logits = torch.einsum("bcp,bnc->bnp", pred_masks_logits, query_features) + bias
                pred_masks_logits = pred_masks_logits.flatten(0, 1)

            tgt_masks = tgt_masks.to(pred_masks_logits.dtype)
            tgt_masks_flat = point_sample(
                tgt_masks.unsqueeze(1),
                point_coords.repeat(tgt_masks.shape[0], 1, 1),
                align_corners=False,
                mode="nearest",
            ).squeeze(1)

            # Binary cross-entropy with logits cost (mean over pixels), computed pairwise efficiently
            cost_mask_ce = batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat)

            # Dice loss cost (1 - dice coefficient)
            cost_mask_dice = batch_dice_loss(pred_masks_logits, tgt_masks_flat)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        if masks_present:
            C = C + self.cost_mask_ce * cost_mask_ce + self.cost_mask_dice * cost_mask_dice
        C = C.view(bs, num_queries, -1).float().cpu()  # convert to float because bfloat16 doesn't play nicely with CPU

        # we assume any good match will not cause NaN or Inf, so we replace them with a large value
        max_cost = C.max() if C.numel() > 0 else 0
        C[C.isinf() | C.isnan()] = max_cost * 2

        if num_queries % group_detr != 0:
            raise ValueError(f"num_queries={num_queries} must be divisible by group_detr={group_detr}.")
        if sa_total_budgets is not None:
            if len(sa_total_budgets) != 3:
                raise ValueError("sa_total_budgets must contain small, medium, and large budgets.")
            if len(sa_area_thresholds) != 2 or sa_area_thresholds[0] >= sa_area_thresholds[1]:
                raise ValueError("sa_area_thresholds must be two increasing pixel-area thresholds.")

        sizes = [len(v["boxes"]) for v in targets]
        target_offsets = np.cumsum([0, *sizes])
        indices_by_batch = [([], []) for _ in range(bs)]
        base_rows = [[None for _ in range(bs)] for _ in range(group_detr)]
        g_num_queries = num_queries // group_detr
        for group_index in range(group_detr):
            query_start = group_index * g_num_queries
            query_end = query_start + g_num_queries
            for batch_index, target_size in enumerate(sizes):
                target_start = target_offsets[batch_index]
                target_end = target_offsets[batch_index + 1]
                cost = C[batch_index, query_start:query_end, target_start:target_end]
                src_local, tgt_local = linear_sum_assignment(cost)
                base_rows[group_index][batch_index] = src_local
                indices_by_batch[batch_index][0].extend((src_local + query_start).tolist())
                indices_by_batch[batch_index][1].extend(tgt_local.tolist())

        match_info = {
            "sa_requested": 0,
            "sa_matched": 0,
            "sa_matched_small": 0,
            "sa_matched_medium": 0,
            "sa_matched_large": 0,
        }
        if sa_total_budgets is not None:
            for batch_index, target in enumerate(targets):
                slots_by_group, requested_by_scale = self._build_sa_slots(
                    target,
                    group_detr,
                    sa_total_budgets,
                    sa_area_thresholds,
                    sa_layer_index,
                )
                match_info["sa_requested"] += sum(requested_by_scale)
                target_start = target_offsets[batch_index]
                target_end = target_offsets[batch_index + 1]

                first_sa_group = 0 if group_detr == 1 else 1
                for group_index in range(first_sa_group, group_detr):
                    extra_slots = slots_by_group[group_index]
                    if not extra_slots:
                        continue

                    used_rows = set(base_rows[group_index][batch_index].tolist())
                    candidate_rows = np.asarray(
                        [row for row in range(g_num_queries) if row not in used_rows], dtype=np.int64
                    )
                    if candidate_rows.size == 0:
                        continue

                    extra_targets = np.asarray([slot[0] for slot in extra_slots], dtype=np.int64)
                    query_start = group_index * g_num_queries
                    query_end = query_start + g_num_queries
                    group_cost = C[batch_index, query_start:query_end, target_start:target_end]
                    extra_cost = group_cost[candidate_rows][:, extra_targets]
                    extra_src_index, extra_slot_index = linear_sum_assignment(extra_cost)
                    matched_rows = candidate_rows[extra_src_index]
                    matched_targets = extra_targets[extra_slot_index]

                    indices_by_batch[batch_index][0].extend((matched_rows + query_start).tolist())
                    indices_by_batch[batch_index][1].extend(matched_targets.tolist())
                    match_info["sa_matched"] += len(matched_rows)
                    for slot_index in extra_slot_index.tolist():
                        scale_name = ("small", "medium", "large")[extra_slots[slot_index][1]]
                        match_info[f"sa_matched_{scale_name}"] += 1

        indices = [
            (torch.as_tensor(src, dtype=torch.int64), torch.as_tensor(tgt, dtype=torch.int64))
            for src, tgt in indices_by_batch
        ]
        if return_match_info:
            return indices, match_info
        return indices


def build_matcher(args):
    if args.segmentation_head:
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
            cost_mask_ce=args.mask_ce_loss_coef,
            cost_mask_dice=args.mask_dice_loss_coef,
            mask_point_sample_ratio=args.mask_point_sample_ratio,
        )
    else:
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
        )
