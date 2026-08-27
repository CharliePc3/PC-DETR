import torch
import torch.nn.functional as F

from rfdetr.util import box_ops
from rfdetr.util.misc import inverse_sigmoid


def _noised_boxes(boxes, box_noise_scale, negative=False):
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
    diff = torch.zeros_like(boxes)
    diff[:, :2] = boxes[:, 2:] / 2
    diff[:, 2:] = boxes[:, 2:] / 2

    rand_sign = torch.randint_like(boxes, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
    rand_part = torch.rand_like(boxes)
    if negative:
        rand_part = rand_part + 1.0
    rand_part = rand_part * rand_sign

    boxes_xyxy = boxes_xyxy + rand_part * diff * box_noise_scale
    boxes_xyxy = boxes_xyxy.clamp(min=0.0, max=1.0)
    boxes = box_ops.box_xyxy_to_cxcywh(boxes_xyxy)
    return boxes.clamp(min=0.0, max=1.0)


def prepare_for_cdn(
    targets,
    dn_number,
    label_noise_scale,
    box_noise_scale,
    num_queries,
    num_classes,
    hidden_dim,
    label_embed,
    bbox_reparam,
    dn_negative=True,
    group_detr=1,
):
    if dn_number <= 0:
        return None, None, None, None

    device = label_embed.weight.device
    dtype = label_embed.weight.dtype
    batch_size = len(targets)
    known_num = [len(t["labels"]) for t in targets]
    max_gt = max(known_num) if known_num else 0
    if max_gt == 0:
        return None, None, None, None

    group_size = max_gt * (2 if dn_negative else 1)
    dn_groups = max(dn_number // group_size, 1)
    pad_size = group_size * dn_groups

    input_query_label = torch.zeros(batch_size, group_detr, pad_size, hidden_dim, device=device, dtype=dtype)
    input_query_bbox = torch.zeros(batch_size, group_detr, pad_size, 4, device=device, dtype=dtype)

    pos_indices = []
    neg_indices = []
    tgt_indices = []
    target_offset = 0
    for batch_idx, target in enumerate(targets):
        labels = target["labels"].to(device)
        boxes = target["boxes"].to(device)
        num_gt = len(labels)
        if num_gt == 0:
            target_offset += num_gt
            continue

        for detr_group_idx in range(group_detr):
            for dn_group_idx in range(dn_groups):
                group_start = dn_group_idx * group_size
                pos_out_idx = torch.arange(num_gt, device=device) + group_start

                noisy_labels = labels.clone()
                if label_noise_scale > 0:
                    chosen = torch.rand(num_gt, device=device) < label_noise_scale
                    if chosen.any():
                        noisy_labels[chosen] = torch.randint(0, num_classes, (int(chosen.sum()),), device=device)

                pos_boxes = _noised_boxes(boxes, box_noise_scale, negative=False)
                input_query_label[batch_idx, detr_group_idx, pos_out_idx] = label_embed(noisy_labels)
                input_query_bbox[batch_idx, detr_group_idx, pos_out_idx] = (
                    pos_boxes if bbox_reparam else inverse_sigmoid(pos_boxes)
                )

                pos_indices.append(
                    torch.stack(
                        [
                            torch.full_like(pos_out_idx, batch_idx),
                            torch.full_like(pos_out_idx, detr_group_idx),
                            pos_out_idx,
                        ],
                        dim=1,
                    )
                )
                tgt_indices.append(torch.arange(num_gt, device=device) + target_offset)

                if dn_negative:
                    neg_out_idx = pos_out_idx + max_gt
                    neg_boxes = _noised_boxes(boxes, box_noise_scale, negative=True)
                    input_query_label[batch_idx, detr_group_idx, neg_out_idx] = label_embed(noisy_labels)
                    input_query_bbox[batch_idx, detr_group_idx, neg_out_idx] = (
                        neg_boxes if bbox_reparam else inverse_sigmoid(neg_boxes)
                    )
                    neg_indices.append(
                        torch.stack(
                            [
                                torch.full_like(neg_out_idx, batch_idx),
                                torch.full_like(neg_out_idx, detr_group_idx),
                                neg_out_idx,
                            ],
                            dim=1,
                        )
                    )
        target_offset += num_gt

    tgt_size = pad_size + num_queries
    attn_mask = torch.zeros(tgt_size, tgt_size, device=device, dtype=torch.bool)
    attn_mask[pad_size:, :pad_size] = True
    for group_idx in range(dn_groups):
        start = group_idx * group_size
        end = start + group_size
        attn_mask[start:end, :start] = True
        attn_mask[start:end, end:pad_size] = True

    input_query_label = input_query_label.flatten(1, 2)
    input_query_bbox = input_query_bbox.flatten(1, 2)

    pos_indices = torch.cat(pos_indices, dim=0) if pos_indices else torch.empty(0, 3, device=device, dtype=torch.long)
    neg_indices = torch.cat(neg_indices, dim=0) if neg_indices else torch.empty(0, 3, device=device, dtype=torch.long)
    tgt_indices = torch.cat(tgt_indices, dim=0) if tgt_indices else torch.empty(0, device=device, dtype=torch.long)

    dn_meta = {
        "pad_size": pad_size,
        "num_dn_group": dn_groups,
        "group_size": group_size,
        "max_gt": max_gt,
        "group_detr": group_detr,
        "num_queries": num_queries,
        "pos_indices": pos_indices,
        "neg_indices": neg_indices,
        "tgt_indices": tgt_indices,
    }
    return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, set_aux_loss):
    if dn_meta is None or dn_meta.get("pad_size", 0) == 0:
        return outputs_class, outputs_coord, dn_meta

    pad_size = dn_meta["pad_size"]
    group_detr = dn_meta.get("group_detr", 1)
    num_queries = dn_meta.get("num_queries", outputs_class.shape[2] // group_detr - pad_size)
    group_len = pad_size + num_queries

    output_shape = outputs_class.shape
    if output_shape[2] != group_detr * group_len:
        raise ValueError(
            f"Unexpected CDN decoder output length {output_shape[2]}, expected {group_detr * group_len} "
            f"from group_detr={group_detr}, pad_size={pad_size}, num_queries={num_queries}."
        )

    outputs_class_grouped = outputs_class.view(output_shape[0], output_shape[1], group_detr, group_len, output_shape[3])
    outputs_coord_grouped = outputs_coord.view(
        outputs_coord.shape[0], outputs_coord.shape[1], group_detr, group_len, outputs_coord.shape[3]
    )

    output_known_class = outputs_class_grouped[:, :, :, :pad_size, :].flatten(2, 3).contiguous()
    output_known_coord = outputs_coord_grouped[:, :, :, :pad_size, :].flatten(2, 3).contiguous()
    outputs_class = outputs_class_grouped[:, :, :, pad_size:, :].flatten(2, 3).contiguous()
    outputs_coord = outputs_coord_grouped[:, :, :, pad_size:, :].flatten(2, 3).contiguous()

    out = {"pred_logits": output_known_class[-1], "pred_boxes": output_known_coord[-1]}
    if aux_loss:
        out["aux_outputs"] = set_aux_loss(output_known_class, output_known_coord, None)
    dn_meta["output_known_lbs_bboxes"] = out
    return outputs_class, outputs_coord, dn_meta


def sigmoid_focal_loss_raw(inputs, targets, alpha=0.25, gamma=2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss


def _zero_cdn_losses(zero, suffix=""):
    return {
        f"loss_ce_dn{suffix}": zero,
        f"loss_bbox_dn{suffix}": zero,
        f"loss_giou_dn{suffix}": zero,
        f"loss_ce_dn_neg{suffix}": zero,
    }


def _compute_cdn_loss_for_outputs(dn_meta, outputs, targets, focal_alpha, num_boxes, normalizer_group_detr=1, suffix=""):
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    device = pred_logits.device

    pos_indices = dn_meta["pos_indices"].to(device)
    neg_indices = dn_meta["neg_indices"].to(device)
    tgt_indices = dn_meta["tgt_indices"].to(device)

    zero = pred_logits.sum() * 0.0
    losses = _zero_cdn_losses(zero, suffix)
    pad_size = dn_meta["pad_size"]

    if pos_indices.numel() > 0:
        dn_num_boxes = max(
            float(num_boxes) * float(dn_meta["num_dn_group"]) * float(normalizer_group_detr),
            1.0,
        )
        batch_idx = pos_indices[:, 0]
        query_idx = pos_indices[:, 1] * pad_size + pos_indices[:, 2]
        target_labels = torch.cat([t["labels"] for t in targets]).to(device)[tgt_indices]
        target_boxes = torch.cat([t["boxes"] for t in targets]).to(device)[tgt_indices]

        src_logits = pred_logits[batch_idx, query_idx]
        cls_targets = torch.zeros_like(src_logits)
        cls_targets.scatter_(1, target_labels.unsqueeze(1), 1)
        losses[f"loss_ce_dn{suffix}"] = sigmoid_focal_loss_raw(
            src_logits, cls_targets, alpha=focal_alpha, gamma=2
        ).mean(1).sum() / dn_num_boxes

        src_boxes = pred_boxes[batch_idx, query_idx]
        losses[f"loss_bbox_dn{suffix}"] = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / dn_num_boxes
        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses[f"loss_giou_dn{suffix}"] = loss_giou.sum() / dn_num_boxes

    if neg_indices.numel() > 0:
        neg_query_idx = neg_indices[:, 1] * pad_size + neg_indices[:, 2]
        neg_logits = pred_logits[neg_indices[:, 0], neg_query_idx]
        neg_targets = torch.zeros_like(neg_logits)
        normalizer = max(float(neg_logits.shape[0]), 1.0)
        losses[f"loss_ce_dn_neg{suffix}"] = sigmoid_focal_loss_raw(
            neg_logits, neg_targets, alpha=focal_alpha, gamma=2
        ).mean(1).sum() / normalizer

    return losses


def compute_cdn_loss(dn_meta, targets, num_classes, focal_alpha, num_boxes, normalizer_group_detr=1):
    if dn_meta is None or "output_known_lbs_bboxes" not in dn_meta:
        device = targets[0]["labels"].device if targets else torch.device("cpu")
        zero = torch.as_tensor(0.0, device=device)
        return _zero_cdn_losses(zero)

    outputs = dn_meta["output_known_lbs_bboxes"]
    losses = _compute_cdn_loss_for_outputs(
        dn_meta,
        outputs,
        targets,
        focal_alpha,
        num_boxes,
        normalizer_group_detr=normalizer_group_detr,
    )

    for i, aux_outputs in enumerate(outputs.get("aux_outputs", [])):
        losses.update(
            _compute_cdn_loss_for_outputs(
                dn_meta,
                aux_outputs,
                targets,
                focal_alpha,
                num_boxes,
                normalizer_group_detr=normalizer_group_detr,
                suffix=f"_{i}",
            )
        )

    return losses
