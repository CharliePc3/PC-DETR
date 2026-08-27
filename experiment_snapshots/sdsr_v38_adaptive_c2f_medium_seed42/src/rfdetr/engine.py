# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""

import math
import random
from typing import Iterable

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import rfdetr.util.misc as utils
from rfdetr.datasets.coco import compute_multi_scale_scales
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.datasets.dense_o2o import DenseO2OAugmenter
from rfdetr.util.logger import get_logger
from rfdetr.util.misc import get_world_size

try:
    from torch.amp import GradScaler, autocast

    DEPRECATED_AMP = False
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

    DEPRECATED_AMP = True
from typing import Callable, DefaultDict, List

import numpy as np

from rfdetr.util.misc import NestedTensor

logger = get_logger()
BYTES_TO_MB = 1024.0 * 1024.0


def _is_cuda(device: torch.device) -> bool:
    """Return True if device is a CUDA device with an active CUDA context."""
    return (
        isinstance(device, torch.device)
        and device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_initialized()
    )


def _get_cuda_autocast_dtype() -> torch.dtype:
    """Return the autocast dtype that is supported on the current CUDA device."""
    if not torch.cuda.is_available():
        return torch.bfloat16

    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported):
        return torch.bfloat16 if is_bf16_supported() else torch.float16

    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def get_autocast_args(args):
    autocast_dtype = _get_cuda_autocast_dtype()
    if DEPRECATED_AMP:
        return {"enabled": args.amp, "dtype": autocast_dtype}
    else:
        return {"device_type": "cuda", "enabled": args.amp, "dtype": autocast_dtype}


def projector_distillation_loss(student_features, teacher_features, level_weights):
    if len(student_features) != len(teacher_features):
        raise ValueError("Student and teacher must expose the same number of projector levels.")
    if len(level_weights) != len(student_features):
        raise ValueError("Projector distillation requires one weight per feature level.")
    weight_sum = float(sum(level_weights))
    if weight_sum <= 0:
        raise ValueError("Projector distillation level weights must have a positive sum.")

    loss = student_features[0].new_zeros((), dtype=torch.float32)
    for student, teacher, weight in zip(
        student_features, teacher_features, level_weights
    ):
        if student.shape != teacher.shape:
            raise ValueError(
                "Student and teacher projector feature shapes differ: "
                f"{tuple(student.shape)} != {tuple(teacher.shape)}."
            )
        student = F.normalize(student.float(), dim=1, eps=1e-6)
        teacher = F.normalize(teacher.float(), dim=1, eps=1e-6)
        cosine_distance = 1.0 - (student * teacher).sum(dim=1).mean()
        loss = loss + float(weight) * cosine_distance
    return loss / weight_sum


def _normalized_target_boxes_xyxy(target, valid_h, valid_w):
    boxes = target.get("boxes")
    if boxes is None or boxes.numel() == 0:
        return None
    boxes = boxes.float()
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            (cx - 0.5 * width) * valid_w,
            (cy - 0.5 * height) * valid_h,
            (cx + 0.5 * width) * valid_w,
            (cy + 0.5 * height) * valid_h,
        ),
        dim=-1,
    )


def _feature_box_masks(feature, sample_mask, target, min_box_tokens=1):
    height, width = feature.shape[-2:]
    valid = ~F.interpolate(
        sample_mask[None, None].float(), size=(height, width), mode="nearest"
    )[0, 0].bool()
    valid_h = valid.any(dim=1).sum().float()
    valid_w = valid.any(dim=0).sum().float()
    boxes = _normalized_target_boxes_xyxy(target, valid_h, valid_w)
    if boxes is None:
        return valid, []

    ys, xs = torch.meshgrid(
        torch.arange(height, device=feature.device, dtype=torch.float32) + 0.5,
        torch.arange(width, device=feature.device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    box_masks = []
    for box in boxes:
        inside = (
            (xs >= box[0])
            & (xs <= box[2])
            & (ys >= box[1])
            & (ys <= box[3])
            & valid
        )
        if int(inside.sum()) >= int(min_box_tokens):
            box_masks.append(inside)
    return valid, box_masks


def object_conditioned_token_loss(
    features,
    sample_masks,
    targets,
    level_weights,
    margin=0.2,
    min_box_tokens=1,
):
    """Encourage object tokens to be cohesive and separated from background."""
    if len(features) != len(level_weights):
        raise ValueError("Object-token refinement requires one weight per feature level.")
    weight_sum = float(sum(level_weights))
    if weight_sum <= 0:
        raise ValueError("Object-token refinement level weights must have a positive sum.")

    total = features[0].new_zeros((), dtype=torch.float32)
    for feature, level_weight in zip(features, level_weights):
        tokens = F.normalize(feature.float(), dim=1, eps=1e-6)
        image_losses = []
        for image_index, target in enumerate(targets):
            valid, box_masks = _feature_box_masks(
                feature[image_index],
                sample_masks[image_index],
                target,
                min_box_tokens=min_box_tokens,
            )
            if not box_masks:
                continue
            union = torch.stack(box_masks).any(dim=0)
            background = valid & ~union
            background_proto = None
            if background.any():
                background_proto = F.normalize(
                    tokens[image_index, :, background].mean(dim=1), dim=0, eps=1e-6
                ).detach()

            box_losses = []
            for inside in box_masks:
                box_tokens = tokens[image_index, :, inside].transpose(0, 1)
                object_proto = F.normalize(
                    box_tokens.mean(dim=0), dim=0, eps=1e-6
                ).detach()
                object_similarity = box_tokens @ object_proto
                compactness = 1.0 - object_similarity.mean()
                if background_proto is None:
                    separation = compactness.new_zeros(())
                else:
                    background_similarity = box_tokens @ background_proto
                    separation = F.softplus(
                        float(margin) + background_similarity - object_similarity
                    ).mean()
                box_losses.append(compactness + separation)
            image_losses.append(torch.stack(box_losses).mean())
        if image_losses:
            total = total + float(level_weight) * torch.stack(image_losses).mean()
    return total / weight_sum


def object_local_token_loss(
    features,
    sample_masks,
    targets,
    level_weights,
    margin=0.2,
    min_box_tokens=1,
):
    """Encourage local object continuity without collapsing whole-object features."""
    if len(features) != len(level_weights):
        raise ValueError("Local object refinement requires one weight per feature level.")
    weight_sum = float(sum(level_weights))
    if weight_sum <= 0:
        raise ValueError("Local object refinement level weights must have a positive sum.")

    total = features[0].new_zeros((), dtype=torch.float32)
    for feature, level_weight in zip(features, level_weights):
        tokens = F.normalize(feature.float(), dim=1, eps=1e-6)
        image_losses = []
        for image_index, target in enumerate(targets):
            valid, box_masks = _feature_box_masks(
                feature[image_index],
                sample_masks[image_index],
                target,
                min_box_tokens=min_box_tokens,
            )
            if not box_masks:
                continue

            union = torch.stack(box_masks).any(dim=0)
            background = valid & ~union
            token_map = tokens[image_index]
            horizontal_similarity = (token_map[:, :, :-1] * token_map[:, :, 1:]).sum(dim=0)
            vertical_similarity = (token_map[:, :-1, :] * token_map[:, 1:, :]).sum(dim=0)

            box_losses = []
            for inside in box_masks:
                same_horizontal = inside[:, :-1] & inside[:, 1:]
                same_vertical = inside[:-1, :] & inside[1:, :]
                cross_horizontal = (
                    (inside[:, :-1] & background[:, 1:])
                    | (background[:, :-1] & inside[:, 1:])
                )
                cross_vertical = (
                    (inside[:-1, :] & background[1:, :])
                    | (background[:-1, :] & inside[1:, :])
                )

                same_values = []
                if same_horizontal.any():
                    same_values.append(horizontal_similarity[same_horizontal])
                if same_vertical.any():
                    same_values.append(vertical_similarity[same_vertical])
                cross_values = []
                if cross_horizontal.any():
                    cross_values.append(horizontal_similarity[cross_horizontal])
                if cross_vertical.any():
                    cross_values.append(vertical_similarity[cross_vertical])

                if same_values:
                    same_similarity = torch.cat(same_values).mean()
                    compactness = 1.0 - same_similarity
                else:
                    same_similarity = token_map.new_tensor(1.0)
                    compactness = token_map.new_zeros(())
                if cross_values:
                    cross_similarity = torch.cat(cross_values).mean()
                    separation = F.softplus(
                        float(margin) + cross_similarity - same_similarity.detach()
                    )
                else:
                    separation = compactness.new_zeros(())
                box_losses.append(compactness + separation)
            image_losses.append(torch.stack(box_losses).mean())
        if image_losses:
            total = total + float(level_weight) * torch.stack(image_losses).mean()
    return total / weight_sum


def object_weighted_feature_teacher_loss(
    student_features,
    teacher_features,
    sample_masks,
    targets,
    level_weights,
    background_weight=0.1,
    min_box_tokens=1,
):
    """Preserve refined feature behavior with box-balanced cosine distillation."""
    if len(student_features) != len(teacher_features):
        raise ValueError("Student and teacher must expose the same backbone levels.")
    if len(student_features) != len(level_weights):
        raise ValueError("Feature-teacher refinement requires one weight per feature level.")
    weight_sum = float(sum(level_weights))
    if weight_sum <= 0:
        raise ValueError("Feature-teacher level weights must have a positive sum.")

    total = student_features[0].new_zeros((), dtype=torch.float32)
    for student, teacher, level_weight in zip(
        student_features, teacher_features, level_weights
    ):
        if student.shape != teacher.shape:
            raise ValueError(
                "Student and teacher backbone feature shapes differ: "
                f"{tuple(student.shape)} != {tuple(teacher.shape)}."
            )
        student = F.normalize(student.float(), dim=1, eps=1e-6)
        teacher = F.normalize(teacher.float(), dim=1, eps=1e-6)
        distance = 1.0 - (student * teacher).sum(dim=1)
        image_losses = []
        for image_index, target in enumerate(targets):
            valid, box_masks = _feature_box_masks(
                student[image_index],
                sample_masks[image_index],
                target,
                min_box_tokens=min_box_tokens,
            )
            if not box_masks:
                image_losses.append(distance[image_index][valid].mean())
                continue
            object_loss = torch.stack(
                [distance[image_index][inside].mean() for inside in box_masks]
            ).mean()
            union = torch.stack(box_masks).any(dim=0)
            background = valid & ~union
            if background.any() and background_weight > 0:
                object_loss = object_loss + float(background_weight) * distance[
                    image_index
                ][background].mean()
            image_losses.append(object_loss)
        if image_losses:
            total = total + float(level_weight) * torch.stack(image_losses).mean()
    return total / weight_sum


def build_backbone_parameter_anchor(model: torch.nn.Module, block_indexes):
    from rfdetr.models.backbone.backbone import is_refined_backbone_parameter

    anchors = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and is_refined_backbone_parameter(name, block_indexes):
            anchors.append((name, parameter, parameter.detach().clone()))
    if block_indexes and not anchors:
        raise ValueError(
            f"No DINOv3 parameters matched refinement blocks {tuple(block_indexes)}."
        )
    return anchors


def backbone_parameter_anchor_loss(anchors):
    if not anchors:
        raise ValueError("Backbone parameter anchoring requires at least one parameter.")
    loss = anchors[0][1].new_zeros((), dtype=torch.float32)
    for _name, parameter, reference in anchors:
        delta = parameter.float() - reference.float()
        loss = loss + 0.5 * delta.square().sum()
    return loss


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    schedules: dict = {},
    num_training_steps_per_epoch=None,
    vit_encoder_num_layers=None,
    args=None,
    callbacks: DefaultDict[str, List[Callable]] = None,
    projector_distill_teacher: torch.nn.Module = None,
    backbone_parameter_anchor=None,
    backbone_refine_teacher: torch.nn.Module = None,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    print_freq = args.print_freq if args is not None else 10
    start_steps = epoch * num_training_steps_per_epoch

    # Add gradient scaler for AMP
    if DEPRECATED_AMP:
        scaler = GradScaler(enabled=args.amp)
    else:
        scaler = GradScaler("cuda", enabled=args.amp)

    optimizer.zero_grad()

    distill_active = (
        projector_distill_teacher is not None
        and float(getattr(args, "projector_distill_coef", 0.0)) > 0
        and epoch < int(getattr(args, "projector_distill_stop_epoch", 0))
    )
    anchor_active = (
        bool(backbone_parameter_anchor)
        and float(getattr(args, "backbone_refine_anchor_coef", 0.0)) > 0
        and epoch < int(getattr(args, "backbone_refine_anchor_stop_epoch", 0))
    )
    online_refine_mode = getattr(args, "online_refine_mode", "none")
    online_refine_active = (
        online_refine_mode != "none"
        and float(getattr(args, "online_refine_coef", 0.0)) > 0
        and epoch >= int(getattr(args, "online_refine_start_epoch", 0))
        and epoch < int(getattr(args, "online_refine_stop_epoch", 0))
    )
    online_refine_weight = 0.0
    if online_refine_active:
        start_epoch = int(args.online_refine_start_epoch)
        stop_epoch = int(args.online_refine_stop_epoch)
        remaining_fraction = (stop_epoch - epoch) / max(stop_epoch - start_epoch, 1)
        online_refine_weight = float(args.online_refine_coef) * remaining_fraction
    online_refine_positions = []
    if online_refine_active:
        output_indexes = list(args.out_feature_indexes)
        online_refine_positions = [
            output_indexes.index(layer) for layer in args.online_refine_layers
        ]
        if online_refine_mode == "feature_teacher" and backbone_refine_teacher is None:
            raise ValueError("feature_teacher mode requires a frozen backbone teacher.")

    student_projector_features = []
    student_backbone_features = []
    projector_hook = None
    backbone_hook = None
    if distill_active:
        student_model = model.module if hasattr(model, "module") else model

        def capture_projector_features(_module, _inputs, output):
            student_projector_features[:] = list(output)

        projector_hook = student_model.backbone[0].projector.register_forward_hook(
            capture_projector_features
        )
    if online_refine_active:
        student_model = model.module if hasattr(model, "module") else model

        def capture_backbone_features(_module, _inputs, output):
            student_backbone_features[:] = list(output)

        backbone_hook = student_model.backbone[0].encoder.register_forward_hook(
            capture_backbone_features
        )

    dense_o2o_augmenter = None
    if getattr(args, "use_dense_o2o", False):
        dense_o2o_augmenter = DenseO2OAugmenter(
            mode=args.dense_o2o_mode,
            start_epoch=args.dense_o2o_start_epoch,
            image_stop_epoch=args.dense_o2o_image_stop_epoch,
            copyblend_stop_epoch=args.dense_o2o_copyblend_stop_epoch,
            mosaic_prob=args.dense_o2o_mosaic_prob,
            mixup_prob=args.dense_o2o_mixup_prob,
            copyblend_prob=args.dense_o2o_copyblend_prob,
            copyblend_area_threshold=args.dense_o2o_copyblend_area_threshold,
            copyblend_num_objects=args.dense_o2o_copyblend_num_objects,
            copyblend_expand_ratios=args.dense_o2o_copyblend_expand_ratios,
            seed=args.seed,
        )

    # Check if batch size is divisible by gradient accumulation steps
    if batch_size % args.grad_accum_steps != 0:
        logger.error(
            f"Batch size ({batch_size}) must be divisible by gradient accumulation steps ({args.grad_accum_steps})"
        )
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by gradient accumulation steps ({args.grad_accum_steps})"
        )

    logger.info(
        f"Training config: grad_accum_steps={args.grad_accum_steps}, "
        f"total_batch_size={batch_size * get_world_size()}, "
        f"dataloader_length={len(data_loader)}"
    )

    sub_batch_size = batch_size // args.grad_accum_steps

    header = f"Epoch: [{epoch + 1}/{args.epochs}]"
    use_progress_bar = bool(getattr(args, "progress_bar", False))
    if use_progress_bar:
        progress_iter = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc=header,
            colour="green",
            disable=not utils.is_main_process(),
        )
    else:
        progress_iter = enumerate(metric_logger.log_every(data_loader, print_freq, header))

    for data_iter_step, (samples, targets) in progress_iter:
        it = start_steps + data_iter_step
        callback_dict = {
            "step": it,
            "model": model,
            "epoch": epoch,
        }
        for callback in callbacks["on_train_batch_start"]:
            callback(callback_dict)
        if "dp" in schedules:
            if args.distributed:
                model.module.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
            else:
                model.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
        if "do" in schedules:
            if args.distributed:
                model.module.update_dropout(schedules["do"][it])
            else:
                model.update_dropout(schedules["do"][it])

        if args.multi_scale and not args.do_random_resize_via_padding:
            scales = compute_multi_scale_scales(
                args.resolution, args.expanded_scales, args.patch_size, args.num_windows
            )
            random.seed(it)
            scale = random.choice(scales)
            with torch.no_grad():
                samples.tensors = F.interpolate(samples.tensors, size=scale, mode="bilinear", align_corners=False)
                samples.mask = (
                    F.interpolate(samples.mask.unsqueeze(1).float(), size=scale, mode="nearest").squeeze(1).bool()
                )

        dense_o2o_stats = {}
        if dense_o2o_augmenter is not None:
            augmented_tensors, augmented_masks, targets, dense_o2o_stats = dense_o2o_augmenter(
                samples.tensors,
                samples.mask,
                targets,
                epoch=epoch,
                step=data_iter_step,
                rank=utils.get_rank(),
            )
            samples = NestedTensor(augmented_tensors, augmented_masks)

        for i in range(args.grad_accum_steps):
            start_idx = i * sub_batch_size
            final_idx = start_idx + sub_batch_size
            new_samples_tensors = samples.tensors[start_idx:final_idx]
            new_samples = NestedTensor(new_samples_tensors, samples.mask[start_idx:final_idx])
            new_samples = new_samples.to(device)
            new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets[start_idx:final_idx]]

            teacher_features = None
            if distill_active:
                with torch.no_grad(), autocast(**get_autocast_args(args)):
                    teacher_nested_features = projector_distill_teacher.backbone[0](
                        new_samples
                    )
                teacher_features = [
                    feature.tensors.detach() for feature in teacher_nested_features
                ]

            backbone_teacher_features = None
            if online_refine_active and online_refine_mode == "feature_teacher":
                with torch.no_grad(), autocast(**get_autocast_args(args)):
                    backbone_teacher_features = backbone_refine_teacher(
                        new_samples.tensors
                    )
                backbone_teacher_features = [
                    backbone_teacher_features[position].detach()
                    for position in online_refine_positions
                ]

            with autocast(**get_autocast_args(args)):
                outputs = model(new_samples, new_targets)
                loss_dict = criterion(outputs, new_targets)
                weight_dict = dict(criterion.weight_dict)
                if distill_active:
                    if not student_projector_features:
                        raise RuntimeError("Student projector hook did not capture features.")
                    loss_dict["loss_projector_distill"] = projector_distillation_loss(
                        student_projector_features,
                        teacher_features,
                        args.projector_distill_level_weights,
                    )
                    weight_dict["loss_projector_distill"] = args.projector_distill_coef
                if anchor_active:
                    loss_dict["loss_backbone_anchor"] = backbone_parameter_anchor_loss(
                        backbone_parameter_anchor
                    )
                    weight_dict["loss_backbone_anchor"] = args.backbone_refine_anchor_coef
                if online_refine_active:
                    if not student_backbone_features:
                        raise RuntimeError("Student backbone hook did not capture features.")
                    selected_student_features = [
                        student_backbone_features[position]
                        for position in online_refine_positions
                    ]
                    if online_refine_mode == "object_token":
                        online_loss = object_conditioned_token_loss(
                            selected_student_features,
                            new_samples.mask,
                            new_targets,
                            args.online_refine_layer_weights,
                            margin=args.online_refine_margin,
                            min_box_tokens=args.online_refine_min_box_tokens,
                        )
                    elif online_refine_mode == "object_local":
                        online_loss = object_local_token_loss(
                            selected_student_features,
                            new_samples.mask,
                            new_targets,
                            args.online_refine_layer_weights,
                            margin=args.online_refine_margin,
                            min_box_tokens=args.online_refine_min_box_tokens,
                        )
                    elif online_refine_mode == "feature_teacher":
                        online_loss = object_weighted_feature_teacher_loss(
                            selected_student_features,
                            backbone_teacher_features,
                            new_samples.mask,
                            new_targets,
                            args.online_refine_layer_weights,
                            background_weight=args.online_refine_background_weight,
                            min_box_tokens=args.online_refine_min_box_tokens,
                        )
                    else:
                        raise ValueError(
                            f"Unsupported online_refine_mode: {online_refine_mode}"
                        )
                    loss_dict["loss_online_refine"] = online_loss
                    weight_dict["loss_online_refine"] = online_refine_weight
                losses = sum(
                    (1 / args.grad_accum_steps) * loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys()
                    if k in weight_dict
                )
                del outputs

            scaler.scale(losses).backward()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            logger.error(f"Loss is {loss_value}, stopping training. Loss dict: {loss_dict_reduced}")
            raise ValueError(f"Loss is {loss_value}, stopping training")

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
        if ema_m is not None:
            if epoch >= 0:
                ema_m.update(model)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

        if dense_o2o_stats:
            metric_logger.update(**dense_o2o_stats)
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if use_progress_bar:
            log_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            postfix = {
                "lr": f"{log_dict['lr']:.6f}",
                "class_loss": f"{log_dict['class_error']:.2f}",
                "box_loss": f"{log_dict['loss_bbox']:.2f}",
                "loss": f"{log_dict['loss']:.2f}",
            }
            if _is_cuda(device):
                postfix["max_mem"] = f"{torch.cuda.max_memory_allocated(device=device) / BYTES_TO_MB:.0f} MB"
            progress_iter.set_postfix(postfix)
    if projector_hook is not None:
        projector_hook.remove()
    if backbone_hook is not None:
        backbone_hook.remove()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Epoch {epoch + 1} stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def sweep_confidence_thresholds(per_class_data, conf_thresholds, classes_with_gt):
    """Sweep confidence thresholds and compute precision/recall/F1 at each."""
    num_classes = len(per_class_data)
    results = []

    for conf_thresh in conf_thresholds:
        per_class_precisions = []
        per_class_recalls = []
        per_class_f1s = []

        for k in range(num_classes):
            data = per_class_data[k]
            scores = data["scores"]
            matches = data["matches"]
            ignore = data["ignore"]
            total_gt = data["total_gt"]

            above_thresh = scores >= conf_thresh
            valid = above_thresh & ~ignore

            valid_matches = matches[valid]

            tp = np.sum(valid_matches != 0)
            fp = np.sum(valid_matches == 0)
            fn = total_gt - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_precisions.append(precision)
            per_class_recalls.append(recall)
            per_class_f1s.append(f1)

        if len(classes_with_gt) > 0:
            macro_precision = np.mean([per_class_precisions[k] for k in classes_with_gt])
            macro_recall = np.mean([per_class_recalls[k] for k in classes_with_gt])
            macro_f1 = np.mean([per_class_f1s[k] for k in classes_with_gt])
        else:
            macro_precision = 0.0
            macro_recall = 0.0
            macro_f1 = 0.0

        results.append(
            {
                "confidence_threshold": conf_thresh,
                "macro_f1": macro_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "per_class_prec": np.array(per_class_precisions),
                "per_class_rec": np.array(per_class_recalls),
                "per_class_f1": np.array(per_class_f1s),
            }
        )

    return results


def coco_extended_metrics(coco_eval):
    """
    Compute precision/recall by sweeping confidence thresholds to maximize macro-F1.
    Uses evalImgs directly to compute metrics from raw matching data.
    """

    def _get_iou_index(iou_value: float) -> int:
        matched = np.argwhere(np.isclose(coco_eval.params.iouThrs, iou_value))
        if matched.size == 0:
            raise ValueError(f"IoU threshold {iou_value:.2f} not found in COCO evaluator thresholds")
        return int(matched.item())

    def _safe_nanmean(values) -> float:
        arr = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(arr)
        if not finite.any():
            return float("nan")
        return float(arr[finite].mean())

    iou50_idx = _get_iou_index(0.50)
    iou75_idx = _get_iou_index(0.75)
    cat_ids = coco_eval.params.catIds
    num_classes = len(cat_ids)
    area_idx = 0
    maxdet_idx = 2

    # Unflatten evalImgs into a nested dict
    evalImgs_unflat = {}
    for e in coco_eval.evalImgs:
        if e is None:
            continue
        cat_id = e["category_id"]
        area_rng = tuple(e["aRng"])
        img_id = e["image_id"]

        if cat_id not in evalImgs_unflat:
            evalImgs_unflat[cat_id] = {}
        if area_rng not in evalImgs_unflat[cat_id]:
            evalImgs_unflat[cat_id][area_rng] = {}
        evalImgs_unflat[cat_id][area_rng][img_id] = e

    area_rng_all = tuple(coco_eval.params.areaRng[area_idx])

    per_class_data = []
    for cid in cat_ids:
        dt_scores = []
        dt_matches = []
        dt_ignore = []
        total_gt = 0

        for img_id in coco_eval.params.imgIds:
            e = evalImgs_unflat.get(cid, {}).get(area_rng_all, {}).get(img_id)
            if e is None:
                continue

            num_dt = len(e["dtIds"])
            # num_gt = len(e['gtIds'])

            gt_ignore = e["gtIgnore"]
            total_gt += sum(1 for ig in gt_ignore if not ig)

            for d in range(num_dt):
                dt_scores.append(e["dtScores"][d])
                dt_matches.append(e["dtMatches"][iou50_idx, d])
                dt_ignore.append(e["dtIgnore"][iou50_idx, d])

        per_class_data.append(
            {
                "scores": np.array(dt_scores),
                "matches": np.array(dt_matches),
                "ignore": np.array(dt_ignore, dtype=bool),
                "total_gt": total_gt,
            }
        )

    conf_thresholds = np.linspace(0.0, 1.0, 101)
    classes_with_gt = [k for k in range(num_classes) if per_class_data[k]["total_gt"] > 0]

    confidence_sweep_metric_dicts = sweep_confidence_thresholds(per_class_data, conf_thresholds, classes_with_gt)

    best = max(confidence_sweep_metric_dicts, key=lambda x: x["macro_f1"])

    map_50_95, map_50, map_75 = float(coco_eval.stats[0]), float(coco_eval.stats[1]), float(coco_eval.stats[2])

    per_class = []
    cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
    for k, cid in enumerate(cat_ids):
        # [T, R, K, A, M] -> [T, R]
        p_slice = coco_eval.eval["precision"][:, :, k, area_idx, maxdet_idx]

        # [T, R]
        p_masked = np.where(p_slice > -1, p_slice, np.nan)

        # We do this as two sequential nanmeans to avoid
        # underweighting columns with more nans, since each
        # column corresponds to a different IoU threshold
        # [T, R] -> [T]
        ap_per_iou = [_safe_nanmean(p_masked[i]) for i in range(p_masked.shape[0])]

        # [T] -> [1]
        ap_50_95 = _safe_nanmean(ap_per_iou)
        ap_50 = _safe_nanmean(p_masked[iou50_idx])
        ap_75 = _safe_nanmean(p_masked[iou75_idx])

        if (
            np.isnan(ap_50_95)
            or np.isnan(ap_50)
            or np.isnan(best["per_class_prec"][k])
            or np.isnan(best["per_class_rec"][k])
        ):
            continue

        per_class.append(
            {
                "class": cat_id_to_name[int(cid)],
                "map@50:95": ap_50_95,
                "map@50": ap_50,
                "map@75": ap_75,
                "precision": best["per_class_prec"][k],
                "recall": best["per_class_rec"][k],
                "f1_score": best["per_class_f1"][k],
            }
        )

    per_class.append(
        {
            "class": "all",
            "map@50:95": map_50_95,
            "map@50": map_50,
            "map@75": map_75,
            "precision": best["macro_precision"],
            "recall": best["macro_recall"],
            "f1_score": best["macro_f1"],
            "confidence_threshold": best["confidence_threshold"],
        }
    )

    return {
        "class_map": per_class,
        "map": map_50,
        "map@50:95": map_50_95,
        "map@50": map_50,
        "map@75": map_75,
        "precision": best["macro_precision"],
        "recall": best["macro_recall"],
        "f1_score": best["macro_f1"],
        "confidence_threshold": best["confidence_threshold"],
    }


def evaluate(model, criterion, postprocess, data_loader, base_ds, device, args=None, header="Eval"):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    iou_types = ("bbox",) if not args.segmentation_head else ("bbox", "segm")
    coco_evaluator = CocoEvaluator(base_ds, iou_types, args.eval_max_dets)

    print_freq = args.print_freq if args is not None else 10
    use_progress_bar = bool(getattr(args, "progress_bar", False))
    if use_progress_bar:
        progress_iter = tqdm(
            data_loader,
            total=len(data_loader),
            desc=header,
            colour="green",
            disable=not utils.is_main_process(),
        )
    else:
        progress_iter = metric_logger.log_every(data_loader, print_freq, header)

    for samples, targets in progress_iter:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        # Add autocast for evaluation
        with autocast(**get_autocast_args(args)):
            outputs = model(samples)

        if args.fp16_eval:
            for key in outputs.keys():
                if key == "enc_outputs":
                    for sub_key in outputs[key].keys():
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == "aux_outputs":
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx].keys():
                            outputs[key][idx][sub_key] = outputs[key][idx][sub_key].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results_all = postprocess(outputs, orig_target_sizes)
        res = {target["image_id"].item(): output for target, output in zip(targets, results_all)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if use_progress_bar:
            log_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            postfix = {
                "class_loss": f"{log_dict['class_error']:.2f}",
                "box_loss": f"{log_dict['loss_bbox']:.2f}",
                "loss": f"{log_dict['loss']:.2f}",
            }
            if _is_cuda(device):
                postfix["max_mem"] = f"{torch.cuda.max_memory_allocated(device) / BYTES_TO_MB:.0f} MB"
            progress_iter.set_postfix(postfix)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Evaluation results: {metric_logger}")
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        results_json = coco_extended_metrics(coco_evaluator.coco_eval["bbox"])
        stats["results_json"] = results_json
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

        if "segm" in iou_types:
            results_json_masks = coco_extended_metrics(coco_evaluator.coco_eval["segm"])
            stats["results_json_masks"] = results_json_masks
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    return stats, coco_evaluator
