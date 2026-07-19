# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Training-only Dense O2O augmentations adapted from DEIMv2.

Dense O2O increases the number of objects in each training image while retaining
ordinary one-to-one Hungarian matching. The ``image`` mode uses Mosaic and
MixUp; ``enhanced`` additionally uses DEIMv2-style object-level CopyBlend.
"""

import random
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


class DenseO2OAugmenter:
    def __init__(
        self,
        mode: str = "image",
        start_epoch: int = 2,
        image_stop_epoch: int = 12,
        copyblend_stop_epoch: int = 21,
        mosaic_prob: float = 0.5,
        mixup_prob: float = 0.5,
        copyblend_prob: float = 0.5,
        copyblend_area_threshold: float = 100.0,
        copyblend_num_objects: int = 3,
        copyblend_expand_ratios: Tuple[float, float] = (0.1, 0.25),
        seed: int = 42,
    ) -> None:
        if mode not in ("image", "enhanced"):
            raise ValueError("Dense O2O mode must be 'image' or 'enhanced'.")
        if not 0 <= start_epoch < image_stop_epoch:
            raise ValueError("Dense O2O requires 0 <= start_epoch < image_stop_epoch.")
        if mode == "enhanced" and copyblend_stop_epoch <= start_epoch:
            raise ValueError("Enhanced Dense O2O requires copyblend_stop_epoch > start_epoch.")
        for name, probability in (
            ("mosaic_prob", mosaic_prob),
            ("mixup_prob", mixup_prob),
            ("copyblend_prob", copyblend_prob),
        ):
            if not 0 <= probability <= 1:
                raise ValueError(f"{name} must be in [0, 1].")
        if copyblend_num_objects < 1:
            raise ValueError("copyblend_num_objects must be positive.")
        if (
            len(copyblend_expand_ratios) != 2
            or copyblend_expand_ratios[0] < 0
            or copyblend_expand_ratios[0] > copyblend_expand_ratios[1]
        ):
            raise ValueError("copyblend_expand_ratios must contain two non-negative increasing values.")

        self.mode = mode
        self.start_epoch = int(start_epoch)
        self.image_stop_epoch = int(image_stop_epoch)
        self.copyblend_stop_epoch = int(copyblend_stop_epoch)
        self.mosaic_prob = float(mosaic_prob)
        self.mixup_prob = float(mixup_prob)
        self.copyblend_prob = float(copyblend_prob)
        self.copyblend_area_threshold = float(copyblend_area_threshold)
        self.copyblend_num_objects = int(copyblend_num_objects)
        self.copyblend_expand_ratios = tuple(float(value) for value in copyblend_expand_ratios)
        self.seed = int(seed)

    @staticmethod
    def _clone_target(target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.clone() if torch.is_tensor(value) else value for key, value in target.items()}

    @classmethod
    def _set_instances(
        cls,
        target: Dict[str, torch.Tensor],
        boxes: torch.Tensor,
        labels: torch.Tensor,
        height: int,
        width: int,
    ) -> Dict[str, torch.Tensor]:
        updated = cls._clone_target(target)
        updated["boxes"] = boxes
        updated["labels"] = labels
        updated["area"] = boxes[:, 2].clamp(min=0) * width * boxes[:, 3].clamp(min=0) * height
        updated["iscrowd"] = torch.zeros(len(boxes), dtype=torch.int64, device=boxes.device)
        updated["size"] = torch.as_tensor([height, width], dtype=torch.int64, device=boxes.device)
        return updated

    @staticmethod
    def _validate_inputs(images: torch.Tensor, masks: torch.Tensor, targets: Sequence[Dict[str, torch.Tensor]]) -> None:
        if images.ndim != 4:
            raise ValueError(f"Dense O2O expects BCHW images, got shape {tuple(images.shape)}.")
        if len(images) != len(targets):
            raise ValueError("Dense O2O image and target batch sizes differ.")
        if len(images) < 2:
            raise ValueError("Dense O2O requires a batch containing at least two images.")
        if masks is not None and masks.any():
            raise ValueError("Dense O2O currently requires square, unpadded training batches.")
        if any("masks" in target for target in targets):
            raise ValueError("Dense O2O is not yet supported for segmentation targets.")

    def _apply_mosaic(
        self,
        images: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        rng: random.Random,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], int]:
        batch_size, _, height, width = images.shape
        if batch_size < 4 or self.mosaic_prob == 0:
            return images, targets, 0
        if height % 2 or width % 2:
            raise ValueError("Dense O2O Mosaic requires even image dimensions.")

        half_height, half_width = height // 2, width // 2
        resized = F.interpolate(images, size=(half_height, half_width), mode="bilinear", align_corners=False)
        output_images = images.clone()
        output_targets = [self._clone_target(target) for target in targets]
        mosaic_count = 0

        for destination in range(batch_size):
            if rng.random() >= self.mosaic_prob:
                continue
            candidates = [index for index in range(batch_size) if index != destination]
            source_indices = [destination, *rng.sample(candidates, 3)]
            rng.shuffle(source_indices)
            canvas = torch.empty_like(images[destination])
            boxes_list, labels_list = [], []

            for quadrant, source in enumerate(source_indices):
                row, column = divmod(quadrant, 2)
                y1, x1 = row * half_height, column * half_width
                canvas[:, y1 : y1 + half_height, x1 : x1 + half_width] = resized[source]

                source_boxes = targets[source]["boxes"].clone()
                source_boxes[:, 0] = (source_boxes[:, 0] + column) * 0.5
                source_boxes[:, 1] = (source_boxes[:, 1] + row) * 0.5
                source_boxes[:, 2:] *= 0.5
                boxes_list.append(source_boxes)
                labels_list.append(targets[source]["labels"].clone())

            output_images[destination] = canvas
            boxes = torch.cat(boxes_list, dim=0)
            labels = torch.cat(labels_list, dim=0)
            output_targets[destination] = self._set_instances(
                targets[destination], boxes, labels, height, width
            )
            mosaic_count += 1

        return output_images, output_targets, mosaic_count

    def _apply_mixup(
        self,
        images: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        rng: random.Random,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        beta = rng.uniform(0.45, 0.55)
        mixed_images = images * beta + images.roll(shifts=1, dims=0) * (1.0 - beta)
        shifted_targets = targets[-1:] + targets[:-1]
        height, width = images.shape[-2:]
        mixed_targets = []
        for target, shifted_target in zip(targets, shifted_targets):
            boxes = torch.cat([target["boxes"], shifted_target["boxes"]], dim=0)
            labels = torch.cat([target["labels"], shifted_target["labels"]], dim=0)
            mixed_targets.append(self._set_instances(target, boxes, labels, height, width))
        return mixed_images, mixed_targets

    def _apply_copyblend(
        self,
        images: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        rng: random.Random,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], int]:
        height, width = images.shape[-2:]
        object_pool = []
        for image_index, target in enumerate(targets):
            for box, label in zip(target["boxes"], target["labels"]):
                pixel_area = float(box[2] * width * box[3] * height)
                if pixel_area >= self.copyblend_area_threshold:
                    object_pool.append((image_index, box, label))
        if not object_pool:
            return images, targets, 0

        output_images = images.clone()
        output_targets = [self._clone_target(target) for target in targets]
        beta = rng.uniform(0.45, 0.55)
        copied_objects = 0

        for destination in range(len(images)):
            selected = rng.sample(object_pool, min(self.copyblend_num_objects, len(object_pool)))
            new_boxes, new_labels = [], []
            for source_index, source_box, source_label in selected:
                cx, cy, box_width, box_height = source_box.detach().cpu().tolist()
                object_x1 = max(0, int(round((cx - box_width / 2) * width)))
                object_y1 = max(0, int(round((cy - box_height / 2) * height)))
                object_x2 = min(width, int(round((cx + box_width / 2) * width)))
                object_y2 = min(height, int(round((cy + box_height / 2) * height)))
                if object_x2 <= object_x1 or object_y2 <= object_y1:
                    continue

                expand_ratio = rng.uniform(*self.copyblend_expand_ratios)
                margin_x = int(round((object_x2 - object_x1) * expand_ratio))
                margin_y = int(round((object_y2 - object_y1) * expand_ratio))
                patch_x1 = max(0, object_x1 - margin_x)
                patch_y1 = max(0, object_y1 - margin_y)
                patch_x2 = min(width, object_x2 + margin_x)
                patch_y2 = min(height, object_y2 + margin_y)
                patch_width, patch_height = patch_x2 - patch_x1, patch_y2 - patch_y1
                if patch_width <= 0 or patch_height <= 0:
                    continue

                destination_x1 = rng.randint(0, width - patch_width) if patch_width < width else 0
                destination_y1 = rng.randint(0, height - patch_height) if patch_height < height else 0
                destination_x2 = destination_x1 + patch_width
                destination_y2 = destination_y1 + patch_height
                source_patch = images[source_index, :, patch_y1:patch_y2, patch_x1:patch_x2]
                destination_patch = output_images[
                    destination, :, destination_y1:destination_y2, destination_x1:destination_x2
                ]
                output_images[destination, :, destination_y1:destination_y2, destination_x1:destination_x2] = (
                    destination_patch * beta + source_patch * (1.0 - beta)
                )

                new_object_x1 = destination_x1 + object_x1 - patch_x1
                new_object_y1 = destination_y1 + object_y1 - patch_y1
                new_object_x2 = destination_x1 + object_x2 - patch_x1
                new_object_y2 = destination_y1 + object_y2 - patch_y1
                new_box = source_box.new_tensor(
                    [
                        (new_object_x1 + new_object_x2) / (2 * width),
                        (new_object_y1 + new_object_y2) / (2 * height),
                        (new_object_x2 - new_object_x1) / width,
                        (new_object_y2 - new_object_y1) / height,
                    ]
                )
                new_boxes.append(new_box)
                new_labels.append(source_label.clone())

            if new_boxes:
                boxes = torch.cat([targets[destination]["boxes"], torch.stack(new_boxes)], dim=0)
                labels = torch.cat([targets[destination]["labels"], torch.stack(new_labels)], dim=0)
                output_targets[destination] = self._set_instances(
                    targets[destination], boxes, labels, height, width
                )
                copied_objects += len(new_boxes)

        return output_images, output_targets, copied_objects

    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        targets: Sequence[Dict[str, torch.Tensor]],
        epoch: int,
        step: int,
        rank: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]], Dict[str, float]]:
        self._validate_inputs(images, masks, targets)
        targets = [self._clone_target(target) for target in targets]
        image_active = self.start_epoch <= epoch < self.image_stop_epoch
        copyblend_active = (
            self.mode == "enhanced" and self.start_epoch <= epoch < self.copyblend_stop_epoch
        )
        objects_before = sum(len(target["boxes"]) for target in targets)
        stats = {
            "dense_o2o_active": float(image_active or copyblend_active),
            "dense_o2o_objects_before": float(objects_before),
            "dense_o2o_objects_after": float(objects_before),
            "dense_o2o_mosaic_images": 0.0,
            "dense_o2o_mixup": 0.0,
            "dense_o2o_copyblend_objects": 0.0,
        }
        if not image_active and not copyblend_active:
            return images, masks, targets, stats

        rng_seed = self.seed + rank * 10000019 + epoch * 1000003 + step * 9176
        rng = random.Random(rng_seed)
        if image_active:
            images, targets, mosaic_count = self._apply_mosaic(images, targets, rng)
            stats["dense_o2o_mosaic_images"] = float(mosaic_count)

        if image_active and rng.random() < self.mixup_prob:
            images, targets = self._apply_mixup(images, targets, rng)
            stats["dense_o2o_mixup"] = 1.0
        elif copyblend_active and rng.random() < self.copyblend_prob:
            images, targets, copied_objects = self._apply_copyblend(images, targets, rng)
            stats["dense_o2o_copyblend_objects"] = float(copied_objects)

        stats["dense_o2o_objects_after"] = float(sum(len(target["boxes"]) for target in targets))
        return images, masks, targets, stats
