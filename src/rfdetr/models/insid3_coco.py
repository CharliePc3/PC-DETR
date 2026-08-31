# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Turn INSID3 concept masks into COCO-style instance predictions.

The original INSID3 operation returns one concept-level patch mask for a
reference/target pair. This module supplies the missing instance protocol in
two forms:

* full-image few-shot prediction, where K support masks vote and connected
  components become instances;
* detection-guided prediction, where one RF-DETR box defines one target crop
  and therefore one intended instance.

The classes operate on pre-extracted DINOv3 features. Image loading, support
selection, detector inference, and COCO evaluation live in the accompanying
tool so the core logic remains unit-testable without a GPU or COCO files.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util

from rfdetr.models.insid3 import (
    DinoV3InContextSegmenter,
    InContextSegmentationResult,
    agglomerative_cluster_labels,
)


@dataclass(frozen=True)
class InContextSupport:
    """One annotated support represented in the shared DINOv3 feature space."""

    category_id: int
    features: torch.Tensor
    mask: torch.Tensor
    image_id: int | None = None
    annotation_id: int | None = None


@dataclass
class InstanceMaskPrediction:
    """One category-labelled binary mask and its ranking score."""

    category_id: int
    score: float
    mask: torch.Tensor


def connected_component_masks(
    mask: torch.Tensor,
    *,
    min_area: int = 1,
    connectivity: int = 8,
) -> list[torch.Tensor]:
    """Split a two-dimensional binary mask into area-sorted components."""

    if mask.ndim != 2:
        raise ValueError("mask must have shape [H, W].")
    if min_area < 1:
        raise ValueError("min_area must be positive.")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")

    array = mask.detach().bool().cpu().numpy()
    height, width = array.shape
    visited = np.zeros_like(array, dtype=bool)
    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    components: list[torch.Tensor] = []
    for start_y, start_x in zip(*np.nonzero(array & ~visited)):
        if visited[start_y, start_x]:
            continue
        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for dy, dx in offsets:
                neighbour_y, neighbour_x = y + dy, x + dx
                if (
                    0 <= neighbour_y < height
                    and 0 <= neighbour_x < width
                    and array[neighbour_y, neighbour_x]
                    and not visited[neighbour_y, neighbour_x]
                ):
                    visited[neighbour_y, neighbour_x] = True
                    queue.append((neighbour_y, neighbour_x))
        if len(pixels) < min_area:
            continue
        component = torch.zeros_like(mask, dtype=torch.bool, device="cpu")
        ys, xs = zip(*pixels)
        component[list(ys), list(xs)] = True
        components.append(component.to(mask.device))

    components.sort(key=lambda component: int(component.sum()), reverse=True)
    return components


def aggregate_support_results(
    results: Sequence[InContextSegmentationResult],
    *,
    merge_threshold: float,
    vote_threshold: float = 0.5,
    score_temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate K INSID3 outputs into a foreground mask and confidence map."""

    if not results:
        raise ValueError("at least one support result is required.")
    if not 0 < vote_threshold <= 1:
        raise ValueError("vote_threshold must lie in (0, 1].")
    if score_temperature <= 0:
        raise ValueError("score_temperature must be positive.")
    shape = results[0].mask.shape
    if any(result.mask.shape != shape for result in results):
        raise ValueError("all support results must have the same patch shape.")

    votes = torch.stack([result.mask.float() for result in results]).mean(dim=0)
    confidences = torch.stack(
        [
            torch.sigmoid(
                (result.score_map.float() - merge_threshold) / score_temperature
            )
            * result.mask.float()
            for result in results
        ]
    ).mean(dim=0)
    foreground = votes >= vote_threshold
    return foreground, confidences


def resize_binary_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize one binary mask with nearest-neighbour interpolation."""

    if mask.ndim != 2:
        raise ValueError("mask must have shape [H, W].")
    return (
        F.interpolate(mask[None, None].float(), size=size, mode="nearest")[0, 0]
        > 0.5
    )


def expanded_integer_box(
    box_xyxy: Sequence[float],
    image_size: tuple[int, int],
    *,
    expansion: float = 0.15,
) -> tuple[int, int, int, int]:
    """Expand and clip an ``xyxy`` detector box for PIL/tensor cropping."""

    if len(box_xyxy) != 4:
        raise ValueError("box_xyxy must contain four values.")
    if expansion < 0:
        raise ValueError("expansion must be non-negative.")
    image_height, image_width = image_size
    x0, y0, x1, y1 = (float(value) for value in box_xyxy)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box_xyxy must have positive width and height.")
    pad_x = (x1 - x0) * expansion
    pad_y = (y1 - y0) * expansion
    left = max(int(np.floor(x0 - pad_x)), 0)
    top = max(int(np.floor(y0 - pad_y)), 0)
    right = min(int(np.ceil(x1 + pad_x)), image_width)
    bottom = min(int(np.ceil(y1 + pad_y)), image_height)
    if right <= left or bottom <= top:
        raise ValueError("expanded box is empty after clipping.")
    return left, top, right, bottom


def paste_crop_mask(
    crop_mask: torch.Tensor,
    crop_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Resize a crop mask to its crop box and paste it into a full-image canvas."""

    left, top, right, bottom = crop_box
    image_height, image_width = image_size
    if not (0 <= left < right <= image_width and 0 <= top < bottom <= image_height):
        raise ValueError("crop_box must lie inside image_size.")
    resized = resize_binary_mask(crop_mask, (bottom - top, right - left))
    canvas = torch.zeros(
        image_height,
        image_width,
        dtype=torch.bool,
        device=crop_mask.device,
    )
    canvas[top:bottom, left:right] = resized
    return canvas


class _SupportAggregationMixin:
    def __init__(
        self,
        segmenter: DinoV3InContextSegmenter,
        *,
        num_debias_components: int = 0,
        cluster_similarity_threshold: float = 0.6,
        merge_threshold: float = 0.2,
        vote_threshold: float = 0.5,
        score_temperature: float = 0.1,
        min_component_patches: int = 1,
    ) -> None:
        if min_component_patches < 1:
            raise ValueError("min_component_patches must be positive.")
        self.segmenter = segmenter
        self.num_debias_components = int(num_debias_components)
        self.cluster_similarity_threshold = float(cluster_similarity_threshold)
        self.merge_threshold = float(merge_threshold)
        self.vote_threshold = float(vote_threshold)
        self.score_temperature = float(score_temperature)
        self.min_component_patches = int(min_component_patches)

    def _aggregate(
        self,
        supports: Sequence[InContextSupport],
        target_features: torch.Tensor,
        positional_basis: torch.Tensor | None,
        cluster_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not supports:
            raise ValueError("at least one support is required.")
        if cluster_labels is None:
            cluster_labels = agglomerative_cluster_labels(
                target_features.flatten(1).T,
                self.cluster_similarity_threshold,
            )
        results = [
            self.segmenter.predict_from_features(
                support.features,
                support.mask,
                target_features,
                positional_basis=positional_basis,
                num_debias_components=self.num_debias_components,
                cluster_similarity_threshold=self.cluster_similarity_threshold,
                merge_threshold=self.merge_threshold,
                cluster_labels=cluster_labels,
            )
            for support in supports
        ]
        return aggregate_support_results(
            results,
            merge_threshold=self.merge_threshold,
            vote_threshold=self.vote_threshold,
            score_temperature=self.score_temperature,
        )


class FullImageFewShotInstanceSegmenter(_SupportAggregationMixin):
    """Produce instances from full-image K-shot masks via connected components."""

    @torch.no_grad()
    def predict(
        self,
        supports: Sequence[InContextSupport],
        target_features: torch.Tensor,
        output_size: tuple[int, int],
        *,
        positional_basis: torch.Tensor | None = None,
        max_instances_per_category: int = 100,
    ) -> list[InstanceMaskPrediction]:
        grouped: dict[int, list[InContextSupport]] = defaultdict(list)
        for support in supports:
            grouped[int(support.category_id)].append(support)

        predictions: list[InstanceMaskPrediction] = []
        cluster_labels = agglomerative_cluster_labels(
            target_features.flatten(1).T,
            self.cluster_similarity_threshold,
        )
        for category_id, category_supports in sorted(grouped.items()):
            foreground, confidence = self._aggregate(
                category_supports,
                target_features,
                positional_basis,
                cluster_labels,
            )
            components = connected_component_masks(
                foreground,
                min_area=self.min_component_patches,
            )[:max_instances_per_category]
            for component in components:
                score = float(confidence[component].mean())
                predictions.append(
                    InstanceMaskPrediction(
                        category_id=category_id,
                        score=score,
                        mask=resize_binary_mask(component, output_size),
                    )
                )
        return predictions


class DetectionGuidedInContextSegmenter(_SupportAggregationMixin):
    """Use one detector crop as the spatial prompt for one INSID3 instance."""

    @torch.no_grad()
    def predict_crop(
        self,
        category_id: int,
        detection_score: float,
        supports: Sequence[InContextSupport],
        target_crop_features: torch.Tensor,
        crop_output_size: tuple[int, int],
        *,
        positional_basis: torch.Tensor | None = None,
    ) -> InstanceMaskPrediction | None:
        category_supports = [
            support
            for support in supports
            if int(support.category_id) == int(category_id)
        ]
        if not category_supports:
            return None
        foreground, confidence = self._aggregate(
            category_supports, target_crop_features, positional_basis
        )
        components = connected_component_masks(
            foreground,
            min_area=self.min_component_patches,
        )
        if not components:
            return None

        center_y = foreground.shape[0] // 2
        center_x = foreground.shape[1] // 2
        centered = [
            component
            for component in components
            if bool(component[center_y, center_x])
        ]
        candidates = centered if centered else components
        component = max(
            candidates,
            key=lambda candidate: float(confidence[candidate].sum()),
        )
        mask_quality = float(confidence[component].mean())
        return InstanceMaskPrediction(
            category_id=int(category_id),
            score=float(detection_score) * mask_quality,
            mask=resize_binary_mask(component, crop_output_size),
        )


def predictions_to_coco_results(
    image_id: int,
    predictions: Sequence[InstanceMaskPrediction],
) -> list[dict[str, object]]:
    """Encode instance predictions into JSON-safe COCO segmentation records."""

    results: list[dict[str, object]] = []
    for prediction in predictions:
        mask = prediction.mask.detach().bool().cpu().numpy().astype(np.uint8)
        if mask.ndim != 2:
            raise ValueError("prediction masks must have shape [H, W].")
        if not mask.any():
            continue
        rle = mask_util.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
        results.append(
            {
                "image_id": int(image_id),
                "category_id": int(prediction.category_id),
                "segmentation": rle,
                "score": float(prediction.score),
            }
        )
    return results
