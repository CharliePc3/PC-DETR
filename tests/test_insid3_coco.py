# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
from pycocotools import mask as mask_util

from rfdetr.models.insid3 import InContextSegmentationResult
from rfdetr.models.insid3_coco import (
    DetectionGuidedInContextSegmenter,
    FullImageFewShotInstanceSegmenter,
    InContextSupport,
    InstanceMaskPrediction,
    aggregate_support_results,
    connected_component_masks,
    expanded_integer_box,
    paste_crop_mask,
    predictions_to_coco_results,
)


def make_result(mask: torch.Tensor, foreground_score: float = 0.8):
    labels = torch.arange(mask.numel()).reshape(mask.shape)
    score_map = torch.where(
        mask,
        torch.full_like(mask, foreground_score, dtype=torch.float32),
        torch.zeros_like(mask, dtype=torch.float32),
    )
    return InContextSegmentationResult(
        mask=mask,
        candidate_mask=mask,
        cluster_labels=labels,
        seed_cluster=0,
        num_clusters=mask.numel(),
        num_candidate_patches=int(mask.sum()),
        cluster_scores=score_map.flatten(),
        score_map=score_map,
    )


class FakeInContextSegmenter:
    def predict_from_features(
        self,
        reference_features,
        reference_mask,
        target_features,
        **kwargs,
    ):
        del reference_features, target_features, kwargs
        return make_result(reference_mask.bool())


def test_connected_components_respects_connectivity_and_area():
    mask = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ],
        dtype=torch.bool,
    )

    components_4 = connected_component_masks(mask, connectivity=4)
    components_8 = connected_component_masks(mask, connectivity=8, min_area=2)

    assert [int(component.sum()) for component in components_4] == [2, 1, 1]
    assert [int(component.sum()) for component in components_8] == [2, 2]


def test_support_aggregation_requires_consensus():
    first = make_result(torch.tensor([[1, 1], [0, 0]], dtype=torch.bool))
    second = make_result(torch.tensor([[1, 0], [1, 0]], dtype=torch.bool))

    foreground, confidence = aggregate_support_results(
        [first, second],
        merge_threshold=0.2,
        vote_threshold=1.0,
    )

    torch.testing.assert_close(
        foreground, torch.tensor([[1, 0], [0, 0]], dtype=torch.bool)
    )
    assert float(confidence[0, 0]) > float(confidence[0, 1])


def test_full_image_few_shot_splits_instances():
    support_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ],
        dtype=torch.bool,
    )
    support = InContextSupport(
        category_id=7,
        features=torch.ones(2, 4, 4),
        mask=support_mask,
        image_id=3,
    )
    predictor = FullImageFewShotInstanceSegmenter(
        FakeInContextSegmenter(),
        min_component_patches=2,
    )

    predictions = predictor.predict(
        [support],
        torch.ones(2, 4, 4),
        (8, 8),
        max_instances_per_category=5,
    )

    assert len(predictions) == 2
    assert all(prediction.category_id == 7 for prediction in predictions)
    assert [int(prediction.mask.sum()) for prediction in predictions] == [16, 8]


def test_detection_guided_prefers_center_component():
    support_mask = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    support = InContextSupport(
        category_id=4,
        features=torch.ones(2, 6, 6),
        mask=support_mask,
    )
    predictor = DetectionGuidedInContextSegmenter(FakeInContextSegmenter())

    prediction = predictor.predict_crop(
        4,
        0.8,
        [support],
        torch.ones(2, 6, 6),
        (12, 12),
    )

    assert prediction is not None
    assert int(prediction.mask.sum()) == 4
    assert 0 < prediction.score <= 0.8


def test_expanded_box_and_crop_paste_are_clipped():
    box = expanded_integer_box((-2.0, 1.0, 8.0, 9.0), (10, 12), expansion=0.25)
    pasted = paste_crop_mask(torch.ones(2, 2, dtype=torch.bool), box, (10, 12))

    assert box == (0, 0, 11, 10)
    assert pasted.shape == (10, 12)
    assert int(pasted.sum()) == 110


def test_coco_results_round_trip_rle():
    mask = torch.zeros(5, 6, dtype=torch.bool)
    mask[1:4, 2:5] = True
    records = predictions_to_coco_results(
        11,
        [InstanceMaskPrediction(category_id=3, score=0.7, mask=mask)],
    )

    assert len(records) == 1
    assert records[0]["image_id"] == 11
    decoded = mask_util.decode(records[0]["segmentation"])
    torch.testing.assert_close(torch.from_numpy(decoded).bool(), mask)
