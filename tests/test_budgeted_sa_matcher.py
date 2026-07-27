import torch

from rfdetr.models.matcher import HungarianMatcher


def _outputs(num_queries):
    return {
        "pred_logits": torch.zeros(1, num_queries, 1),
        "pred_boxes": torch.full((1, num_queries, 4), 0.25),
    }


def _targets():
    return [
        {
            "labels": torch.zeros(3, dtype=torch.int64),
            "boxes": torch.tensor(
                [
                    [0.2, 0.2, 0.1, 0.1],
                    [0.5, 0.5, 0.4, 0.4],
                    [0.5, 0.5, 0.98, 0.98],
                ]
            ),
            "size": torch.tensor([100, 100]),
            "image_id": torch.tensor([7]),
        }
    ]


def _matcher():
    return HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)


def test_budgeted_sa_supports_single_group_auxiliary_matching():
    indices, info = _matcher()(
        _outputs(12),
        _targets(),
        group_detr=1,
        sa_total_budgets=(1, 3, 4),
        sa_area_thresholds=(32**2, 96**2),
        return_match_info=True,
    )

    src, tgt = indices[0]
    assert len(src) == 8
    assert len(src.unique()) == 8
    assert torch.bincount(tgt, minlength=3).tolist() == [1, 3, 4]
    assert info == {
        "sa_requested": 5,
        "sa_matched": 5,
        "sa_matched_small": 0,
        "sa_matched_medium": 2,
        "sa_matched_large": 3,
    }


def test_grouped_sa_keeps_extra_matches_out_of_inference_group():
    indices, info = _matcher()(
        _outputs(24),
        _targets(),
        group_detr=2,
        sa_total_budgets=(2, 3, 4),
        sa_area_thresholds=(32**2, 96**2),
        return_match_info=True,
    )

    src, tgt = indices[0]
    assert len(src) == 9
    assert len(src.unique()) == 9
    assert torch.bincount(tgt, minlength=3).tolist() == [2, 3, 4]
    assert torch.all(src[-3:] >= 12)
    assert info["sa_matched"] == 3
