import torch

from rfdetr.models.dn_components import prepare_for_cdn


def _target(num_boxes):
    sizes = torch.linspace(0.01, 0.4, steps=num_boxes)
    boxes = torch.stack(
        [
            torch.full_like(sizes, 0.5),
            torch.full_like(sizes, 0.5),
            sizes,
            sizes,
        ],
        dim=1,
    )
    return {"labels": torch.arange(num_boxes) % 10, "boxes": boxes}


def _prepare(group_detr, budget):
    return prepare_for_cdn(
        targets=[_target(40), _target(8)],
        dn_number=25,
        label_noise_scale=0.0,
        box_noise_scale=0.0,
        num_queries=300,
        num_classes=10,
        hidden_dim=32,
        label_embed=torch.nn.Embedding(10, 32),
        bbox_reparam=True,
        dn_negative=True,
        group_detr=group_detr,
        dn_total_query_budget=budget,
    )


def test_group_total_budget_caps_dense_cdn_queries():
    labels, boxes, _, meta = _prepare(group_detr=6, budget=300)

    assert meta["original_max_gt"] == 40
    assert meta["max_gt"] == 25
    assert meta["truncated_gt_count"] == 15
    assert meta["pad_size"] == 50
    assert meta["total_dn_queries"] == 300
    assert labels.shape == (2, 300, 32)
    assert boxes.shape == (2, 300, 4)
    assert int(meta["tgt_indices"].max()) < 48


def test_same_total_budget_is_group_count_invariant():
    _, _, _, group5 = _prepare(group_detr=5, budget=300)
    _, _, _, group6 = _prepare(group_detr=6, budget=300)

    assert group5["pad_size"] == 60
    assert group6["pad_size"] == 50
    assert group5["total_dn_queries"] == group6["total_dn_queries"] == 300


def test_zero_budget_preserves_legacy_unbounded_behavior():
    labels, _, _, meta = _prepare(group_detr=6, budget=0)

    assert meta["max_gt"] == 40
    assert meta["pad_size"] == 80
    assert meta["total_dn_queries"] == 480
    assert meta["truncated_gt_count"] == 0
    assert labels.shape[1] == 480
