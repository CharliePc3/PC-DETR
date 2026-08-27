import torch

from rfdetr.models.lwdetr import SetCriterion


def _criterion(mode):
    criterion = SetCriterion(
        num_classes=3,
        matcher=None,
        weight_dict={},
        focal_alpha=0.25,
        losses=["labels", "boxes"],
        group_detr=2,
        ia_bce_loss=True,
        matcher_quality_mode=mode,
        matcher_quality_start_epoch=2,
        matcher_quality_ramp_epoch=8,
        matcher_quality_thresholds=(0.1, 0.2, 0.3),
        matcher_quality_log=True,
    )
    criterion.train()
    criterion.set_epoch(8)
    return criterion


def _inputs(requires_grad=False):
    target_box = torch.tensor([0.5, 0.5, 0.4, 0.4])
    pred_boxes = torch.tensor(
        [
            [0.05, 0.05, 0.1, 0.1],
            [0.5, 0.5, 0.4, 0.4],
            [0.7, 0.7, 0.1, 0.1],
            [0.05, 0.05, 0.1, 0.1],
            [0.5, 0.5, 0.4, 0.4],
            [0.7, 0.7, 0.1, 0.1],
        ]
    ).unsqueeze(0)
    outputs = {
        "pred_boxes": pred_boxes,
        "pred_logits": torch.zeros(1, 6, 3, requires_grad=requires_grad),
    }
    targets = [
        {
            "boxes": target_box.unsqueeze(0),
            "labels": torch.tensor([1]),
            "size": torch.tensor([100, 100]),
        }
    ]
    indices = [(torch.tensor([0, 3]), torch.tensor([0, 0]))]
    return outputs, targets, indices


def test_aux_group_gate_preserves_primary_group_match():
    criterion = _criterion("aux_group_gate")
    outputs, targets, indices = _inputs()

    filtered, kept_fraction = criterion._gate_auxiliary_groups(
        outputs, targets, indices
    )

    torch.testing.assert_close(filtered[0][0], torch.tensor([0]))
    torch.testing.assert_close(filtered[0][1], torch.tensor([0]))
    assert kept_fraction == 0.5


def test_classification_ignore_masks_only_low_quality_queries():
    criterion = _criterion("cls_ignore")
    outputs, targets, indices = _inputs(requires_grad=True)
    accepted, ignored = criterion._classification_indices(outputs, targets, indices)

    assert accepted[0][0].numel() == 0
    torch.testing.assert_close(ignored[0], torch.tensor([0, 3]))

    loss = criterion.loss_labels(
        outputs,
        targets,
        accepted,
        num_boxes=2,
        ignored_indices=ignored,
    )["loss_ce"]
    loss.backward()

    assert torch.count_nonzero(outputs["pred_logits"].grad[0, [0, 3]]) == 0
    assert torch.count_nonzero(outputs["pred_logits"].grad[0, [1, 2, 4, 5]]) > 0
