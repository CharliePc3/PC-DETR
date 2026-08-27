import torch

from rfdetr.engine import (
    object_conditioned_token_loss,
    object_local_token_loss,
    object_weighted_feature_teacher_loss,
)


def _target():
    return {"boxes": torch.tensor([[0.25, 0.25, 0.5, 0.5]])}


def test_object_conditioned_token_loss_is_finite_and_backpropagates():
    feature = torch.randn(1, 4, 4, 4, requires_grad=True)
    loss = object_conditioned_token_loss(
        [feature],
        torch.zeros(1, 4, 4, dtype=torch.bool),
        [_target()],
        [1.0],
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()


def test_object_local_token_loss_is_finite_and_backpropagates():
    feature = torch.randn(1, 4, 4, 4, requires_grad=True)
    loss = object_local_token_loss(
        [feature],
        torch.zeros(1, 4, 4, dtype=torch.bool),
        [_target()],
        [1.0],
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()


def test_feature_teacher_loss_is_zero_for_identical_features():
    feature = torch.randn(1, 4, 4, 4, requires_grad=True)
    loss = object_weighted_feature_teacher_loss(
        [feature],
        [feature.detach().clone()],
        torch.zeros(1, 4, 4, dtype=torch.bool),
        [_target()],
        [1.0],
    )

    assert loss.item() < 1e-6


def test_feature_teacher_loss_detects_foreground_drift():
    teacher = torch.zeros(1, 2, 4, 4)
    teacher[:, 0] = 1.0
    student = teacher.clone()
    student[:, :, :2, :2] = torch.tensor([0.0, 1.0]).view(1, 2, 1, 1)
    student.requires_grad_(True)

    loss = object_weighted_feature_teacher_loss(
        [student],
        [teacher],
        torch.zeros(1, 4, 4, dtype=torch.bool),
        [_target()],
        [1.0],
        background_weight=0.0,
    )

    assert loss.item() > 0.9
    loss.backward()
    assert student.grad is not None
