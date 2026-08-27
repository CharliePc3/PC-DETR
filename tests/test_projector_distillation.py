import pytest
import torch

from rfdetr.engine import projector_distillation_loss


def test_projector_distillation_is_zero_for_identical_features():
    student = [torch.randn(2, 8, 4, 4) for _ in range(3)]
    teacher = [feature.detach().clone() for feature in student]

    loss = projector_distillation_loss(student, teacher, [0.5, 1.0, 0.5])

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)


def test_projector_distillation_backpropagates_only_to_student():
    student = [torch.randn(2, 8, 4, 4, requires_grad=True) for _ in range(3)]
    teacher = [torch.randn(2, 8, 4, 4, requires_grad=False) for _ in range(3)]

    loss = projector_distillation_loss(student, teacher, [1.0, 1.0, 1.0])
    loss.backward()

    assert loss.item() > 0
    assert all(feature.grad is not None for feature in student)
    assert all(feature.grad.abs().sum() > 0 for feature in student)
    assert all(feature.grad is None for feature in teacher)


def test_projector_distillation_rejects_mismatched_shapes():
    student = [torch.randn(1, 8, 4, 4)]
    teacher = [torch.randn(1, 8, 2, 2)]

    with pytest.raises(ValueError, match="feature shapes differ"):
        projector_distillation_loss(student, teacher, [1.0])


def test_projector_distillation_rejects_invalid_level_weights():
    features = [torch.randn(1, 8, 4, 4)]

    with pytest.raises(ValueError, match="one weight per feature level"):
        projector_distillation_loss(features, features, [1.0, 1.0])
    with pytest.raises(ValueError, match="positive sum"):
        projector_distillation_loss(features, features, [0.0])
