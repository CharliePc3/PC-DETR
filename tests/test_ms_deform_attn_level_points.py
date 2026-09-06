import torch

from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn


def _run_attention(n_points):
    torch.manual_seed(17)
    module = MSDeformAttn(
        d_model=32,
        n_levels=3,
        n_heads=4,
        n_points=n_points,
        scale_routing=True,
    )
    query = torch.randn(2, 5, 32, requires_grad=True)
    value = torch.randn(2, 21, 32, requires_grad=True)
    spatial_shapes = torch.tensor([[4, 4], [2, 2], [1, 1]], dtype=torch.long)
    level_starts = torch.tensor([0, 16, 20], dtype=torch.long)
    reference_boxes = torch.rand(2, 5, 3, 4)
    output = module(
        query,
        reference_boxes,
        value,
        spatial_shapes,
        level_starts,
    )
    output.square().mean().backward()
    return module, output, query.grad, value.grad


def test_scalar_and_uniform_per_level_points_are_bitwise_equal():
    scalar, scalar_output, scalar_query_grad, scalar_value_grad = _run_attention(2)
    per_level, per_level_output, per_level_query_grad, per_level_value_grad = (
        _run_attention([2, 2, 2])
    )

    assert scalar.state_dict().keys() == per_level.state_dict().keys()
    for name, value in scalar.state_dict().items():
        assert torch.equal(value, per_level.state_dict()[name]), name
    assert torch.equal(scalar_output, per_level_output)
    assert torch.equal(scalar_query_grad, per_level_query_grad)
    assert torch.equal(scalar_value_grad, per_level_value_grad)


def test_heterogeneous_per_level_points_support_forward_and_backward():
    module, output, query_grad, value_grad = _run_attention([2, 3, 1])

    assert module.points_per_level == (2, 3, 1)
    assert output.shape == (2, 5, 32)
    assert torch.isfinite(output).all()
    assert torch.isfinite(query_grad).all()
    assert torch.isfinite(value_grad).all()
