import pytest
import torch

from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn
from rfdetr.models.transformer import Transformer


def _attention_inputs(d_model=32, n_levels=3):
    spatial_shapes = torch.tensor([[8, 8], [4, 4], [2, 2]], dtype=torch.long)
    level_start_index = torch.cat(
        [spatial_shapes.new_zeros(1), spatial_shapes.prod(1).cumsum(0)[:-1]]
    )
    input_length = int(spatial_shapes.prod(1).sum())
    query = torch.randn(2, 5, d_model)
    reference_points = torch.rand(2, 5, n_levels, 4)
    reference_points[..., 2:] = reference_points[..., 2:] * 0.4 + 0.05
    input_flatten = torch.randn(2, input_length, d_model)
    return query, reference_points, input_flatten, spatial_shapes, level_start_index


@pytest.mark.parametrize("routing_mode", ["legacy", "cell"])
def test_zero_initialized_scale_routing_preserves_attention(routing_mode):
    torch.manual_seed(7)
    baseline = MSDeformAttn(d_model=32, n_levels=3, n_heads=4, n_points=2)
    torch.manual_seed(7)
    routed = MSDeformAttn(
        d_model=32,
        n_levels=3,
        n_heads=4,
        n_points=2,
        scale_routing=True,
        scale_routing_mode=routing_mode,
    )

    inputs = _attention_inputs()
    baseline_output = baseline(*inputs)
    routed_output = routed(*inputs)

    torch.testing.assert_close(routed_output, baseline_output)


def test_cell_scale_routing_receives_gradients():
    attention = MSDeformAttn(
        d_model=32,
        n_levels=3,
        n_heads=4,
        n_points=2,
        scale_routing=True,
        scale_routing_mode="cell",
    )
    inputs = _attention_inputs()

    attention(*inputs).square().mean().backward()

    assert attention.scale_router[-1].weight.grad is not None
    assert torch.count_nonzero(attention.scale_router[-1].weight.grad) > 0


def test_scale_routing_can_be_limited_to_early_decoder_layers():
    transformer = Transformer(
        d_model=32,
        sa_nhead=4,
        ca_nhead=4,
        num_decoder_layers=4,
        dim_feedforward=64,
        num_feature_levels=3,
        dec_n_points=2,
        scale_routing=True,
        scale_routing_layers=[0, 1, 2],
    )

    routers = [layer.cross_attn.scale_router for layer in transformer.decoder.layers]
    assert all(router is not None for router in routers[:3])
    assert routers[3] is None


def test_scale_router_does_not_perturb_shared_transformer_initialization():
    kwargs = {
        "d_model": 32,
        "sa_nhead": 4,
        "ca_nhead": 4,
        "num_decoder_layers": 4,
        "dim_feedforward": 64,
        "num_feature_levels": 3,
        "dec_n_points": 2,
    }
    torch.manual_seed(42)
    baseline = Transformer(**kwargs)
    torch.manual_seed(42)
    routed = Transformer(**kwargs, scale_routing=True)

    baseline_parameters = dict(baseline.named_parameters())
    routed_parameters = dict(routed.named_parameters())
    for name, parameter in baseline_parameters.items():
        torch.testing.assert_close(routed_parameters[name], parameter, rtol=0, atol=0)


def test_scale_routing_layers_require_routing():
    with pytest.raises(ValueError, match="require scale_routing"):
        Transformer(
            d_model=32,
            sa_nhead=4,
            ca_nhead=4,
            num_decoder_layers=4,
            dim_feedforward=64,
            num_feature_levels=3,
            dec_n_points=2,
            scale_routing=False,
            scale_routing_layers=[0, 1, 2],
        )


def test_last_level_attention_bias_only_cold_starts_p5():
    attention = MSDeformAttn(
        d_model=32,
        n_levels=3,
        n_heads=4,
        n_points=2,
        last_level_initial_bias=-2.0,
    )

    bias = attention.attention_weights.bias.view(4, 3, 2)

    torch.testing.assert_close(bias[:, :2], torch.zeros_like(bias[:, :2]))
    torch.testing.assert_close(bias[:, -1], torch.full_like(bias[:, -1], -2.0))
    assert attention.attention_weights.bias.requires_grad
