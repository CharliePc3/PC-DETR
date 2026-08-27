import pytest
import torch

from rfdetr.models.backbone.projector import MultiScaleProjector


@pytest.mark.parametrize(
    "p5_mode",
    [
        "full",
        "pool",
        "dwconv",
        "group2",
        "group2_mix",
        "group2_mix128",
        "group2_fullmix",
        "group2_first_full",
        "group2_first_full_dw",
        "group2_first2_full",
        "group2_last_full",
        "group4",
        "group8",
        "fusion",
        "fusion_wide",
        "fusion_refine",
        "fusion_residual",
    ],
)
def test_p5_modes_produce_the_same_pyramid_shapes(p5_mode):
    projector = MultiScaleProjector(
        [32] * 4,
        24,
        [2.0, 1.0, 0.5],
        p5_mode=p5_mode,
        layer_norm=True,
    )
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]

    outputs = projector(features)

    assert [output.shape for output in outputs] == [
        (2, 24, 18, 22),
        (2, 24, 9, 11),
        (2, 24, 5, 6),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)


def test_lightweight_p5_reduces_parameters():
    kwargs = {
        "in_channels": [384] * 4,
        "out_channels": 256,
        "scale_factors": [2.0, 1.0, 0.5],
        "layer_norm": True,
    }
    full = MultiScaleProjector(**kwargs, p5_mode="full")
    pooled = MultiScaleProjector(**kwargs, p5_mode="pool")
    depthwise = MultiScaleProjector(**kwargs, p5_mode="dwconv")
    group2 = MultiScaleProjector(**kwargs, p5_mode="group2")
    group2_mix = MultiScaleProjector(**kwargs, p5_mode="group2_mix")
    group2_mix128 = MultiScaleProjector(**kwargs, p5_mode="group2_mix128")
    group2_fullmix = MultiScaleProjector(**kwargs, p5_mode="group2_fullmix")
    group2_first_full = MultiScaleProjector(**kwargs, p5_mode="group2_first_full")
    group2_first2_full = MultiScaleProjector(**kwargs, p5_mode="group2_first2_full")
    group2_last_full = MultiScaleProjector(**kwargs, p5_mode="group2_last_full")
    group4 = MultiScaleProjector(**kwargs, p5_mode="group4")
    group8 = MultiScaleProjector(**kwargs, p5_mode="group8")
    fusion = MultiScaleProjector(**kwargs, p5_mode="fusion")
    wide_fusion = MultiScaleProjector(**kwargs, p5_mode="fusion_wide")
    refined_fusion = MultiScaleProjector(**kwargs, p5_mode="fusion_refine")
    residual_fusion = MultiScaleProjector(**kwargs, p5_mode="fusion_residual")

    count = lambda module: sum(parameter.numel() for parameter in module.parameters())

    assert count(pooled) < count(depthwise) < count(full)
    assert count(depthwise) < count(fusion) < count(full)
    assert count(fusion) < count(wide_fusion) < count(residual_fusion)
    assert count(fusion) < count(refined_fusion) < count(full)
    assert count(fusion) < count(residual_fusion) < count(full)
    assert (
        count(residual_fusion)
        < count(group8)
        < count(group4)
        < count(group2)
        < count(group2_mix)
        < count(group2_mix128)
        < count(group2_fullmix)
        < count(group2_last_full)
        < count(group2_first2_full)
        < count(full)
    )
    assert count(group2_first_full) == count(group2_last_full)
    assert count(full) - count(depthwise) > 6_000_000
    assert count(full) - count(fusion) > 6_000_000
    assert count(full) - count(wide_fusion) > 6_000_000
    assert count(full) - count(refined_fusion) > 5_000_000
    assert count(full) - count(residual_fusion) > 5_000_000
    assert count(full) - count(group8) > 4_500_000
    assert count(full) - count(group4) > 3_500_000
    assert count(full) - count(group2) > 2_500_000
    assert count(full) - count(group2_mix) > 2_400_000
    assert count(full) - count(group2_mix128) > 2_200_000
    assert count(full) - count(group2_fullmix) > 2_000_000
    assert count(full) - count(group2_first_full) > 1_900_000
    assert count(full) - count(group2_first2_full) > 1_300_000
    assert count(full) - count(group2_last_full) > 1_900_000


@pytest.mark.parametrize(
    "p5_mode",
    [
        "pool",
        "dwconv",
        "group2",
        "group2_mix",
        "group2_mix128",
        "group2_fullmix",
        "group2_first_full",
        "group2_first2_full",
        "group2_last_full",
        "group4",
        "group8",
        "fusion",
        "fusion_wide",
        "fusion_refine",
        "fusion_residual",
    ],
)
def test_lightweight_p5_preserves_shared_initialization_and_rng(p5_mode):
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "scale_factors": [2.0, 1.0, 0.5],
        "layer_norm": True,
    }
    torch.manual_seed(43)
    full = MultiScaleProjector(**kwargs, p5_mode="full")
    full_rng_state = torch.get_rng_state()

    torch.manual_seed(43)
    lightweight = MultiScaleProjector(**kwargs, p5_mode=p5_mode)
    lightweight_rng_state = torch.get_rng_state()

    torch.testing.assert_close(lightweight_rng_state, full_rng_state, rtol=0, atol=0)
    full_parameters = dict(full.named_parameters())
    lightweight_parameters = dict(lightweight.named_parameters())
    shared_prefixes = (
        "stages_sampling.0.",
        "stages_sampling.1.",
        "stages.0.",
        "stages.1.",
    )
    for name, parameter in full_parameters.items():
        if name.startswith(shared_prefixes):
            torch.testing.assert_close(lightweight_parameters[name], parameter, rtol=0, atol=0)


@pytest.mark.parametrize(
    "p5_mode",
    [
        "pool",
        "dwconv",
        "group2",
        "group2_mix",
        "group2_mix128",
        "group2_fullmix",
        "group2_first_full",
        "group2_first2_full",
        "group2_last_full",
        "group4",
        "group8",
        "fusion",
        "fusion_wide",
        "fusion_refine",
        "fusion_residual",
    ],
)
def test_lightweight_p5_backpropagates_to_p4_and_backbone_features(p5_mode):
    projector = MultiScaleProjector(
        [32] * 4,
        24,
        [2.0, 1.0, 0.5],
        p5_mode=p5_mode,
        layer_norm=True,
    )
    features = [torch.randn(2, 32, 8, 8, requires_grad=True) for _ in range(4)]

    outputs = projector(features)
    outputs[-1].square().mean().backward()

    assert all(feature.grad is not None for feature in features)
    assert all(torch.isfinite(feature.grad).all() and feature.grad.abs().sum() > 0 for feature in features)


def test_lightweight_p5_requires_adjacent_p4():
    with pytest.raises(ValueError, match="P4 immediately before P5"):
        MultiScaleProjector([32] * 4, 24, [2.0, 0.5], p5_mode="pool")


@pytest.mark.parametrize(
    "c2f_blocks",
    ([3, 3, 1], [1, 3, 3], [1, 3, 1], [1, 1, 1]),
)
def test_asymmetric_c2f_depth_preserves_pyramid_shapes(c2f_blocks):
    projector = MultiScaleProjector(
        [32] * 4,
        24,
        [2.0, 1.0, 0.5],
        p5_mode="group2_first_full",
        c2f_blocks_by_scale=c2f_blocks,
        layer_norm=True,
    )
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]

    assert [output.shape for output in projector(features)] == [
        (2, 24, 18, 22),
        (2, 24, 9, 11),
        (2, 24, 5, 6),
    ]


def test_asymmetric_c2f_depth_reduces_parameters_monotonically():
    kwargs = {
        "in_channels": [384] * 4,
        "out_channels": 256,
        "scale_factors": [2.0, 1.0, 0.5],
        "p5_mode": "group2_first_full",
        "source_indices_by_scale": [[0, 1, 3], [0, 1, 2, 3], [0, 2, 3]],
        "source_selection_mode": "prune",
        "layer_norm": True,
    }
    count = lambda blocks: sum(
        parameter.numel()
        for parameter in MultiScaleProjector(
            **kwargs, c2f_blocks_by_scale=blocks
        ).parameters()
    )

    baseline = count([3, 3, 3])
    p5_only = count([3, 3, 1])
    p3_p5 = count([1, 3, 1])
    all_levels = count([1, 1, 1])

    assert baseline > p5_only > p3_p5 > all_levels


def test_explicit_default_c2f_depth_preserves_initialization_and_rng():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "scale_factors": [2.0, 1.0, 0.5],
        "p5_mode": "group2_first_full",
        "layer_norm": True,
    }
    torch.manual_seed(43)
    implicit = MultiScaleProjector(**kwargs)
    implicit_rng_state = torch.get_rng_state()
    torch.manual_seed(43)
    explicit = MultiScaleProjector(**kwargs, c2f_blocks_by_scale=[3, 3, 3])
    explicit_rng_state = torch.get_rng_state()

    torch.testing.assert_close(explicit_rng_state, implicit_rng_state, rtol=0, atol=0)
    for name, parameter in implicit.named_parameters():
        torch.testing.assert_close(
            dict(explicit.named_parameters())[name], parameter, rtol=0, atol=0
        )


def test_asymmetric_c2f_depth_preserves_later_stage_initialization():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "scale_factors": [2.0, 1.0, 0.5],
        "p5_mode": "group2_first_full",
        "layer_norm": True,
    }
    torch.manual_seed(43)
    baseline = MultiScaleProjector(**kwargs)
    baseline_rng_state = torch.get_rng_state()
    torch.manual_seed(43)
    p3_light = MultiScaleProjector(**kwargs, c2f_blocks_by_scale=[1, 3, 3])
    p3_light_rng_state = torch.get_rng_state()

    torch.testing.assert_close(p3_light_rng_state, baseline_rng_state, rtol=0, atol=0)
    for prefix in ("stages_sampling.1.", "stages_sampling.2.", "stages.1.", "stages.2."):
        baseline_parameters = {
            name: parameter
            for name, parameter in baseline.named_parameters()
            if name.startswith(prefix)
        }
        p3_light_parameters = dict(p3_light.named_parameters())
        for name, parameter in baseline_parameters.items():
            torch.testing.assert_close(
                p3_light_parameters[name], parameter, rtol=0, atol=0
            )


def test_residual_fusion_starts_from_direct_fusion_output():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "scale_factors": [2.0, 1.0, 0.5],
        "layer_norm": True,
    }
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    torch.manual_seed(43)
    direct = MultiScaleProjector(**kwargs, p5_mode="fusion")
    torch.manual_seed(43)
    residual = MultiScaleProjector(**kwargs, p5_mode="fusion_residual")

    torch.testing.assert_close(residual(features)[-1], direct(features)[-1], rtol=0, atol=0)


@pytest.mark.parametrize(
    "p5_mode",
    [
        "group2",
        "group2_mix",
        "group2_mix128",
        "group2_fullmix",
        "group2_first_full",
        "group2_first2_full",
        "group2_last_full",
        "group4",
        "group8",
    ],
)
def test_grouped_p5_preserves_full_fusion_stage_initialization(p5_mode):
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "scale_factors": [2.0, 1.0, 0.5],
        "layer_norm": True,
    }
    torch.manual_seed(43)
    full = MultiScaleProjector(**kwargs, p5_mode="full")
    torch.manual_seed(43)
    grouped = MultiScaleProjector(**kwargs, p5_mode=p5_mode)

    for name, parameter in full.stages[2].named_parameters():
        torch.testing.assert_close(
            dict(grouped.stages[2].named_parameters())[name],
            parameter,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "mixed_mode",
    ["group2_mix", "group2_mix128", "group2_fullmix"],
)
def test_group2_mix_starts_from_exact_group2_output(mixed_mode):
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "scale_factors": [2.0, 1.0, 0.5],
        "layer_norm": True,
    }
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    torch.manual_seed(43)
    group2 = MultiScaleProjector(**kwargs, p5_mode="group2")
    torch.manual_seed(43)
    group2_mix = MultiScaleProjector(**kwargs, p5_mode=mixed_mode)

    for group2_output, mixed_output in zip(group2(features), group2_mix(features)):
        torch.testing.assert_close(mixed_output, group2_output, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("p5_mode", "full_index"),
    [("group2_first_full", 0), ("group2_last_full", 3)],
)
def test_group2_selective_full_preserves_selected_sampler(p5_mode, full_index):
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "scale_factors": [2.0, 1.0, 0.5],
        "layer_norm": True,
    }
    torch.manual_seed(43)
    full = MultiScaleProjector(**kwargs, p5_mode="full")
    torch.manual_seed(43)
    selective = MultiScaleProjector(**kwargs, p5_mode=p5_mode)

    full_sampler = dict(full.stages_sampling[2][full_index].named_parameters())
    selective_sampler = dict(
        selective.stages_sampling[2][full_index].named_parameters()
    )
    for name, parameter in full_sampler.items():
        torch.testing.assert_close(selective_sampler[name], parameter, rtol=0, atol=0)

    for index, sampler in enumerate(selective.stages_sampling[2]):
        expected_groups = 1 if index == full_index else 2
        assert sampler[0].conv.groups == expected_groups


def test_group2_first2_full_preserves_both_early_samplers():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "scale_factors": [2.0, 1.0, 0.5],
        "layer_norm": True,
    }
    torch.manual_seed(43)
    full = MultiScaleProjector(**kwargs, p5_mode="full")
    torch.manual_seed(43)
    selective = MultiScaleProjector(**kwargs, p5_mode="group2_first2_full")

    for index in range(4):
        expected_groups = 1 if index < 2 else 2
        assert selective.stages_sampling[2][index][0].conv.groups == expected_groups
        if index < 2:
            full_sampler = dict(full.stages_sampling[2][index].named_parameters())
            selective_sampler = dict(
                selective.stages_sampling[2][index].named_parameters()
            )
            for name, parameter in full_sampler.items():
                torch.testing.assert_close(
                    selective_sampler[name],
                    parameter,
                    rtol=0,
                    atol=0,
                )


def test_per_scale_source_masks_preserve_shapes_and_block_dropped_gradients():
    projector = MultiScaleProjector(
        [32] * 4,
        24,
        [2.0, 1.0, 0.5],
        p5_mode="group2_first_full",
        source_indices_by_scale=[[0, 1, 3], [0, 1, 2, 3], [0, 2, 3]],
    )
    features = [
        torch.randn(2, 32, 8, 8, requires_grad=True) for _ in range(4)
    ]

    outputs = projector(features)

    assert [output.shape for output in outputs] == [
        (2, 24, 16, 16),
        (2, 24, 8, 8),
        (2, 24, 4, 4),
    ]
    outputs[0].sum().backward(retain_graph=True)
    assert torch.count_nonzero(features[2].grad) == 0
    assert torch.count_nonzero(features[0].grad) > 0


def test_all_source_masks_are_bitwise_identical_to_default_projector():
    kwargs = dict(
        in_channels=[32] * 4,
        out_channels=24,
        scale_factors=[2.0, 1.0, 0.5],
        p5_mode="group2_first_full",
    )
    torch.manual_seed(17)
    default = MultiScaleProjector(**kwargs)
    torch.manual_seed(17)
    explicit = MultiScaleProjector(
        **kwargs,
        source_indices_by_scale=[[0, 1, 2, 3]] * 3,
    )
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    for default_output, explicit_output in zip(default(features), explicit(features)):
        torch.testing.assert_close(default_output, explicit_output, rtol=0, atol=0)


def test_per_scale_source_pruning_removes_branches_and_preserves_shapes():
    kwargs = dict(
        in_channels=[32] * 4,
        out_channels=24,
        scale_factors=[2.0, 1.0, 0.5],
        p5_mode="group2_first_full",
        source_indices_by_scale=[[0, 1, 3], [0, 1, 2, 3], [0, 2, 3]],
    )
    masked = MultiScaleProjector(**kwargs, source_selection_mode="mask")
    pruned = MultiScaleProjector(**kwargs, source_selection_mode="prune")
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    assert [output.shape for output in pruned(features)] == [
        (2, 24, 16, 16),
        (2, 24, 8, 8),
        (2, 24, 4, 4),
    ]
    assert pruned.stage_input_indices == [[0, 1, 3], [0, 1, 2, 3], [0, 2, 3]]
    assert [len(stage) for stage in pruned.stages_sampling] == [3, 4, 3]
    assert pruned.stages_sampling[2][0][0].conv.groups == 1
    assert pruned.stages_sampling[2][1][0].conv.groups == 2
    assert pruned.stages_sampling[2][2][0].conv.groups == 2
    assert sum(parameter.numel() for parameter in pruned.parameters()) < sum(
        parameter.numel() for parameter in masked.parameters()
    )


def _selective_pruned_projector(resample_share_mode="none"):
    return MultiScaleProjector(
        in_channels=[32] * 4,
        out_channels=24,
        scale_factors=[2.0, 1.0, 0.5],
        p5_mode="group2_first_full",
        source_indices_by_scale=[[0, 1, 3], [0, 1, 2, 3], [0, 2, 3]],
        source_selection_mode="prune",
        resample_share_mode=resample_share_mode,
    )


def test_p3_resampling_shares_only_transpose_kernel_weights():
    projector = _selective_pruned_projector("p3")
    samplers = [branch[0] for branch in projector.stages_sampling[0]]

    assert samplers[0].weight is samplers[1].weight
    assert samplers[0].weight is samplers[2].weight
    assert samplers[0].bias is not samplers[1].bias
    assert samplers[0].bias is not samplers[2].bias


def test_p5_resampling_shares_grouped_kernel_but_not_norm():
    projector = _selective_pruned_projector("p5")
    full = projector.stages_sampling[2][0][0]
    grouped_early = projector.stages_sampling[2][1][0]
    grouped_late = projector.stages_sampling[2][2][0]

    assert full.conv.weight is not grouped_early.conv.weight
    assert grouped_early.conv.weight is grouped_late.conv.weight
    assert grouped_early.bn.weight is not grouped_late.bn.weight
    assert grouped_early.bn.bias is not grouped_late.bn.bias


def test_p5_depthwise_pointwise_shares_kernels_but_not_norm():
    projector = MultiScaleProjector(
        in_channels=[32] * 4,
        out_channels=24,
        scale_factors=[2.0, 1.0, 0.5],
        p5_mode="group2_first_full_dw",
        source_indices_by_scale=[[0, 1, 3], [0, 1, 2, 3], [0, 2, 3]],
        source_selection_mode="prune",
        resample_share_mode="p3_p5",
    )
    early = projector.stages_sampling[2][1]
    late = projector.stages_sampling[2][2]

    assert early[0].groups == 32
    assert early[0].weight is late[0].weight
    assert early[1].conv.weight is late[1].conv.weight
    assert early[1].bn.weight is not late[1].bn.weight
    assert early[1].bn.bias is not late[1].bn.bias


def test_resampling_sharing_reduces_parameters_additively():
    count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    baseline = count(_selective_pruned_projector())
    p3 = count(_selective_pruned_projector("p3"))
    p5 = count(_selective_pruned_projector("p5"))
    combined = count(_selective_pruned_projector("p3_p5"))

    assert baseline - p3 == 2 * 32 * 16 * 2 * 2
    assert baseline - p5 == 32 * 16 * 3 * 3
    assert baseline - combined == (baseline - p3) + (baseline - p5)


def test_resampling_sharing_survives_state_dict_round_trip():
    torch.manual_seed(43)
    source = _selective_pruned_projector("p3_p5")
    restored = _selective_pruned_projector("p3_p5")
    restored.load_state_dict(source.state_dict())
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    assert (
        restored.stages_sampling[0][0][0].weight
        is restored.stages_sampling[0][2][0].weight
    )
    assert (
        restored.stages_sampling[2][1][0].conv.weight
        is restored.stages_sampling[2][2][0].conv.weight
    )
    for source_output, restored_output in zip(source(features), restored(features)):
        torch.testing.assert_close(restored_output, source_output, rtol=0, atol=0)
