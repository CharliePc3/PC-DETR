import pytest
import torch
import torch.nn.functional as F

from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.backbone.semantic_reassembly_projector import (
    ScaleDecoupledReassemblyProjector,
    SemanticLocalReassembly,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v2 import (
    ScaleDecoupledReassemblyProjectorV2,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v3 import (
    ScaleDecoupledReassemblyProjectorV3,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v4 import (
    ScaleDecoupledReassemblyProjectorV4,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v5 import (
    ScaleDecoupledReassemblyProjectorV5,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v6 import (
    ScaleDecoupledReassemblyProjectorV6,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v7 import (
    ScaleDecoupledReassemblyProjectorV7,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v8 import (
    ScaleDecoupledReassemblyProjectorV8,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v9 import (
    ScaleDecoupledReassemblyProjectorV9,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v10 import (
    ScaleDecoupledReassemblyProjectorV10,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v11 import (
    ScaleDecoupledReassemblyProjectorV11,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v12 import (
    ScaleDecoupledReassemblyProjectorV12,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v13 import (
    ScaleDecoupledReassemblyProjectorV13,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v14 import (
    ScaleDecoupledReassemblyProjectorV14,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v15 import (
    ScaleDecoupledReassemblyProjectorV15,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v16 import (
    ScaleDecoupledReassemblyProjectorV16,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v17 import (
    ScaleDecoupledReassemblyProjectorV17,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v18 import (
    ScaleDecoupledReassemblyProjectorV18,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v19 import (
    ScaleDecoupledReassemblyProjectorV19,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v20 import (
    ScaleDecoupledReassemblyProjectorV20,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v21 import (
    ScaleDecoupledReassemblyProjectorV21,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v22 import (
    ScaleDecoupledReassemblyProjectorV22,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v23 import (
    ScaleDecoupledReassemblyProjectorV23,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v24 import (
    ScaleDecoupledReassemblyProjectorV24,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v25 import (
    ScaleDecoupledReassemblyProjectorV25,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v26 import (
    ScaleDecoupledReassemblyProjectorV26,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v27 import (
    ScaleDecoupledReassemblyProjectorV27,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v28 import (
    ScaleDecoupledReassemblyProjectorV28,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v29 import (
    ScaleDecoupledReassemblyProjectorV29,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v30 import (
    ScaleDecoupledReassemblyProjectorV30,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v31 import (
    ScaleDecoupledReassemblyProjectorV31,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v32 import (
    ScaleDecoupledReassemblyProjectorV32,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v33 import (
    ScaleDecoupledReassemblyProjectorV33,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v34 import (
    ScaleDecoupledReassemblyProjectorV34,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v35 import (
    ScaleDecoupledReassemblyProjectorV35,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v36 import (
    ScaleDecoupledReassemblyProjectorV36,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v37 import (
    ScaleDecoupledReassemblyProjectorV37R1,
    ScaleDecoupledReassemblyProjectorV37R2,
    ScaleDecoupledReassemblyProjectorV37R3,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v38 import (
    ScaleDecoupledReassemblyProjectorV38Both,
    ScaleDecoupledReassemblyProjectorV38P3,
    ScaleDecoupledReassemblyProjectorV38P4,
)
from rfdetr.models.backbone.semantic_reassembly_projector_v39 import (
    ScaleDecoupledReassemblyProjectorV39P4Static,
)


@pytest.mark.parametrize(
    ("use_local", "use_guide", "use_phase"),
    [
        (True, True, True),
        (True, False, True),
        (False, False, True),
        (True, True, False),
    ],
)
def test_sdsr_shapes_with_odd_token_grid(use_local, use_guide, use_phase):
    model = ScaleDecoupledReassemblyProjector(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        detail_channels=8,
        use_local_reassembly=use_local,
        use_directional_guide=use_guide,
        use_phase_downsample=use_phase,
    )
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]
    image = torch.randn(2, 3, 144, 176)
    mask = torch.zeros(2, 144, 176, dtype=torch.bool)
    mask[1, -16:] = True

    outputs = model(features, image=image, mask=mask)

    assert [output.shape for output in outputs] == [
        (2, 24, 18, 22),
        (2, 24, 9, 11),
        (2, 24, 5, 6),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)


def test_local_reassembly_starts_close_to_bilinear():
    torch.manual_seed(0)
    source = torch.randn(2, 16, 9, 11)
    reassembly = SemanticLocalReassembly(16, 8, use_directional_guide=False)

    output = reassembly(source)
    reference = F.interpolate(
        source, scale_factor=2, mode="bilinear", align_corners=False
    )

    assert torch.allclose(output, reference, atol=1.0e-3, rtol=1.0e-3)


@pytest.mark.parametrize(
    ("levels", "expected_shapes"),
    [
        (["P3"], [(1, 24, 16, 20)]),
        (["P4"], [(1, 24, 8, 10)]),
        (["P5"], [(1, 24, 4, 5)]),
        (["P3", "P5"], [(1, 24, 16, 20), (1, 24, 4, 5)]),
    ],
)
def test_sdsr_only_builds_requested_levels(levels, expected_shapes):
    model = ScaleDecoupledReassemblyProjector([32] * 4, 24, levels, detail_channels=8)
    features = [torch.randn(1, 32, 8, 10) for _ in range(4)]

    outputs = model(features, image=torch.randn(1, 3, 128, 160))

    assert [output.shape for output in outputs] == expected_shapes
    assert (model.local_reassembly is not None) == ("P3" in levels)
    assert (model.p5_downsample is not None) == ("P5" in levels)


def test_sdsr_trainable_paths_receive_gradients_and_reduce_parameters():
    model = ScaleDecoupledReassemblyProjector(
        [32] * 4, 24, ["P3", "P4", "P5"], detail_channels=8
    )
    features = [torch.randn(2, 32, 8, 8, requires_grad=True) for _ in range(4)]
    outputs = model(features, image=torch.randn(2, 3, 128, 128))

    sum(
        (index + 1) * output.square().mean() for index, output in enumerate(outputs)
    ).backward()

    gradients = [
        model.router.route_logits.grad,
        model.local_reassembly.weight_predictor[-1].weight.grad,
        model.detail_stem.stem[0].weight.grad,
        model.p5_downsample.phase_logits.weight.grad,
        model.p5_downsample.gate.weight.grad,
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        for gradient in gradients
    )

    full_sdsr = ScaleDecoupledReassemblyProjector(
        [384] * 4, 256, ["P3", "P4", "P5"], detail_channels=32
    )
    full_multiscale = MultiScaleProjector(
        [384] * 4, 256, [2.0, 1.0, 0.5], layer_norm=True
    )
    sdsr_parameters = sum(parameter.numel() for parameter in full_sdsr.parameters())
    multiscale_parameters = sum(
        parameter.numel() for parameter in full_multiscale.parameters()
    )
    assert sdsr_parameters < multiscale_parameters * 0.1


@pytest.mark.parametrize(
    ("use_local", "use_guide"),
    [(True, True), (True, False), (False, False)],
)
def test_sdsr_v2_shapes_with_odd_token_grid(use_local, use_guide):
    model = ScaleDecoupledReassemblyProjectorV2(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        rank_channels=8,
        detail_channels=8,
        use_local_reassembly=use_local,
        use_directional_guide=use_guide,
    )
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]
    image = torch.randn(2, 3, 144, 176)
    mask = torch.zeros(2, 144, 176, dtype=torch.bool)
    mask[1, -16:] = True

    outputs = model(features, image=image, mask=mask)

    assert [output.shape for output in outputs] == [
        (2, 24, 18, 22),
        (2, 24, 9, 11),
        (2, 24, 5, 6),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)


def test_sdsr_v2_preserves_layers_and_initializes_scale_specific_gates():
    model = ScaleDecoupledReassemblyProjectorV2(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        rank_channels=8,
        detail_channels=8,
    )
    features = [torch.randn(2, 32, 8, 8, requires_grad=True) for _ in range(4)]
    outputs = model(features, image=torch.randn(2, 3, 128, 128))

    sum(
        (index + 1) * output.square().mean() for index, output in enumerate(outputs)
    ).backward()

    assert all(
        feature.grad is not None and feature.grad.abs().sum() > 0
        for feature in features
    )
    assert all(
        projection[-1].weight.grad is not None
        and projection[-1].weight.grad.abs().sum() > 0
        for projection in model.layer_fusion.projections
    )
    assert model.layer_fusion.scale_gate_logits.grad is not None
    assert model.layer_fusion.scale_gate_logits.grad.abs().sum() > 0
    assert model.local_reassembly.weight_predictor[-1].weight.grad.abs().sum() > 0
    assert model.detail_stem.stem[0].weight.grad.abs().sum() > 0
    assert model.p5_downsample.pointwise.weight.grad.abs().sum() > 0

    gates = model.layer_fusion.gate_values().detach().mean(dim=-1)
    assert gates[0, 0] > gates[0, -1]
    assert torch.allclose(gates[1], torch.ones_like(gates[1]))
    assert gates[2, 0] < gates[2, -1]


@pytest.mark.parametrize(
    ("levels", "expected_shapes"),
    [
        (["P3"], [(1, 24, 16, 20)]),
        (["P4"], [(1, 24, 8, 10)]),
        (["P5"], [(1, 24, 4, 5)]),
        (["P3", "P5"], [(1, 24, 16, 20), (1, 24, 4, 5)]),
    ],
)
def test_sdsr_v2_only_builds_requested_levels(levels, expected_shapes):
    model = ScaleDecoupledReassemblyProjectorV2(
        [32] * 4,
        24,
        levels,
        rank_channels=8,
        detail_channels=8,
    )
    features = [torch.randn(1, 32, 8, 10) for _ in range(4)]

    outputs = model(features, image=torch.randn(1, 3, 128, 160))

    assert [output.shape for output in outputs] == expected_shapes
    assert (model.local_reassembly is not None) == ("P3" in levels)
    assert (model.p5_downsample is not None) == ("P5" in levels)


def test_sdsr_v2_reduces_projector_parameters_without_extreme_compression():
    sdsr_v1 = ScaleDecoupledReassemblyProjector(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
        detail_channels=32,
        use_phase_downsample=False,
    )
    sdsr_v2 = ScaleDecoupledReassemblyProjectorV2(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
        rank_channels=64,
        detail_channels=32,
    )
    multiscale = MultiScaleProjector([384] * 4, 256, [2.0, 1.0, 0.5], layer_norm=True)

    v1_parameters = sum(parameter.numel() for parameter in sdsr_v1.parameters())
    v2_parameters = sum(parameter.numel() for parameter in sdsr_v2.parameters())
    multiscale_parameters = sum(
        parameter.numel() for parameter in multiscale.parameters()
    )

    assert v1_parameters < v2_parameters < multiscale_parameters * 0.25


@pytest.mark.parametrize(
    ("use_detail", "anti_alias"),
    [(False, False), (True, False), (True, True)],
)
def test_sdsr_v3_fuses_at_target_scales_with_odd_token_grid(use_detail, anti_alias):
    model = ScaleDecoupledReassemblyProjectorV3(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        rank_channels=8,
        detail_channels=8,
        use_local_reassembly=use_detail,
        use_directional_guide=use_detail,
        use_phase_downsample=anti_alias,
    )
    fusion_input_sizes = {}

    def record_input_size(level):
        def hook(_module, inputs):
            fusion_input_sizes[level] = inputs[0].shape[-2:]

        return hook

    hooks = [
        fusion.register_forward_pre_hook(record_input_size(level))
        for level, fusion in model.layer_fusion.fusions.items()
    ]
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]
    image = torch.randn(2, 3, 144, 176)
    mask = torch.zeros(2, 144, 176, dtype=torch.bool)
    mask[1, -16:] = True

    outputs = model(features, image=image, mask=mask)
    for hook in hooks:
        hook.remove()

    assert [output.shape for output in outputs] == [
        (2, 24, 18, 22),
        (2, 24, 9, 11),
        (2, 24, 5, 6),
    ]
    assert fusion_input_sizes == {"P3": (18, 22), "P4": (9, 11), "P5": (5, 6)}
    assert all(torch.isfinite(output).all() for output in outputs)


def test_sdsr_v3_trainable_paths_and_parameter_budget():
    model = ScaleDecoupledReassemblyProjectorV3(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        rank_channels=8,
        detail_channels=8,
        use_phase_downsample=True,
    )
    features = [torch.randn(2, 32, 8, 8, requires_grad=True) for _ in range(4)]
    outputs = model(features, image=torch.randn(2, 3, 128, 128))

    sum(
        (index + 1) * output.square().mean() for index, output in enumerate(outputs)
    ).backward()

    assert all(
        feature.grad is not None and feature.grad.abs().sum() > 0
        for feature in features
    )
    assert all(
        projection[-1].weight.grad is not None
        and projection[-1].weight.grad.abs().sum() > 0
        for projection in model.layer_fusion.projections
    )
    assert all(
        downsample.pointwise.weight.grad is not None
        and downsample.pointwise.weight.grad.abs().sum() > 0
        for downsample in model.layer_fusion.p5_downsamples
    )
    assert model.detail_stem.stem[0].weight.grad.abs().sum() > 0
    assert model.spatial_detail.confidence[-1].weight.grad.abs().sum() > 0

    full_v3 = ScaleDecoupledReassemblyProjectorV3(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
        rank_channels=64,
        detail_channels=32,
    )
    multiscale = MultiScaleProjector([384] * 4, 256, [2.0, 1.0, 0.5], layer_norm=True)
    v3_parameters = sum(parameter.numel() for parameter in full_v3.parameters())
    multiscale_parameters = sum(
        parameter.numel() for parameter in multiscale.parameters()
    )

    assert v3_parameters < multiscale_parameters * 0.25


@pytest.mark.parametrize(
    ("use_local", "use_guide", "anti_alias"),
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ],
)
def test_sdsr_v4_layerwise_reassembly_shapes(use_local, use_guide, anti_alias):
    model = ScaleDecoupledReassemblyProjectorV4(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        rank_channels=8,
        detail_channels=8,
        use_local_reassembly=use_local,
        use_directional_guide=use_guide,
        use_phase_downsample=anti_alias,
    )
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]
    image = torch.randn(2, 3, 144, 176)
    mask = torch.zeros(2, 144, 176, dtype=torch.bool)
    mask[1, -16:] = True

    outputs = model(features, image=image, mask=mask)

    assert [output.shape for output in outputs] == [
        (2, 24, 18, 22),
        (2, 24, 9, 11),
        (2, 24, 5, 6),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)
    assert (model.layer_fusion.p3_reassembly is not None) == use_local
    assert (model.detail_stem is not None) == (use_local and use_guide)


def test_sdsr_v4_shared_reassembly_receives_gradients_and_stays_lightweight():
    model = ScaleDecoupledReassemblyProjectorV4(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        rank_channels=8,
        detail_channels=8,
    )
    features = [torch.randn(2, 32, 8, 8, requires_grad=True) for _ in range(4)]
    outputs = model(features, image=torch.randn(2, 3, 128, 128))

    sum(
        (index + 1) * output.square().mean() for index, output in enumerate(outputs)
    ).backward()

    reassembly = model.layer_fusion.p3_reassembly
    assert all(
        feature.grad is not None and feature.grad.abs().sum() > 0
        for feature in features
    )
    assert reassembly.weight_predictor[-1].weight.grad.abs().sum() > 0
    assert model.detail_stem.stem[0].weight.grad.abs().sum() > 0

    full_v4 = ScaleDecoupledReassemblyProjectorV4(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
        rank_channels=64,
        detail_channels=32,
    )
    multiscale = MultiScaleProjector([384] * 4, 256, [2.0, 1.0, 0.5], layer_norm=True)
    v4_parameters = sum(parameter.numel() for parameter in full_v4.parameters())
    multiscale_parameters = sum(
        parameter.numel() for parameter in multiscale.parameters()
    )

    assert v4_parameters < multiscale_parameters * 0.2


@pytest.mark.parametrize(
    ("use_local", "use_guide", "anti_alias"),
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ],
)
def test_sdsr_v5_joint_reassembly_shapes(use_local, use_guide, anti_alias):
    model = ScaleDecoupledReassemblyProjectorV5(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        rank_channels=8,
        detail_channels=8,
        use_local_reassembly=use_local,
        use_directional_guide=use_guide,
        use_phase_downsample=anti_alias,
    )
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]
    image = torch.randn(2, 3, 144, 176)
    mask = torch.zeros(2, 144, 176, dtype=torch.bool)
    mask[1, -16:] = True

    outputs = model(features, image=image, mask=mask)

    assert [output.shape for output in outputs] == [
        (2, 24, 18, 22),
        (2, 24, 9, 11),
        (2, 24, 5, 6),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)
    reassembly = model.layer_fusion.p3_reassembly
    assert (reassembly is not None) == use_local
    if reassembly is not None:
        assert reassembly.context_projection[0].in_channels == 4 * 8


def test_sdsr_v5_joint_reassembly_receives_gradients_and_stays_lightweight():
    model = ScaleDecoupledReassemblyProjectorV5(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        rank_channels=8,
        detail_channels=8,
    )
    features = [torch.randn(2, 32, 8, 8, requires_grad=True) for _ in range(4)]
    outputs = model(features, image=torch.randn(2, 3, 128, 128))

    sum(
        (index + 1) * output.square().mean() for index, output in enumerate(outputs)
    ).backward()

    reassembly = model.layer_fusion.p3_reassembly
    assert all(
        feature.grad is not None and feature.grad.abs().sum() > 0
        for feature in features
    )
    assert reassembly.weight_predictor[-1].weight.grad.abs().sum() > 0
    assert model.detail_stem.stem[0].weight.grad.abs().sum() > 0

    full_v5 = ScaleDecoupledReassemblyProjectorV5(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
        rank_channels=64,
        detail_channels=32,
    )
    multiscale = MultiScaleProjector([384] * 4, 256, [2.0, 1.0, 0.5], layer_norm=True)
    v5_parameters = sum(parameter.numel() for parameter in full_v5.parameters())
    multiscale_parameters = sum(
        parameter.numel() for parameter in multiscale.parameters()
    )

    assert v5_parameters < multiscale_parameters * 0.2


@pytest.mark.parametrize(
    ("use_local", "use_guide", "anti_alias"),
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ],
)
def test_sdsr_v6_phase_detail_shapes(use_local, use_guide, anti_alias):
    model = ScaleDecoupledReassemblyProjectorV6(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        rank_channels=8,
        detail_channels=8,
        use_local_reassembly=use_local,
        use_directional_guide=use_guide,
        use_phase_downsample=anti_alias,
    )
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]
    image = torch.randn(2, 3, 144, 176)
    mask = torch.zeros(2, 144, 176, dtype=torch.bool)
    mask[1, -16:] = True

    outputs = model(features, image=image, mask=mask)

    assert [output.shape for output in outputs] == [
        (2, 24, 18, 22),
        (2, 24, 9, 11),
        (2, 24, 5, 6),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)
    assert model.layer_fusion.phase_detail.num_sources == 2


def test_sdsr_v6_phase_path_receives_gradients_and_stays_lightweight():
    model = ScaleDecoupledReassemblyProjectorV6(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        rank_channels=8,
        detail_channels=8,
        use_directional_guide=False,
    )
    features = [torch.randn(2, 32, 8, 8, requires_grad=True) for _ in range(4)]
    outputs = model(features)

    sum(
        (index + 1) * output.square().mean() for index, output in enumerate(outputs)
    ).backward()

    phase_detail = model.layer_fusion.phase_detail
    assert all(
        feature.grad is not None and feature.grad.abs().sum() > 0
        for feature in features
    )
    assert phase_detail.phase_expansion.weight.grad.abs().sum() > 0
    assert phase_detail.local_mixing.weight.grad.abs().sum() > 0
    assert phase_detail.channel_scale.grad.abs().sum() > 0

    full_v6 = ScaleDecoupledReassemblyProjectorV6(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
        rank_channels=64,
        detail_channels=32,
        use_directional_guide=False,
    )
    multiscale = MultiScaleProjector([384] * 4, 256, [2.0, 1.0, 0.5], layer_norm=True)
    v6_parameters = sum(parameter.numel() for parameter in full_v6.parameters())
    multiscale_parameters = sum(
        parameter.numel() for parameter in multiscale.parameters()
    )

    assert v6_parameters < multiscale_parameters * 0.2


@pytest.mark.parametrize(
    ("cross_scale_mode", "anti_alias"),
    [
        ("none", False),
        ("topdown", False),
        ("bidirectional", False),
        ("bidirectional", True),
    ],
)
def test_sdsr_v7_scale_first_shapes(cross_scale_mode, anti_alias):
    model = ScaleDecoupledReassemblyProjectorV7(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        rank_channels=8,
        cross_scale_mode=cross_scale_mode,
        cross_scale_rank=8,
        use_phase_downsample=anti_alias,
    )
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]
    mask = torch.zeros(2, 144, 176, dtype=torch.bool)
    mask[1, -16:] = True

    outputs = model(features, mask=mask)

    assert [output.shape for output in outputs] == [
        (2, 24, 18, 22),
        (2, 24, 9, 11),
        (2, 24, 5, 6),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)
    for output in outputs:
        output_mask = F.interpolate(
            mask[:, None].float(),
            size=output.shape[-2:],
            mode="nearest",
        ).to(torch.bool)
        assert (output.masked_select(output_mask.expand_as(output)) == 0).all()


def test_sdsr_v7_scale_first_and_cross_scale_paths_receive_gradients():
    model = ScaleDecoupledReassemblyProjectorV7(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        rank_channels=8,
        cross_scale_mode="bidirectional",
        cross_scale_rank=8,
        use_phase_downsample=True,
    )
    features = [torch.randn(2, 32, 8, 8, requires_grad=True) for _ in range(4)]
    outputs = model(features)

    sum(
        (index + 1) * output.square().mean() for index, output in enumerate(outputs)
    ).backward()

    assert all(
        feature.grad is not None and feature.grad.abs().sum() > 0
        for feature in features
    )
    assert all(
        projection.phase_projection.weight.grad.abs().sum() > 0
        for projection in model.layer_fusion.p3_projections
    )
    assert all(
        projection.depthwise.weight.grad.abs().sum() > 0
        for projection in model.layer_fusion.p5_projections
    )
    assert all(
        message.channel_scale.grad.abs().sum() > 0
        for message in [
            *model.scale_calibration.top_down,
            *model.scale_calibration.bottom_up,
        ]
    )

    full_v7 = ScaleDecoupledReassemblyProjectorV7(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
        rank_channels=64,
        cross_scale_mode="bidirectional",
        cross_scale_rank=32,
    )
    multiscale = MultiScaleProjector(
        [384] * 4,
        256,
        [2.0, 1.0, 0.5],
        layer_norm=True,
    )
    v7_parameters = sum(parameter.numel() for parameter in full_v7.parameters())
    multiscale_parameters = sum(
        parameter.numel() for parameter in multiscale.parameters()
    )

    assert v7_parameters < multiscale_parameters * 0.25


@pytest.mark.parametrize(
    ("cross_scale_mode", "anti_alias"),
    [
        ("none", False),
        ("topdown", False),
        ("bidirectional", False),
        ("bidirectional", True),
    ],
)
def test_sdsr_v8_csp_fusion_shapes(cross_scale_mode, anti_alias):
    model = ScaleDecoupledReassemblyProjectorV8(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        rank_channels=8,
        cross_scale_mode=cross_scale_mode,
        cross_scale_rank=8,
        use_phase_downsample=anti_alias,
    )
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]
    mask = torch.zeros(2, 144, 176, dtype=torch.bool)
    mask[1, -16:] = True

    outputs = model(features, mask=mask)

    assert [output.shape for output in outputs] == [
        (2, 24, 18, 22),
        (2, 24, 9, 11),
        (2, 24, 5, 6),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)
    for output in outputs:
        output_mask = F.interpolate(
            mask[:, None].float(),
            size=output.shape[-2:],
            mode="nearest",
        ).to(torch.bool)
        assert (output.masked_select(output_mask.expand_as(output)) == 0).all()


def test_sdsr_v8_csp_routing_and_cross_scale_paths_receive_gradients():
    model = ScaleDecoupledReassemblyProjectorV8(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        rank_channels=8,
        cross_scale_mode="bidirectional",
        cross_scale_rank=8,
        use_phase_downsample=True,
    )
    features = [torch.randn(2, 32, 8, 8, requires_grad=True) for _ in range(4)]
    outputs = model(features)

    sum(
        (index + 1) * output.square().mean()
        for index, output in enumerate(outputs)
    ).backward()

    assert all(
        feature.grad is not None and feature.grad.abs().sum() > 0
        for feature in features
    )
    assert model.layer_fusion.scale_gate_logits.grad.abs().sum() > 0
    assert all(
        projection[1].weight.grad.abs().sum() > 0
        for projection in model.layer_fusion.projections
    )
    assert all(
        transform.depthwise.weight.grad.abs().sum() > 0
        for transform in model.layer_fusion.p3_transforms
    )
    assert all(
        downsample.depthwise.weight.grad.abs().sum() > 0
        for downsample in model.layer_fusion.p5_downsamples
    )
    assert all(
        block.depthwise.weight.grad.abs().sum() > 0
        for fusion in model.layer_fusion.fusions.values()
        for block in fusion.blocks
    )
    assert all(
        message.channel_scale.grad.abs().sum() > 0
        for message in [
            *model.scale_calibration.top_down,
            *model.scale_calibration.bottom_up,
        ]
    )

    full_v8 = ScaleDecoupledReassemblyProjectorV8(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
        rank_channels=64,
        cross_scale_mode="bidirectional",
        cross_scale_rank=32,
    )
    multiscale = MultiScaleProjector(
        [384] * 4,
        256,
        [2.0, 1.0, 0.5],
        layer_norm=True,
    )
    v8_parameters = sum(parameter.numel() for parameter in full_v8.parameters())
    multiscale_parameters = sum(
        parameter.numel() for parameter in multiscale.parameters()
    )

    assert v8_parameters < multiscale_parameters * 0.3


@pytest.mark.parametrize(
    ("cross_scale_mode", "anti_alias"),
    [
        ("none", True),
        ("topdown", True),
        ("bidirectional", True),
        ("none", False),
    ],
)
def test_sdsr_v9_anchor_reassembly_shapes(cross_scale_mode, anti_alias):
    model = ScaleDecoupledReassemblyProjectorV9(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        rank_channels=8,
        cross_scale_mode=cross_scale_mode,
        cross_scale_rank=8,
        use_phase_downsample=anti_alias,
    )
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]
    mask = torch.zeros(2, 144, 176, dtype=torch.bool)
    mask[1, -16:] = True

    outputs = model(features, mask=mask)

    assert [output.shape for output in outputs] == [
        (2, 24, 18, 22),
        (2, 24, 9, 11),
        (2, 24, 5, 6),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)
    for output in outputs:
        output_mask = F.interpolate(
            mask[:, None].float(),
            size=output.shape[-2:],
            mode="nearest",
        ).to(torch.bool)
        assert (output.masked_select(output_mask.expand_as(output)) == 0).all()


def test_sdsr_v9_anchor_and_scale_residuals_receive_gradients():
    model = ScaleDecoupledReassemblyProjectorV9(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        rank_channels=8,
        cross_scale_mode="bidirectional",
        cross_scale_rank=8,
        use_phase_downsample=True,
    )
    features = [torch.randn(2, 32, 8, 8, requires_grad=True) for _ in range(4)]
    outputs = model(features)

    sum(
        (index + 1) * output.square().mean()
        for index, output in enumerate(outputs)
    ).backward()

    assert all(
        feature.grad is not None and feature.grad.abs().sum() > 0
        for feature in features
    )
    assert model.anchor.fusion.cv1.conv.weight.grad.abs().sum() > 0
    assert model.p3_reassembly.layer_gate_logits.grad.abs().sum() > 0
    assert model.p3_reassembly.phase_projection.weight.grad.abs().sum() > 0
    assert model.p3_reassembly.residual_scale.grad.abs().sum() > 0
    assert model.p5_reassembly.layer_gate_logits.grad.abs().sum() > 0
    assert model.p5_reassembly.downsample.weight.grad.abs().sum() > 0
    assert model.p5_reassembly.residual_scale.grad.abs().sum() > 0
    assert all(
        message.channel_scale.grad.abs().sum() > 0
        for message in [
            *model.scale_calibration.top_down,
            *model.scale_calibration.bottom_up,
        ]
    )

    full_v9 = ScaleDecoupledReassemblyProjectorV9(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
        rank_channels=64,
        cross_scale_mode="bidirectional",
        cross_scale_rank=32,
    )
    multiscale = MultiScaleProjector(
        [384] * 4,
        256,
        [2.0, 1.0, 0.5],
        layer_norm=True,
    )
    v9_parameters = sum(parameter.numel() for parameter in full_v9.parameters())
    multiscale_parameters = sum(
        parameter.numel() for parameter in multiscale.parameters()
    )

    assert v9_parameters < multiscale_parameters * 0.3


@pytest.mark.parametrize(
    ("cross_scale_mode", "anti_alias"),
    [
        ("none", True),
        ("topdown", True),
        ("bidirectional", True),
        ("none", False),
    ],
)
def test_sdsr_v10_alias_aware_fusion_shapes(cross_scale_mode, anti_alias):
    model = ScaleDecoupledReassemblyProjectorV10(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        cross_scale_mode=cross_scale_mode,
        cross_scale_rank=8,
        use_phase_downsample=anti_alias,
    )
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]
    mask = torch.zeros(2, 144, 176, dtype=torch.bool)
    mask[1, -16:] = True

    outputs = model(features, mask=mask)

    assert [output.shape for output in outputs] == [
        (2, 24, 18, 22),
        (2, 24, 9, 11),
        (2, 24, 5, 6),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)
    for output in outputs:
        output_mask = F.interpolate(
            mask[:, None].float(),
            size=output.shape[-2:],
            mode="nearest",
        ).to(torch.bool)
        assert (output.masked_select(output_mask.expand_as(output)) == 0).all()


def test_sdsr_v10_scale_branches_receive_gradients_and_reduce_parameters():
    model = ScaleDecoupledReassemblyProjectorV10(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        cross_scale_mode="bidirectional",
        cross_scale_rank=8,
        use_phase_downsample=True,
    )
    features = [torch.randn(2, 32, 8, 8, requires_grad=True) for _ in range(4)]
    outputs = model(features)

    sum(
        (index + 1) * output.square().mean()
        for index, output in enumerate(outputs)
    ).backward()

    assert all(
        feature.grad is not None and feature.grad.abs().sum() > 0
        for feature in features
    )
    assert model.branches["P3"].fusion.cv1.conv.weight.grad.abs().sum() > 0
    assert model.branches["P4"].fusion.cv1.conv.weight.grad.abs().sum() > 0
    assert model.branches["P5"].fusion.cv1.conv.weight.grad.abs().sum() > 0
    assert all(
        downsample.detail.weight.grad.abs().sum() > 0
        and downsample.detail_scale.grad.abs().sum() > 0
        for downsample in model.branches["P5"].downsampling
    )
    assert all(
        message.channel_scale.grad.abs().sum() > 0
        for message in [
            *model.scale_calibration.top_down,
            *model.scale_calibration.bottom_up,
        ]
    )

    full_v10 = ScaleDecoupledReassemblyProjectorV10(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
    )
    multiscale = MultiScaleProjector(
        [384] * 4,
        256,
        [2.0, 1.0, 0.5],
        layer_norm=True,
    )
    v10_parameters = sum(
        parameter.numel() for parameter in full_v10.parameters()
    )
    multiscale_parameters = sum(
        parameter.numel() for parameter in multiscale.parameters()
    )

    assert v10_parameters < multiscale_parameters * 0.55


def test_sdsr_v10_preserves_multiscale_p3_p4_initialization_and_outputs():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
    }
    torch.manual_seed(1042)
    multiscale = MultiScaleProjector(
        **kwargs,
        scale_factors=[2.0, 1.0, 0.5],
        layer_norm=True,
    )
    torch.manual_seed(1042)
    v10 = ScaleDecoupledReassemblyProjectorV10(
        **kwargs,
        levels=["P3", "P4", "P5"],
    )
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    multiscale_outputs = multiscale(features)
    v10_outputs = v10(features)

    torch.testing.assert_close(v10_outputs[0], multiscale_outputs[0])
    torch.testing.assert_close(v10_outputs[1], multiscale_outputs[1])
    for key, value in v10.branches["P5"].fusion.state_dict().items():
        torch.testing.assert_close(
            value,
            multiscale.stages[2][0].state_dict()[key],
        )


@pytest.mark.parametrize(
    ("cross_scale_mode", "anti_alias"),
    [
        ("none", True),
        ("topdown", True),
        ("bidirectional", True),
        ("none", False),
    ],
)
def test_sdsr_v11_two_basis_fusion_shapes(cross_scale_mode, anti_alias):
    model = ScaleDecoupledReassemblyProjectorV11(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        cross_scale_mode=cross_scale_mode,
        cross_scale_rank=8,
        use_phase_downsample=anti_alias,
    )
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]
    mask = torch.zeros(2, 144, 176, dtype=torch.bool)
    mask[1, -16:] = True

    outputs = model(features, mask=mask)

    assert [output.shape for output in outputs] == [
        (2, 24, 18, 22),
        (2, 24, 9, 11),
        (2, 24, 5, 6),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)
    for output in outputs:
        output_mask = F.interpolate(
            mask[:, None].float(),
            size=output.shape[-2:],
            mode="nearest",
        ).to(torch.bool)
        assert (output.masked_select(output_mask.expand_as(output)) == 0).all()


def test_sdsr_v11_phase_basis_gradients_and_parameter_budget():
    model = ScaleDecoupledReassemblyProjectorV11(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        cross_scale_mode="bidirectional",
        cross_scale_rank=8,
        use_phase_downsample=True,
    )
    features = [torch.randn(2, 32, 8, 8, requires_grad=True) for _ in range(4)]
    outputs = model(features)

    sum(
        (index + 1) * output.square().mean()
        for index, output in enumerate(outputs)
    ).backward()

    assert all(
        feature.grad is not None and feature.grad.abs().sum() > 0
        for feature in features
    )
    assert model.branches["P3"].fusion.cv1.conv.weight.grad.abs().sum() > 0
    assert model.branches["P4"].fusion.cv1.conv.weight.grad.abs().sum() > 0
    assert model.branches["P5"].fusion.cv1.conv.weight.grad.abs().sum() > 0
    assert all(
        downsample.detail.weight.grad.abs().sum() > 0
        and downsample.detail_scale.grad.abs().sum() > 0
        for downsample in model.branches["P5"].downsampling
    )

    full_v11 = ScaleDecoupledReassemblyProjectorV11(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
    )
    multiscale = MultiScaleProjector(
        [384] * 4,
        256,
        [2.0, 1.0, 0.5],
        layer_norm=True,
    )
    v11_parameters = sum(
        parameter.numel() for parameter in full_v11.parameters()
    )
    multiscale_parameters = sum(
        parameter.numel() for parameter in multiscale.parameters()
    )

    assert v11_parameters < multiscale_parameters * 0.6


def test_sdsr_v11_preserves_multiscale_p3_p4_outputs():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
    }
    torch.manual_seed(1042)
    multiscale = MultiScaleProjector(
        **kwargs,
        scale_factors=[2.0, 1.0, 0.5],
        layer_norm=True,
    )
    torch.manual_seed(1042)
    v11 = ScaleDecoupledReassemblyProjectorV11(
        **kwargs,
        levels=["P3", "P4", "P5"],
    )
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    multiscale_outputs = multiscale(features)
    v11_outputs = v11(features)

    torch.testing.assert_close(v11_outputs[0], multiscale_outputs[0])
    torch.testing.assert_close(v11_outputs[1], multiscale_outputs[1])


@pytest.mark.parametrize(
    ("cross_scale_mode", "anti_alias"),
    [
        ("none", True),
        ("topdown", True),
        ("bidirectional", True),
        ("none", False),
    ],
)
def test_sdsr_v12_four_basis_fusion_shapes(cross_scale_mode, anti_alias):
    model = ScaleDecoupledReassemblyProjectorV12(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        cross_scale_mode=cross_scale_mode,
        cross_scale_rank=8,
        use_phase_downsample=anti_alias,
    )
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]
    mask = torch.zeros(2, 144, 176, dtype=torch.bool)
    mask[1, -16:] = True

    outputs = model(features, mask=mask)

    assert [output.shape for output in outputs] == [
        (2, 24, 18, 22),
        (2, 24, 9, 11),
        (2, 24, 5, 6),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)
    for output in outputs:
        output_mask = F.interpolate(
            mask[:, None].float(),
            size=output.shape[-2:],
            mode="nearest",
        ).to(torch.bool)
        assert (output.masked_select(output_mask.expand_as(output)) == 0).all()


def test_sdsr_v12_four_basis_gradients_and_parameter_budget():
    model = ScaleDecoupledReassemblyProjectorV12(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        cross_scale_mode="bidirectional",
        cross_scale_rank=8,
        use_phase_downsample=True,
    )
    features = [torch.randn(2, 32, 8, 8, requires_grad=True) for _ in range(4)]
    outputs = model(features)

    sum(
        (index + 1) * output.square().mean()
        for index, output in enumerate(outputs)
    ).backward()

    assert all(
        feature.grad is not None and feature.grad.abs().sum() > 0
        for feature in features
    )
    assert model.branches["P3"].fusion.cv1.conv.weight.grad.abs().sum() > 0
    assert model.branches["P4"].fusion.cv1.conv.weight.grad.abs().sum() > 0
    assert model.branches["P5"].fusion.cv1.conv.weight.grad.abs().sum() > 0
    assert all(
        downsample.detail.weight.grad.abs().sum() > 0
        and downsample.detail_scale.grad.abs().sum() > 0
        for downsample in model.branches["P5"].downsampling
    )

    full_v12 = ScaleDecoupledReassemblyProjectorV12(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
    )
    multiscale = MultiScaleProjector(
        [384] * 4,
        256,
        [2.0, 1.0, 0.5],
        layer_norm=True,
    )
    v12_parameters = sum(
        parameter.numel() for parameter in full_v12.parameters()
    )
    multiscale_parameters = sum(
        parameter.numel() for parameter in multiscale.parameters()
    )

    assert v12_parameters < multiscale_parameters * 0.65


def test_sdsr_v12_preserves_multiscale_p3_p4_outputs():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
    }
    torch.manual_seed(1042)
    multiscale = MultiScaleProjector(
        **kwargs,
        scale_factors=[2.0, 1.0, 0.5],
        layer_norm=True,
    )
    torch.manual_seed(1042)
    v12 = ScaleDecoupledReassemblyProjectorV12(
        **kwargs,
        levels=["P3", "P4", "P5"],
    )
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    multiscale_outputs = multiscale(features)
    v12_outputs = v12(features)

    torch.testing.assert_close(v12_outputs[0], multiscale_outputs[0])
    torch.testing.assert_close(v12_outputs[1], multiscale_outputs[1])


def test_sdsr_v13_highpass_initialization_and_parameter_parity():
    v11 = ScaleDecoupledReassemblyProjectorV11(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
    )
    v13 = ScaleDecoupledReassemblyProjectorV13(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
    )

    assert sum(p.numel() for p in v13.parameters()) == sum(
        p.numel() for p in v11.parameters()
    )
    expected = torch.tensor(
        [[-1.0, -2.0, -1.0], [-2.0, 12.0, -2.0], [-1.0, -2.0, -1.0]]
    ) / 16.0
    for downsample in v13.branches["P5"].downsampling:
        torch.testing.assert_close(
            downsample.detail.weight[:, 0],
            expected.expand_as(downsample.detail.weight[:, 0]),
        )
        torch.testing.assert_close(
            downsample.detail_scale,
            torch.full_like(downsample.detail_scale, 0.5),
        )


def test_sdsr_v13_shapes_masks_and_gradients():
    model = ScaleDecoupledReassemblyProjectorV13(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        cross_scale_mode="bidirectional",
        cross_scale_rank=8,
    )
    features = [torch.randn(2, 32, 9, 11, requires_grad=True) for _ in range(4)]
    mask = torch.zeros(2, 144, 176, dtype=torch.bool)
    mask[1, -16:] = True

    outputs = model(features, mask=mask)
    assert [output.shape for output in outputs] == [
        (2, 24, 18, 22),
        (2, 24, 9, 11),
        (2, 24, 5, 6),
    ]
    sum(output.square().mean() for output in outputs).backward()
    assert all(
        feature.grad is not None and feature.grad.abs().sum() > 0
        for feature in features
    )
    assert all(
        downsample.detail.weight.grad.abs().sum() > 0
        and downsample.detail_scale.grad.abs().sum() > 0
        for downsample in model.branches["P5"].downsampling
    )
    for output in outputs:
        output_mask = F.interpolate(
            mask[:, None].float(),
            size=output.shape[-2:],
            mode="nearest",
        ).to(torch.bool)
        assert (output.masked_select(output_mask.expand_as(output)) == 0).all()


def test_sdsr_v13_preserves_v11_p3_p4_initialization_and_outputs():
    kwargs = {"in_channels": [32] * 4, "out_channels": 24}
    torch.manual_seed(1042)
    v11 = ScaleDecoupledReassemblyProjectorV11(
        **kwargs,
        levels=["P3", "P4", "P5"],
    )
    torch.manual_seed(1042)
    v13 = ScaleDecoupledReassemblyProjectorV13(
        **kwargs,
        levels=["P3", "P4", "P5"],
    )
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    v11_outputs = v11(features)
    v13_outputs = v13(features)
    torch.testing.assert_close(v13_outputs[0], v11_outputs[0])
    torch.testing.assert_close(v13_outputs[1], v11_outputs[1])


def test_sdsr_v14_layer_adaptive_shapes_gradients_and_budget():
    model = ScaleDecoupledReassemblyProjectorV14(
        [32] * 4,
        24,
        ["P3", "P4", "P5"],
        cross_scale_mode="bidirectional",
        cross_scale_rank=8,
    )
    features = [torch.randn(2, 32, 9, 11, requires_grad=True) for _ in range(4)]
    mask = torch.zeros(2, 144, 176, dtype=torch.bool)
    mask[1, -16:] = True

    outputs = model(features, mask=mask)
    assert [output.shape for output in outputs] == [
        (2, 24, 18, 22),
        (2, 24, 9, 11),
        (2, 24, 5, 6),
    ]
    sum(output.square().mean() for output in outputs).backward()
    assert model.branches["P5"].first_sampler.conv.groups == 1
    assert model.branches["P5"].first_sampler.conv.weight.grad.abs().sum() > 0
    assert all(
        sampler.detail.groups == 32
        and sampler.detail.weight.grad.abs().sum() > 0
        for sampler in model.branches["P5"].later_samplers
    )
    assert all(
        feature.grad is not None and feature.grad.abs().sum() > 0
        for feature in features
    )
    for output in outputs:
        output_mask = F.interpolate(
            mask[:, None].float(),
            size=output.shape[-2:],
            mode="nearest",
        ).to(torch.bool)
        assert (output.masked_select(output_mask.expand_as(output)) == 0).all()

    full_v14 = ScaleDecoupledReassemblyProjectorV14(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
    )
    multiscale = MultiScaleProjector(
        [384] * 4,
        256,
        [2.0, 1.0, 0.5],
        layer_norm=True,
    )
    assert sum(p.numel() for p in full_v14.parameters()) < sum(
        p.numel() for p in multiscale.parameters()
    ) * 0.7


def test_sdsr_v14_preserves_multiscale_p3_p4_outputs():
    kwargs = {"in_channels": [32] * 4, "out_channels": 24}
    torch.manual_seed(1042)
    multiscale = MultiScaleProjector(
        **kwargs,
        scale_factors=[2.0, 1.0, 0.5],
        layer_norm=True,
    )
    torch.manual_seed(1042)
    v14 = ScaleDecoupledReassemblyProjectorV14(
        **kwargs,
        levels=["P3", "P4", "P5"],
    )
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    multiscale_outputs = multiscale(features)
    v14_outputs = v14(features)
    torch.testing.assert_close(v14_outputs[0], multiscale_outputs[0])
    torch.testing.assert_close(v14_outputs[1], multiscale_outputs[1])


def test_sdsr_v15_zero_init_matches_v11_and_opens_bottom_up_path():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
    }
    torch.manual_seed(1042)
    v11 = ScaleDecoupledReassemblyProjectorV11(**kwargs)
    torch.manual_seed(1042)
    v15 = ScaleDecoupledReassemblyProjectorV15(
        **kwargs,
        cross_scale_rank=8,
    )
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    v11_outputs = v11(features)
    v15_outputs = v15(features)
    for expected, actual in zip(v11_outputs, v15_outputs):
        torch.testing.assert_close(actual, expected)
    assert all(
        torch.count_nonzero(message.channel_scale) == 0
        for message in v15.scale_calibration.bottom_up
    )

    sum(output.square().mean() for output in v15_outputs).backward()
    assert all(
        message.channel_scale.grad is not None
        and message.channel_scale.grad.abs().sum() > 0
        for message in v15.scale_calibration.bottom_up
    )

    v15.zero_grad(set_to_none=True)
    for message in v15.scale_calibration.bottom_up:
        message.channel_scale.data.fill_(0.1)
    opened_outputs = v15([feature.detach() for feature in features])
    sum(output.square().mean() for output in opened_outputs).backward()
    assert all(
        message.reduce.weight.grad is not None
        and message.reduce.weight.grad.abs().sum() > 0
        and message.expand.weight.grad is not None
        and message.expand.weight.grad.abs().sum() > 0
        for message in v15.scale_calibration.bottom_up
    )

    full_v11 = ScaleDecoupledReassemblyProjectorV11(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
    )
    full_v15 = ScaleDecoupledReassemblyProjectorV15(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
        cross_scale_rank=32,
    )
    assert sum(p.numel() for p in full_v15.parameters()) < sum(
        p.numel() for p in full_v11.parameters()
    ) * 1.01


def test_sdsr_v16_zero_init_matches_v11_and_opens_p4_p5_path():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
    }
    torch.manual_seed(1042)
    v11 = ScaleDecoupledReassemblyProjectorV11(**kwargs)
    torch.manual_seed(1042)
    v16 = ScaleDecoupledReassemblyProjectorV16(**kwargs)
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    v11_outputs = v11(features)
    v16_outputs = v16(features)
    for expected, actual in zip(v11_outputs, v16_outputs):
        torch.testing.assert_close(actual, expected)

    gate = v16.scale_calibration.channel_scale
    sum(output.square().mean() for output in v16_outputs).backward()
    assert gate.grad is not None
    assert gate.grad.abs().sum() > 0

    with torch.no_grad():
        gate.fill_(0.25)
    opened_outputs = v16([feature.detach() for feature in features])
    torch.testing.assert_close(opened_outputs[0], v11_outputs[0])
    torch.testing.assert_close(opened_outputs[1], v11_outputs[1])
    assert not torch.allclose(opened_outputs[2], v11_outputs[2])

    full_v11 = ScaleDecoupledReassemblyProjectorV11(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
    )
    full_v16 = ScaleDecoupledReassemblyProjectorV16(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
    )
    assert sum(p.numel() for p in full_v16.parameters()) == (
        sum(p.numel() for p in full_v11.parameters()) + 256
    )


def test_sdsr_v17_preserves_v11_output_and_doubles_initial_gate_gradient():
    kwargs = {
        "in_channels": [16] * 4,
        "out_channels": 12,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 1,
    }
    torch.manual_seed(1042)
    v16 = ScaleDecoupledReassemblyProjectorV16(**kwargs)
    torch.manual_seed(1042)
    v17 = ScaleDecoupledReassemblyProjectorV17(**kwargs)
    features = [torch.randn(2, 16, 8, 8) for _ in range(4)]

    outputs_v16 = v16(features)
    outputs_v17 = v17([feature.detach().clone() for feature in features])
    for expected, actual in zip(outputs_v16, outputs_v17):
        torch.testing.assert_close(actual, expected)

    outputs_v16[2].square().mean().backward()
    outputs_v17[2].square().mean().backward()
    grad_v16 = v16.scale_calibration.channel_scale.grad
    grad_v17 = v17.scale_calibration.channel_scale.grad
    torch.testing.assert_close(grad_v17, 2.0 * grad_v16)


def test_sdsr_v18_matches_v11_then_learns_lowpass_bases():
    kwargs = {
        "in_channels": [16] * 4,
        "out_channels": 12,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 1,
    }
    torch.manual_seed(1042)
    v11 = ScaleDecoupledReassemblyProjectorV11(**kwargs)
    torch.manual_seed(1042)
    v18 = ScaleDecoupledReassemblyProjectorV18(**kwargs)
    features = [torch.randn(2, 16, 8, 8) for _ in range(4)]

    outputs_v11 = v11(features)
    outputs_v18 = v18([feature.detach().clone() for feature in features])
    for expected, actual in zip(outputs_v11, outputs_v18):
        torch.testing.assert_close(actual, expected)

    outputs_v18[2].square().mean().backward()
    for downsample in v18.branches["P5"].downsampling:
        assert downsample.primary.weight.grad is not None
        assert downsample.primary.weight.grad.abs().sum() > 0

    full_v11 = ScaleDecoupledReassemblyProjectorV11(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
    )
    full_v18 = ScaleDecoupledReassemblyProjectorV18(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
    )
    assert sum(p.numel() for p in full_v18.parameters()) == (
        sum(p.numel() for p in full_v11.parameters()) + 4 * 384 * 3 * 3
    )


def test_sdsr_v19_preserves_exact_p3_p4_and_deepens_only_p5_fusion():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v11 = ScaleDecoupledReassemblyProjectorV11(**kwargs)
    torch.manual_seed(1042)
    v19 = ScaleDecoupledReassemblyProjectorV19(**kwargs)
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    outputs_v11 = v11(features)
    outputs_v19 = v19([feature.detach().clone() for feature in features])
    torch.testing.assert_close(outputs_v19[0], outputs_v11[0])
    torch.testing.assert_close(outputs_v19[1], outputs_v11[1])
    assert outputs_v19[2].shape == outputs_v11[2].shape

    outputs_v19[2].square().mean().backward()
    extra_block = v19.branches["P5"].fusion.m[-1]
    assert extra_block.cv1.conv.weight.grad is not None
    assert extra_block.cv1.conv.weight.grad.abs().sum() > 0

    full_multiscale = MultiScaleProjector(
        [384] * 4,
        256,
        [2.0, 1.0, 0.5],
        layer_norm=True,
    )
    full_v11 = ScaleDecoupledReassemblyProjectorV11(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
    )
    full_v19 = ScaleDecoupledReassemblyProjectorV19(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
    )
    v11_params = sum(p.numel() for p in full_v11.parameters())
    v19_params = sum(p.numel() for p in full_v19.parameters())
    multiscale_params = sum(p.numel() for p in full_multiscale.parameters())
    assert v11_params < v19_params < multiscale_params * 0.6


def test_sdsr_v20_preserves_v11_core_and_adds_trainable_layer_adapters():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "rank_channels": 8,
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v11 = ScaleDecoupledReassemblyProjectorV11(**kwargs)
    torch.manual_seed(1042)
    v20 = ScaleDecoupledReassemblyProjectorV20(**kwargs)

    for level in ("P3", "P4"):
        for expected, actual in zip(
            v11.branches[level].parameters(),
            v20.branches[level].parameters(),
        ):
            torch.testing.assert_close(actual, expected)
    for expected, actual in zip(
        v11.branches["P5"].fusion.parameters(),
        v20.branches["P5"].fusion.parameters(),
    ):
        torch.testing.assert_close(actual, expected)

    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]
    outputs = v20(features)
    assert [output.shape for output in outputs] == [
        (2, 24, 16, 16),
        (2, 24, 8, 8),
        (2, 24, 4, 4),
    ]
    outputs[2].square().mean().backward()
    for adapter in v20.branches["P5"].adapters:
        assert adapter.reduce.weight.grad is not None
        assert adapter.reduce.weight.grad.abs().sum() > 0
        assert adapter.expand.weight.grad is not None
        assert adapter.expand.weight.grad.abs().sum() > 0

    full_multiscale = MultiScaleProjector(
        [384] * 4,
        256,
        [2.0, 1.0, 0.5],
        layer_norm=True,
    )
    full_v20 = ScaleDecoupledReassemblyProjectorV20(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
        rank_channels=64,
    )
    assert sum(p.numel() for p in full_v20.parameters()) < sum(
        p.numel() for p in full_multiscale.parameters()
    ) * 0.6


def test_sdsr_v21_preserves_v11_core_and_adds_spatial_semantic_residuals():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "rank_channels": 8,
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v11 = ScaleDecoupledReassemblyProjectorV11(**kwargs)
    torch.manual_seed(1042)
    v21 = ScaleDecoupledReassemblyProjectorV21(**kwargs)

    for level in ("P3", "P4"):
        for expected, actual in zip(
            v11.branches[level].parameters(),
            v21.branches[level].parameters(),
        ):
            torch.testing.assert_close(actual, expected)
    for expected, actual in zip(
        v11.branches["P5"].fusion.parameters(),
        v21.branches["P5"].fusion.parameters(),
    ):
        torch.testing.assert_close(actual, expected)

    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]
    outputs = v21(features)
    assert [output.shape for output in outputs] == [
        (2, 24, 16, 16),
        (2, 24, 8, 8),
        (2, 24, 4, 4),
    ]
    outputs[2].square().mean().backward()
    for downsample in v21.branches["P5"].downsampling:
        assert downsample.residual.reduce.weight.grad is not None
        assert downsample.residual.reduce.weight.grad.abs().sum() > 0
        assert downsample.residual.spatial.weight.grad is not None
        assert downsample.residual.spatial.weight.grad.abs().sum() > 0
        assert downsample.residual.expand.weight.grad is not None
        assert downsample.residual.expand.weight.grad.abs().sum() > 0

    full_multiscale = MultiScaleProjector(
        [384] * 4,
        256,
        [2.0, 1.0, 0.5],
        layer_norm=True,
    )
    full_v21 = ScaleDecoupledReassemblyProjectorV21(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
        rank_channels=64,
    )
    assert sum(p.numel() for p in full_v21.parameters()) < sum(
        p.numel() for p in full_multiscale.parameters()
    ) * 0.6


def test_sdsr_v22_matches_v11_at_init_and_learns_grouped_detail():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v11 = ScaleDecoupledReassemblyProjectorV11(**kwargs)
    torch.manual_seed(1042)
    v22 = ScaleDecoupledReassemblyProjectorV22(**kwargs)
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    expected = v11(features)
    actual = v22([feature.detach().clone() for feature in features])
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(
            actual_level, expected_level, rtol=2.0e-5, atol=1.0e-6
        )

    actual[2].square().mean().backward()
    for downsample in v22.branches["P5"].downsampling:
        assert downsample.detail.groups == 16
        gradient = downsample.detail.weight.grad
        assert gradient is not None
        channels_per_group = gradient.shape[1]
        off_diagonal = gradient.detach().clone()
        for output_channel in range(gradient.shape[0]):
            off_diagonal[
                output_channel, output_channel % channels_per_group
            ].zero_()
        assert off_diagonal.abs().sum() > 0

    full_multiscale = MultiScaleProjector(
        [384] * 4,
        256,
        [2.0, 1.0, 0.5],
        layer_norm=True,
    )
    full_v22 = ScaleDecoupledReassemblyProjectorV22(
        [384] * 4,
        256,
        ["P3", "P4", "P5"],
    )
    assert sum(p.numel() for p in full_v22.parameters()) < sum(
        p.numel() for p in full_multiscale.parameters()
    ) * 0.6


def test_sdsr_v23_matches_v11_and_only_expands_deepest_detail():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v11 = ScaleDecoupledReassemblyProjectorV11(**kwargs)
    torch.manual_seed(1042)
    v23 = ScaleDecoupledReassemblyProjectorV23(**kwargs)
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    expected = v11(features)
    actual = v23([feature.detach().clone() for feature in features])
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(
            actual_level, expected_level, rtol=2.0e-5, atol=1.0e-6
        )

    samplers = v23.branches["P5"].downsampling
    assert [sampler.detail.groups for sampler in samplers] == [32, 32, 32, 16]
    actual[2].square().mean().backward()
    gradient = samplers[-1].detail.weight.grad
    assert gradient is not None
    channels_per_group = gradient.shape[1]
    off_diagonal = gradient.detach().clone()
    for output_channel in range(gradient.shape[0]):
        off_diagonal[
            output_channel, output_channel % channels_per_group
        ].zero_()
    assert off_diagonal.abs().sum() > 0

    full_multiscale = MultiScaleProjector(
        [384] * 4, 256, [2.0, 1.0, 0.5], layer_norm=True
    )
    full_v23 = ScaleDecoupledReassemblyProjectorV23(
        [384] * 4, 256, ["P3", "P4", "P5"]
    )
    assert sum(p.numel() for p in full_v23.parameters()) < sum(
        p.numel() for p in full_multiscale.parameters()
    ) * 0.55


def test_sdsr_v24_matches_v11_and_only_expands_earliest_detail():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v11 = ScaleDecoupledReassemblyProjectorV11(**kwargs)
    torch.manual_seed(1042)
    v24 = ScaleDecoupledReassemblyProjectorV24(**kwargs)
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    expected = v11(features)
    actual = v24([feature.detach().clone() for feature in features])
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(
            actual_level, expected_level, rtol=2.0e-5, atol=1.0e-6
        )

    samplers = v24.branches["P5"].downsampling
    assert [sampler.detail.groups for sampler in samplers] == [4, 32, 32, 32]
    full_multiscale = MultiScaleProjector(
        [384] * 4, 256, [2.0, 1.0, 0.5], layer_norm=True
    )
    full_v24 = ScaleDecoupledReassemblyProjectorV24(
        [384] * 4, 256, ["P3", "P4", "P5"]
    )
    assert sum(p.numel() for p in full_v24.parameters()) < sum(
        p.numel() for p in full_multiscale.parameters()
    ) * 0.6


def test_sdsr_v25_matches_v11_with_one_dense_early_detail_sampler():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v11 = ScaleDecoupledReassemblyProjectorV11(**kwargs)
    torch.manual_seed(1042)
    v25 = ScaleDecoupledReassemblyProjectorV25(**kwargs)
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    expected = v11(features)
    actual = v25([feature.detach().clone() for feature in features])
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(
            actual_level, expected_level, rtol=2.0e-5, atol=1.0e-6
        )
    assert [
        sampler.detail.groups
        for sampler in v25.branches["P5"].downsampling
    ] == [1, 32, 32, 32]

    full_multiscale = MultiScaleProjector(
        [384] * 4, 256, [2.0, 1.0, 0.5], layer_norm=True
    )
    full_v25 = ScaleDecoupledReassemblyProjectorV25(
        [384] * 4, 256, ["P3", "P4", "P5"]
    )
    assert sum(p.numel() for p in full_v25.parameters()) < sum(
        p.numel() for p in full_multiscale.parameters()
    ) * 0.7


def test_sdsr_v26_matches_v11_and_only_expands_endpoint_details():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v11 = ScaleDecoupledReassemblyProjectorV11(**kwargs)
    torch.manual_seed(1042)
    v26 = ScaleDecoupledReassemblyProjectorV26(**kwargs)
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    expected = v11(features)
    actual = v26([feature.detach().clone() for feature in features])
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(
            actual_level, expected_level, rtol=2.0e-5, atol=1.0e-6
        )
    assert [
        sampler.detail.groups
        for sampler in v26.branches["P5"].downsampling
    ] == [4, 32, 32, 16]

    full_multiscale = MultiScaleProjector(
        [384] * 4, 256, [2.0, 1.0, 0.5], layer_norm=True
    )
    full_v26 = ScaleDecoupledReassemblyProjectorV26(
        [384] * 4, 256, ["P3", "P4", "P5"]
    )
    assert sum(p.numel() for p in full_v26.parameters()) < sum(
        p.numel() for p in full_multiscale.parameters()
    ) * 0.6


@pytest.mark.parametrize(
    ("projector_class", "expected_groups"),
    [
        (ScaleDecoupledReassemblyProjectorV27, 8),
        (ScaleDecoupledReassemblyProjectorV28, 32),
    ],
)
def test_sdsr_deep_group_sweep_matches_v11(projector_class, expected_groups):
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v11 = ScaleDecoupledReassemblyProjectorV11(**kwargs)
    torch.manual_seed(1042)
    candidate = projector_class(**kwargs)
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    expected = v11(features)
    actual = candidate([feature.detach().clone() for feature in features])
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(
            actual_level, expected_level, rtol=2.0e-5, atol=1.0e-6
        )
    groups = [
        sampler.detail.groups
        for sampler in candidate.branches["P5"].downsampling
    ]
    assert groups == [32, 32, 32, expected_groups]


def test_sdsr_v29_adds_zero_initialized_semantic_basis_exactly():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v11 = ScaleDecoupledReassemblyProjectorV11(**kwargs)
    torch.manual_seed(1042)
    v29 = ScaleDecoupledReassemblyProjectorV29(**kwargs)
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    expected = v11(features)
    actual = v29([feature.detach().clone() for feature in features])
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(actual_level, expected_level, rtol=0, atol=0)

    actual[2].square().mean().backward()
    fusion_gradient = v29.branches["P5"].fusion.cv1.conv.weight.grad
    assert fusion_gradient is not None
    semantic_columns = fusion_gradient[:, 8 * 32 :]
    assert semantic_columns.abs().sum() > 0

    full_multiscale = MultiScaleProjector(
        [384] * 4, 256, [2.0, 1.0, 0.5], layer_norm=True
    )
    full_v29 = ScaleDecoupledReassemblyProjectorV29(
        [384] * 4, 256, ["P3", "P4", "P5"]
    )
    assert sum(p.numel() for p in full_v29.parameters()) < sum(
        p.numel() for p in full_multiscale.parameters()
    ) * 0.6


def test_sdsr_v30_adds_an_exact_zero_gated_p3_detail_path():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v23 = ScaleDecoupledReassemblyProjectorV23(**kwargs)
    torch.manual_seed(1042)
    v30 = ScaleDecoupledReassemblyProjectorV30(**kwargs, detail_channels=16)
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]
    image = torch.randn(2, 3, 128, 128)

    expected = v23(features)
    actual = v30(
        [feature.detach().clone() for feature in features], image=image
    )
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(actual_level, expected_level, rtol=0, atol=0)

    actual[0].square().mean().backward()
    gate_gradient = v30.p3_detail.channel_gate.grad
    assert gate_gradient is not None
    assert gate_gradient.abs().sum() > 0

    full_multiscale = MultiScaleProjector(
        [384] * 4, 256, [2.0, 1.0, 0.5], layer_norm=True
    )
    full_v30 = ScaleDecoupledReassemblyProjectorV30(
        [384] * 4, 256, ["P3", "P4", "P5"]
    )
    assert sum(p.numel() for p in full_v30.parameters()) < sum(
        p.numel() for p in full_multiscale.parameters()
    ) * 0.6


def test_sdsr_v31_adds_a_complementary_p5_basis_exactly():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v23 = ScaleDecoupledReassemblyProjectorV23(**kwargs)
    torch.manual_seed(1042)
    v31 = ScaleDecoupledReassemblyProjectorV31(**kwargs)
    features = [torch.randn(2, 32, 8, 8) for _ in range(4)]

    expected = v23(features)
    actual = v31([feature.detach().clone() for feature in features])
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(actual_level, expected_level, rtol=0, atol=0)

    actual[2].square().mean().backward()
    fusion_gradient = v31.branches["P5"].fusion.cv1.conv.weight.grad
    assert fusion_gradient is not None
    semantic_columns = fusion_gradient[:, 8 * 32 :]
    assert semantic_columns.abs().sum() > 0

    full_multiscale = MultiScaleProjector(
        [384] * 4, 256, [2.0, 1.0, 0.5], layer_norm=True
    )
    full_v31 = ScaleDecoupledReassemblyProjectorV31(
        [384] * 4, 256, ["P3", "P4", "P5"]
    )
    assert sum(p.numel() for p in full_v31.parameters()) < sum(
        p.numel() for p in full_multiscale.parameters()
    ) * 0.6


def test_sdsr_v32_adds_zero_gated_phase_calibration_exactly():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v23 = ScaleDecoupledReassemblyProjectorV23(**kwargs)
    torch.manual_seed(1042)
    v32 = ScaleDecoupledReassemblyProjectorV32(**kwargs)
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]

    expected = v23(features)
    actual = v32([feature.detach().clone() for feature in features])
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(actual_level, expected_level, rtol=0, atol=0)

    actual[2].square().mean().backward()
    calibrated = v32.branches["P5"].downsampling[-1]
    assert calibrated.phase_scale.grad is not None
    assert calibrated.phase_scale.grad.abs().sum() > 0

    full_multiscale = MultiScaleProjector(
        [384] * 4, 256, [2.0, 1.0, 0.5], layer_norm=True
    )
    full_v32 = ScaleDecoupledReassemblyProjectorV32(
        [384] * 4, 256, ["P3", "P4", "P5"]
    )
    assert sum(p.numel() for p in full_v32.parameters()) < sum(
        p.numel() for p in full_multiscale.parameters()
    ) * 0.6


def test_sdsr_v33_parameterizes_cross_channel_detail_as_a_gated_residual():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v23 = ScaleDecoupledReassemblyProjectorV23(**kwargs)
    torch.manual_seed(1042)
    v33 = ScaleDecoupledReassemblyProjectorV33(**kwargs)
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]

    expected = v23(features)
    actual = v33([feature.detach().clone() for feature in features])
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(actual_level, expected_level, rtol=0, atol=0)

    (actual[2] * torch.randn_like(actual[2])).mean().backward()
    detail = v33.branches["P5"].downsampling[-1].detail
    residual_gradient = detail.cross_channel.weight.grad
    assert residual_gradient is not None
    assert residual_gradient.abs().sum() > 0
    diagonal_gradient = residual_gradient * (1 - detail.off_diagonal)
    assert diagonal_gradient.abs().sum() == 0

    full_multiscale = MultiScaleProjector(
        [384] * 4, 256, [2.0, 1.0, 0.5], layer_norm=True
    )
    full_v33 = ScaleDecoupledReassemblyProjectorV33(
        [384] * 4, 256, ["P3", "P4", "P5"]
    )
    assert sum(p.numel() for p in full_v33.parameters()) < sum(
        p.numel() for p in full_multiscale.parameters()
    ) * 0.6


def test_sdsr_v34_packs_v23_initialization_and_reduces_parameters():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v23 = ScaleDecoupledReassemblyProjectorV23(**kwargs)
    torch.manual_seed(1042)
    v34 = ScaleDecoupledReassemblyProjectorV34(**kwargs)
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]

    expected = v23(features)
    actual = v34([feature.detach().clone() for feature in features])
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(
            actual_level, expected_level, rtol=2.0e-5, atol=2.0e-6
        )

    (actual[2] * torch.randn_like(actual[2])).mean().backward()
    semantic_gradient = v34.branches["P5"].semantic_mixer.weight.grad
    assert semantic_gradient is not None
    assert semantic_gradient.abs().sum() > 0

    full_v23 = ScaleDecoupledReassemblyProjectorV23(
        [384] * 4, 256, ["P3", "P4", "P5"]
    )
    full_v34 = ScaleDecoupledReassemblyProjectorV34(
        [384] * 4, 256, ["P3", "P4", "P5"]
    )
    assert sum(p.numel() for p in full_v34.parameters()) < sum(
        p.numel() for p in full_v23.parameters()
    )


def test_sdsr_v35_fuses_sampling_without_changing_v34_outputs():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v34 = ScaleDecoupledReassemblyProjectorV34(**kwargs)
    torch.manual_seed(1042)
    v35 = ScaleDecoupledReassemblyProjectorV35(**kwargs)
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]

    expected = v34(features)
    actual = v35([feature.detach().clone() for feature in features])
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(
            actual_level, expected_level, rtol=2.0e-5, atol=2.0e-6
        )
    assert v34.state_dict().keys() == v35.state_dict().keys()

    actual[2].square().mean().backward()
    detail_gradient = v35.branches["P5"].detail.weight.grad
    assert detail_gradient is not None
    assert detail_gradient.abs().sum() > 0


def test_sdsr_v36_packs_the_full_v23_grouped_operator_exactly():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v23 = ScaleDecoupledReassemblyProjectorV23(**kwargs)
    v23_rng = torch.get_rng_state().clone()
    torch.manual_seed(1042)
    v36 = ScaleDecoupledReassemblyProjectorV36(**kwargs)
    v36_rng = torch.get_rng_state().clone()
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]

    expected = v23(features)
    actual = v36([feature.detach().clone() for feature in features])
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(
            actual_level, expected_level, rtol=2.0e-5, atol=2.0e-6
        )
    assert torch.equal(v23_rng, v36_rng)

    (actual[2] * torch.randn_like(actual[2])).mean().backward()
    detail_gradient = v36.branches["P5"].deep_detail.weight.grad
    assert detail_gradient is not None
    assert detail_gradient.abs().sum() > 0

    full_v23 = ScaleDecoupledReassemblyProjectorV23(
        [384] * 4, 256, ["P3", "P4", "P5"]
    )
    full_v36 = ScaleDecoupledReassemblyProjectorV36(
        [384] * 4, 256, ["P3", "P4", "P5"]
    )
    assert sum(p.numel() for p in full_v36.parameters()) == sum(
        p.numel() for p in full_v23.parameters()
    )


@pytest.mark.parametrize(
    ("projector_class", "routed_levels", "extra_parameters"),
    [
        (ScaleDecoupledReassemblyProjectorV37R1, ("P3",), 2 * 384),
        (ScaleDecoupledReassemblyProjectorV37R2, ("P4",), 4 * 384),
        (
            ScaleDecoupledReassemblyProjectorV37R3,
            ("P3", "P4"),
            6 * 384,
        ),
    ],
)
def test_sdsr_v37_routes_layers_without_changing_v23_initial_outputs(
    projector_class,
    routed_levels,
    extra_parameters,
):
    small_kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v23 = ScaleDecoupledReassemblyProjectorV23(**small_kwargs)
    torch.manual_seed(1042)
    routed = projector_class(**small_kwargs)
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]

    expected = v23(features)
    actual = routed([feature.detach().clone() for feature in features])
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(actual_level, expected_level, rtol=0, atol=0)

    sum(output.square().mean() for output in actual).backward()
    assert set(routed.routed_levels) == set(routed_levels)
    for level in routed_levels:
        gradients = [
            route.grad for route in routed.branches[level].route_logits
        ]
        assert all(gradient is not None for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 0

    full_v23 = ScaleDecoupledReassemblyProjectorV23(
        [384] * 4, 256, ["P3", "P4", "P5"]
    )
    full_routed = projector_class(
        [384] * 4, 256, ["P3", "P4", "P5"]
    )
    parameter_delta = sum(p.numel() for p in full_routed.parameters()) - sum(
        p.numel() for p in full_v23.parameters()
    )
    assert parameter_delta == extra_parameters


@pytest.mark.parametrize(
    "projector_class",
    [
        ScaleDecoupledReassemblyProjectorV38P3,
        ScaleDecoupledReassemblyProjectorV38P4,
        ScaleDecoupledReassemblyProjectorV38Both,
    ],
)
def test_sdsr_v38_starts_from_v23_without_advancing_detector_rng(
    projector_class,
):
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "rank_channels": 8,
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v23 = ScaleDecoupledReassemblyProjectorV23(**kwargs)
    v23_rng = torch.get_rng_state().clone()
    torch.manual_seed(1042)
    candidate = projector_class(**kwargs)
    candidate_rng = torch.get_rng_state().clone()
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]

    expected = v23(features)
    actual = candidate([feature.detach().clone() for feature in features])
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(actual_level, expected_level, rtol=0, atol=0)
    assert torch.equal(candidate_rng, v23_rng)


def test_sdsr_v38_adaptive_paths_receive_gradients_and_normalize_routes():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "rank_channels": 8,
        "num_blocks": 2,
    }
    model = ScaleDecoupledReassemblyProjectorV38Both(**kwargs)
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]
    outputs = model(features)
    sum(
        (output * torch.randn_like(output)).mean() for output in outputs
    ).backward()

    p3 = model.branches["P3"]
    p4 = model.branches["P4"]
    assert p3.residual_scales.grad is not None
    assert p3.residual_scales.grad.abs().sum() > 0
    assert p4.layer_router.router[-1].weight.grad is not None
    assert p4.layer_router.router[-1].weight.grad.abs().sum() > 0

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        p3.residual_scales.fill_(0.1)
        p4.layer_router.router[-1].weight.normal_(std=0.05)
    outputs = model(features)
    sum(
        (output * torch.randn_like(output)).mean() for output in outputs
    ).backward()
    predictor_gradient = p3.local_residual.weight_predictor.weight.grad
    assert predictor_gradient is not None
    assert predictor_gradient.abs().sum() > 0

    gates = p4.layer_router(features)
    torch.testing.assert_close(
        gates.sum(dim=-1),
        torch.full((2,), 4.0),
    )
    assert not torch.allclose(gates[0], gates[1])


def test_sdsr_v39_static_p4_starts_exactly_from_v23_and_learns_four_logits():
    kwargs = {
        "in_channels": [32] * 4,
        "out_channels": 24,
        "levels": ["P3", "P4", "P5"],
        "rank_channels": 8,
        "num_blocks": 2,
    }
    torch.manual_seed(1042)
    v23 = ScaleDecoupledReassemblyProjectorV23(**kwargs)
    v23_rng = torch.get_rng_state().clone()
    torch.manual_seed(1042)
    candidate = ScaleDecoupledReassemblyProjectorV39P4Static(**kwargs)
    candidate_rng = torch.get_rng_state().clone()
    features = [torch.randn(2, 32, 9, 11) for _ in range(4)]

    expected = v23(features)
    actual = candidate([feature.detach().clone() for feature in features])
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(actual_level, expected_level, rtol=0, atol=0)
    assert torch.equal(candidate_rng, v23_rng)

    p4 = candidate.branches["P4"]
    assert p4.layer_logits.numel() == 4
    torch.testing.assert_close(p4.normalized_gates(), torch.ones(4))
    sum(output.square().mean() for output in actual).backward()
    assert p4.layer_logits.grad is not None
    assert p4.layer_logits.grad.abs().sum() > 0
