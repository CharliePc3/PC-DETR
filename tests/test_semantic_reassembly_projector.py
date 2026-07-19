import pytest
import torch
import torch.nn.functional as F

from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.backbone.semantic_reassembly_projector import (
    ScaleDecoupledReassemblyProjector,
    SemanticLocalReassembly,
)


@pytest.mark.parametrize(
    ("use_local", "use_guide", "use_phase"),
    [(True, True, True), (True, False, True), (False, False, True), (True, True, False)],
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
    reference = F.interpolate(source, scale_factor=2, mode="bilinear", align_corners=False)

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
    model = ScaleDecoupledReassemblyProjector([32] * 4, 24, ["P3", "P4", "P5"], detail_channels=8)
    features = [torch.randn(2, 32, 8, 8, requires_grad=True) for _ in range(4)]
    outputs = model(features, image=torch.randn(2, 3, 128, 128))

    sum((index + 1) * output.square().mean() for index, output in enumerate(outputs)).backward()

    gradients = [
        model.router.route_logits.grad,
        model.local_reassembly.weight_predictor[-1].weight.grad,
        model.detail_stem.stem[0].weight.grad,
        model.p5_downsample.phase_logits.weight.grad,
        model.p5_downsample.gate.weight.grad,
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)

    full_sdsr = ScaleDecoupledReassemblyProjector([384] * 4, 256, ["P3", "P4", "P5"], detail_channels=32)
    full_multiscale = MultiScaleProjector([384] * 4, 256, [2.0, 1.0, 0.5], layer_norm=True)
    sdsr_parameters = sum(parameter.numel() for parameter in full_sdsr.parameters())
    multiscale_parameters = sum(parameter.numel() for parameter in full_multiscale.parameters())
    assert sdsr_parameters < multiscale_parameters * 0.1
