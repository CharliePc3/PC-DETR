import torch
from torch import nn

from rfdetr.models.backbone.projector import MultiScaleProjector


PRIOR = (0.2485995, 0.4129705, 0.7943345, 2.5440953)


def _p4_projector(prior=None):
    return MultiScaleProjector(
        in_channels=[4, 4, 4, 4],
        out_channels=4,
        scale_factors=[1.0],
        num_blocks=0,
        p4_depth_prior=prior,
    )


def test_fixed_prior_is_applied_after_sampling_before_fusion():
    projector = _p4_projector(PRIOR)
    projector.stages[0] = nn.Identity()
    features = [torch.full((1, 4, 3, 3), float(index + 1)) for index in range(4)]

    fused_input = projector(features)[0]

    expected = torch.cat(
        [feature * weight for feature, weight in zip(features, PRIOR)], dim=1
    )
    torch.testing.assert_close(fused_input, expected)


def test_prior_is_not_parameter_buffer_or_checkpoint_state():
    projector = _p4_projector(PRIOR)

    assert projector.p4_depth_prior == PRIOR
    assert "p4_depth_prior" not in dict(projector.named_parameters())
    assert "p4_depth_prior" not in dict(projector.named_buffers())
    assert all("p4_depth_prior" not in key for key in projector.state_dict())


def test_original_path_matches_explicit_unit_weights():
    torch.manual_seed(123)
    original = _p4_projector()
    torch.manual_seed(123)
    explicit_uniform = _p4_projector((1.0, 1.0, 1.0, 1.0))
    features = [torch.randn(2, 4, 5, 5) for _ in range(4)]

    assert original.state_dict().keys() == explicit_uniform.state_dict().keys()
    torch.testing.assert_close(original(features)[0], explicit_uniform(features)[0])


def test_invalid_prior_is_rejected():
    for prior in ((1.0, 1.0), (1.0, 1.0, 1.0, 0.0)):
        try:
            _p4_projector(prior)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid prior to fail: {prior}")

    try:
        MultiScaleProjector(
            in_channels=[4] * 4,
            out_channels=4,
            scale_factors=[2.0],
            p4_depth_prior=PRIOR,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a P4 prior without P4 to fail")
