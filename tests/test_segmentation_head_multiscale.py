import torch
import torch.nn.functional as F

from rfdetr.models.segmentation_head import (
    SegmentationHead,
    get_target_boundary_point_coords,
)


def test_zero_gated_multiscale_head_matches_single_level_head():
    torch.manual_seed(7)
    single = SegmentationHead(32, num_blocks=2, downsample_ratio=4)
    multi = SegmentationHead(
        32,
        num_blocks=2,
        downsample_ratio=4,
        num_feature_levels=3,
    )
    multi.load_state_dict(single.state_dict(), strict=False)

    p3 = torch.randn(2, 32, 8, 8)
    p4 = torch.randn(2, 32, 4, 4)
    p5 = torch.randn(2, 32, 2, 2)
    queries = [torch.randn(2, 5, 32), torch.randn(2, 5, 32)]

    single_outputs = single(p3, queries, (64, 64))
    multi_outputs = multi([p3, p4, p5], queries, (64, 64))

    assert len(single_outputs) == len(multi_outputs) == 2
    for single_output, multi_output in zip(single_outputs, multi_outputs):
        torch.testing.assert_close(single_output, multi_output)


def test_multiscale_context_receives_gradients_after_gate_opens():
    torch.manual_seed(11)
    head = SegmentationHead(
        32,
        num_blocks=1,
        downsample_ratio=4,
        num_feature_levels=3,
    )
    with torch.no_grad():
        head.context_gates.fill_(0.1)

    features = [
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 32, 4, 4),
        torch.randn(2, 32, 2, 2),
    ]
    outputs = head(features, [torch.randn(2, 5, 32)], (64, 64))
    assert outputs[0].shape == (2, 5, 16, 16)

    outputs[0].square().mean().backward()
    assert head.context_gates.grad is not None
    assert head.context_gates.grad.abs().sum() > 0
    assert head.context_projs[0][0].weight.grad is not None
    assert head.context_projs[0][0].weight.grad.abs().sum() > 0


def test_target_boundary_sampler_selects_morphological_boundary_pixels():
    masks = torch.zeros(1, 1, 7, 7)
    masks[:, :, 2:5, 2:5] = 1
    torch.manual_seed(3)
    coords = get_target_boundary_point_coords(masks, num_points=64)

    indices = (coords * 7).floor().long()
    sampled_x = indices[..., 0].clamp_max(6)
    sampled_y = indices[..., 1].clamp_max(6)
    dilated = F.max_pool2d(masks, 3, stride=1, padding=1)
    eroded = 1 - F.max_pool2d(1 - masks, 3, stride=1, padding=1)
    boundary = (dilated - eroded) > 0
    assert boundary[0, 0, sampled_y, sampled_x].all()


def test_target_boundary_sampler_handles_empty_masks_and_zero_points():
    masks = torch.zeros(2, 1, 5, 4)
    coords = get_target_boundary_point_coords(masks, num_points=9)
    assert coords.shape == (2, 9, 2)
    assert (coords > 0).all() and (coords < 1).all()
    assert get_target_boundary_point_coords(masks, num_points=0).shape == (2, 0, 2)
