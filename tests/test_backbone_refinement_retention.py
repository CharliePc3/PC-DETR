import torch
from torch import nn

from rfdetr.engine import (
    backbone_parameter_anchor_loss,
    build_backbone_parameter_anchor,
)
from rfdetr.models.backbone.backbone import is_refined_backbone_parameter


def test_refinement_parameter_matcher_selects_blocks_and_final_norm():
    blocks = (8, 9, 10, 11)
    assert is_refined_backbone_parameter(
        "backbone.0.encoder.encoder.blocks.8.attn.qkv.weight", blocks
    )
    assert is_refined_backbone_parameter(
        "backbone.0.encoder.encoder.norm.weight", blocks
    )
    assert not is_refined_backbone_parameter(
        "backbone.0.encoder.encoder.blocks.7.attn.qkv.weight", blocks
    )
    assert not is_refined_backbone_parameter(
        "backbone.0.projector.stages.0.weight", blocks
    )


class _TinyBackboneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "encoder": nn.ModuleDict(
                            {
                                "encoder": nn.ModuleDict(
                                    {
                                        "blocks": nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)]),
                                        "norm": nn.LayerNorm(2),
                                    }
                                )
                            }
                        )
                    }
                )
            ]
        )


def test_backbone_anchor_is_zero_then_backpropagates_to_selected_parameters():
    model = _TinyBackboneModel()
    anchors = build_backbone_parameter_anchor(model, (1,))
    assert anchors
    assert backbone_parameter_anchor_loss(anchors).item() == 0.0

    with torch.no_grad():
        model.backbone[0]["encoder"]["encoder"]["blocks"][1].weight.add_(0.25)
    loss = backbone_parameter_anchor_loss(anchors)
    assert loss.item() > 0
    loss.backward()

    assert model.backbone[0]["encoder"]["encoder"]["blocks"][1].weight.grad is not None
    assert model.backbone[0]["encoder"]["encoder"]["blocks"][0].weight.grad is None
