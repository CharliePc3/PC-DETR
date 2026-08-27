import unittest

import torch
from torch import nn

from rfdetr.models.lwdetr import LWDETR
from rfdetr.models.transformer import MLP, Transformer
from rfdetr.util.misc import NestedTensor


def _decoder_with_head(mode):
    torch.manual_seed(17)
    transformer = Transformer(
        d_model=32,
        sa_nhead=4,
        ca_nhead=4,
        num_decoder_layers=4,
        dim_feedforward=64,
        num_feature_levels=3,
        dec_n_points=2,
        lite_refpoint_refine=False,
    )
    head = MLP(32, 32, 4, 3)
    torch.nn.init.constant_(head.layers[-1].weight, 0.0)
    torch.nn.init.constant_(head.layers[-1].bias, 0.0)
    transformer.decoder.configure_bbox_refinement(head, mode)
    return transformer.decoder


class _SyntheticBackbone(nn.Module):
    def forward(self, samples):
        batch_size = samples.tensors.shape[0]
        device = samples.tensors.device
        features = torch.ones(batch_size, 32, 8, 8, device=device)
        mask = torch.zeros(batch_size, 8, 8, dtype=torch.bool, device=device)
        position = torch.zeros_like(features)
        return [NestedTensor(features, mask)], [position]


def _detector(mode):
    torch.manual_seed(23)
    transformer = Transformer(
        d_model=32,
        sa_nhead=4,
        ca_nhead=4,
        num_queries=7,
        num_decoder_layers=4,
        dim_feedforward=64,
        return_intermediate_dec=True,
        num_feature_levels=1,
        dec_n_points=2,
        lite_refpoint_refine=False,
        bbox_reparam=True,
    )
    model = LWDETR(
        backbone=_SyntheticBackbone(),
        transformer=transformer,
        segmentation_head=None,
        num_classes=5,
        num_queries=7,
        aux_loss=True,
        group_detr=1,
        two_stage=False,
        lite_refpoint_refine=False,
        bbox_refine_mode=mode,
        bbox_reparam=True,
    )
    model.refpoint_embed.weight.data.copy_(
        torch.tensor([0.5, 0.5, 0.2, 0.2]).repeat(7, 1)
    )
    return model


class BBoxRefinementTest(unittest.TestCase):
    def test_specialized_bbox_heads_preserve_shared_initial_output(self):
        for mode in ("layerwise", "residual"):
            with self.subTest(mode=mode):
                shared = _decoder_with_head("shared")
                specialized = _decoder_with_head(mode)
                hidden_states = torch.randn(4, 2, 7, 32)

                torch.testing.assert_close(
                    specialized.bbox_deltas(hidden_states),
                    shared.bbox_deltas(hidden_states),
                    rtol=0,
                    atol=0,
                )

                specialized_parameters = dict(specialized.named_parameters())
                for name, parameter in shared.named_parameters():
                    torch.testing.assert_close(
                        specialized_parameters[name],
                        parameter,
                        rtol=0,
                        atol=0,
                    )

    def test_layerwise_bbox_heads_receive_independent_gradients(self):
        decoder = _decoder_with_head("layerwise")
        hidden_states = torch.randn(4, 2, 7, 32)

        decoder.bbox_deltas(hidden_states).sum().backward()

        heads = decoder.bbox_embed_layers
        self.assertIs(heads[0], decoder.bbox_embed)
        self.assertIsNot(heads[0], heads[1])
        self.assertGreater(torch.count_nonzero(heads[0].layers[-1].weight.grad), 0)
        self.assertGreater(torch.count_nonzero(heads[1].layers[-1].weight.grad), 0)
        self.assertFalse(
            torch.equal(
                heads[0].layers[-1].weight.grad,
                heads[1].layers[-1].weight.grad,
            )
        )

    def test_residual_bbox_heads_start_zero_and_receive_gradients(self):
        decoder = _decoder_with_head("residual")
        hidden_states = torch.randn(4, 2, 7, 32)

        decoder.bbox_deltas(hidden_states).sum().backward()

        for residual in decoder.bbox_embed_residuals:
            self.assertEqual(torch.count_nonzero(residual.weight), 0)
            self.assertIsNotNone(residual.weight.grad)
            self.assertGreater(torch.count_nonzero(residual.weight.grad), 0)

    def test_full_detector_forward_matches_shared_initialization(self):
        samples = torch.randn(2, 3, 32, 32)
        shared = _detector("shared").eval()
        with torch.no_grad():
            expected = shared(samples)

        for mode in ("layerwise", "residual"):
            with self.subTest(mode=mode):
                specialized = _detector(mode).eval()
                with torch.no_grad():
                    actual = specialized(samples)

                torch.testing.assert_close(actual["pred_logits"], expected["pred_logits"])
                torch.testing.assert_close(actual["pred_boxes"], expected["pred_boxes"])
                for actual_aux, expected_aux in zip(
                    actual["aux_outputs"],
                    expected["aux_outputs"],
                ):
                    torch.testing.assert_close(
                        actual_aux["pred_logits"],
                        expected_aux["pred_logits"],
                    )
                    torch.testing.assert_close(
                        actual_aux["pred_boxes"],
                        expected_aux["pred_boxes"],
                    )


if __name__ == "__main__":
    unittest.main()
