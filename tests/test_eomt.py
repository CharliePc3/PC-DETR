import torch
import torch.nn as nn

from rfdetr.models.eomt import EncoderOnlyMaskTransformer


class _FakeAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.num_heads = 4
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Identity()


class _FakeBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _FakeAttention(dim)
        self.ls1 = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.ls2 = nn.Identity()
        self.sample_drop_ratio = 0.0

    def forward(self, tokens, rope=None):
        qkv = self.attn.qkv(self.norm1(tokens))
        batch, length, channels3 = qkv.shape
        channels = channels3 // 3
        qkv = qkv.reshape(batch, length, 3, self.attn.num_heads, channels // self.attn.num_heads)
        query, key, value = [part.transpose(1, 2) for part in torch.unbind(qkv, dim=2)]
        attended = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, length, channels)
        tokens = tokens + self.attn.proj(attended)
        return tokens + self.mlp(self.norm2(tokens))


class _FakeEncoder(nn.Module):
    def __init__(self, dim: int = 16, depth: int = 4) -> None:
        super().__init__()
        self.embed_dim = dim
        self.patch_size = 4
        self.n_storage_tokens = 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, 1, dim))
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.blocks = nn.ModuleList(_FakeBlock(dim) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.rope_embed = None

    def prepare_tokens_with_masks(self, images, masks=None):
        patches = self.patch_embed(images)
        height, width = patches.shape[-2:]
        patches = patches.flatten(2).transpose(1, 2)
        batch = images.shape[0]
        tokens = torch.cat(
            (
                self.cls_token.expand(batch, -1, -1),
                self.storage_tokens.expand(batch, -1, -1),
                patches,
            ),
            dim=1,
        )
        return tokens, (height, width)


def test_encoder_only_mask_transformer_shapes_and_gradients():
    model = EncoderOnlyMaskTransformer(
        _FakeEncoder(),
        num_classes=5,
        num_queries=7,
        num_query_blocks=2,
        mask_dim=8,
        mask_downsample_ratio=4,
    )
    output = model(torch.randn(2, 3, 32, 48))

    assert output["pred_logits"].shape == (2, 7, 6)
    assert output["pred_masks"].shape == (2, 7, 8, 12)
    assert output["query_features"].shape == (2, 7, 16)
    assert output["patch_features"].shape == (2, 16, 8, 12)
    assert len(output["aux_outputs"]) == 1

    output["pred_masks"].mean().backward()
    assert model.segmentation_queries.grad is not None


def test_freeze_image_prefix_leaves_tail_trainable():
    encoder = _FakeEncoder(depth=4)
    model = EncoderOnlyMaskTransformer(
        encoder,
        num_classes=2,
        num_query_blocks=2,
        freeze_image_prefix=True,
    )

    assert not any(parameter.requires_grad for parameter in encoder.blocks[0].parameters())
    assert all(parameter.requires_grad for parameter in encoder.blocks[-1].parameters())
    assert all(parameter.requires_grad for parameter in model.mask_embed.parameters())


def test_masked_attention_shapes_annealing_and_gradients():
    model = EncoderOnlyMaskTransformer(
        _FakeEncoder(),
        num_classes=3,
        num_queries=5,
        num_query_blocks=2,
        mask_dim=8,
        masked_attention=True,
    )
    assert model.set_mask_annealing_progress(0.0) == [1.0, 1.0]
    output = model(torch.randn(2, 3, 32, 32))

    assert output["pred_masks"].shape == (2, 5, 8, 8)
    assert len(output["aux_outputs"]) == 2
    output["pred_masks"].mean().backward()
    assert model.segmentation_queries.grad is not None

    probabilities = model.set_mask_annealing_progress(1.0)
    assert probabilities == [0.0, 0.0]


def test_detection_query_initialization_is_deterministic_and_scaled():
    torch.manual_seed(5)
    source_queries = torch.randn(12, 6)
    first = EncoderOnlyMaskTransformer(
        _FakeEncoder(dim=8, depth=3),
        num_classes=2,
        num_queries=5,
        num_query_blocks=1,
        mask_dim=4,
    )
    second = EncoderOnlyMaskTransformer(
        _FakeEncoder(dim=8, depth=3),
        num_classes=2,
        num_queries=5,
        num_query_blocks=1,
        mask_dim=4,
    )

    metadata = first.initialize_queries_from_detection(source_queries, seed=17)
    second.initialize_queries_from_detection(source_queries, seed=17)

    torch.testing.assert_close(first.segmentation_queries, second.segmentation_queries)
    assert first.segmentation_queries.shape == (1, 5, 8)
    assert abs(float(first.segmentation_queries.std()) - 0.02) < 1e-6
    assert metadata["source_dim"] == 6
    assert metadata["target_dim"] == 8
