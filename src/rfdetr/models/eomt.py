"""Minimal encoder-only mask transformer for DINOv3 experiments.

This is an executable EoMT-style research control rather than a drop-in
replacement for RF-DETR.  It inserts learned segmentation queries before the
patch tokens for the last few native DINOv3 blocks and predicts masks by a dot
product with an upsampled final patch map.  Keeping it isolated makes the first
comparison explicit before coupling these queries to RF-DETR detection queries.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskEmbeddingMLP(nn.Module):
    """Three-layer query projection used by mask-transformer heads."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return self.layers(queries)


class EncoderOnlyMaskTransformer(nn.Module):
    """Inject segmentation queries into the final native DINOv3 blocks.

    The supplied encoder must expose the public attributes used by the local
    DINOv3 implementation: ``prepare_tokens_with_masks``, ``blocks``,
    ``rope_embed``, ``norm``, ``n_storage_tokens``, and ``embed_dim``.

    By default this is the clean unmasked architectural control.  The optional
    masked-attention path predicts a mask before each query block and constrains
    only query-to-patch self-attention. Its probabilities can be annealed to zero
    so evaluation returns to the untouched native DINOv3 block implementation.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        *,
        num_queries: int = 100,
        num_query_blocks: int = 2,
        mask_dim: int = 256,
        mask_downsample_ratio: int = 4,
        freeze_image_prefix: bool = False,
        masked_attention: bool = False,
        mask_annealing_power: float = 0.9,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive.")
        if num_queries < 1:
            raise ValueError("num_queries must be positive.")
        if not 1 <= num_query_blocks <= len(encoder.blocks):
            raise ValueError(
                "num_query_blocks must be between one and the encoder depth."
            )
        if mask_dim < 1 or mask_downsample_ratio < 1:
            raise ValueError("mask dimensions and downsample ratio must be positive.")
        if mask_annealing_power <= 0:
            raise ValueError("mask_annealing_power must be positive.")

        self.encoder = encoder
        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.num_query_blocks = int(num_query_blocks)
        self.mask_downsample_ratio = int(mask_downsample_ratio)
        self.masked_attention = bool(masked_attention)
        self.mask_annealing_power = float(mask_annealing_power)
        self.masking_probabilities = [1.0] * self.num_query_blocks
        embed_dim = int(encoder.embed_dim)

        self.segmentation_queries = nn.Parameter(
            torch.empty(1, self.num_queries, embed_dim)
        )
        self.class_embed = nn.Linear(embed_dim, self.num_classes + 1)
        self.mask_embed = MaskEmbeddingMLP(embed_dim, mask_dim)
        self.pixel_embed = nn.Conv2d(embed_dim, mask_dim, kernel_size=1)
        self.mask_bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.segmentation_queries, std=0.02)

        if freeze_image_prefix:
            self.freeze_image_prefix()

    @torch.no_grad()
    def initialize_queries_from_detection(
        self,
        detection_queries: torch.Tensor,
        *,
        seed: int = 0,
    ) -> dict[str, float | int]:
        """Lift pretrained detector slots into the DINO query space.

        A fixed semi-orthogonal lift preserves the normalized detector-query
        Gram matrix without adding a learned bridge, then restores the 0.02
        standard deviation used by the random-query baseline.
        """

        if detection_queries.ndim != 2:
            raise ValueError("detection_queries must have shape [queries, channels].")
        if detection_queries.shape[0] < self.num_queries:
            raise ValueError(
                f"Need at least {self.num_queries} detection queries, received "
                f"{detection_queries.shape[0]}."
            )
        target_dim = self.segmentation_queries.shape[-1]
        source = detection_queries[: self.num_queries].detach().float().cpu()
        source = F.layer_norm(source, (source.shape[-1],))
        source_dim = source.shape[-1]
        generator = torch.Generator(device="cpu").manual_seed(seed)

        if source_dim == target_dim:
            lifted = source
        elif source_dim < target_dim:
            random_basis = torch.randn(
                target_dim, source_dim, generator=generator
            )
            basis, _ = torch.linalg.qr(random_basis, mode="reduced")
            lifted = source @ basis.T
        else:
            random_basis = torch.randn(
                source_dim, target_dim, generator=generator
            )
            basis, _ = torch.linalg.qr(random_basis, mode="reduced")
            lifted = source @ basis

        lifted = lifted - lifted.mean()
        lifted = lifted * (0.02 / lifted.std().clamp_min(1e-6))
        self.segmentation_queries.copy_(
            lifted.to(
                device=self.segmentation_queries.device,
                dtype=self.segmentation_queries.dtype,
            ).unsqueeze(0)
        )
        return {
            "source_queries": int(detection_queries.shape[0]),
            "source_dim": int(source_dim),
            "target_dim": int(target_dim),
            "seed": int(seed),
            "initialized_std": float(self.segmentation_queries.float().std()),
        }

    @property
    def query_start_block(self) -> int:
        return len(self.encoder.blocks) - self.num_query_blocks

    def freeze_image_prefix(self) -> None:
        """Freeze image tokenization and blocks preceding query insertion."""

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for block in self.encoder.blocks[self.query_start_block :]:
            block.requires_grad_(True)
        self.encoder.norm.requires_grad_(True)

    def set_mask_annealing_progress(self, progress: float) -> list[float]:
        """Anneal early query blocks before later ones, following EoMT Fig. 4.

        ``progress`` is the normalized training progress in [0, 1]. At zero,
        every query block uses its predicted attention mask. At one, all blocks
        use ordinary self-attention. Earlier blocks reach zero first.
        """

        if not 0.0 <= progress <= 1.0:
            raise ValueError("mask annealing progress must lie in [0, 1].")
        depth = self.num_query_blocks
        self.masking_probabilities = [
            max(1.0 - progress * depth / (block_index + 1), 0.0)
            ** self.mask_annealing_power
            for block_index in range(depth)
        ]
        return list(self.masking_probabilities)

    def _query_to_patch_attention_mask(
        self,
        mask_logits: torch.Tensor,
        *,
        num_tokens: int,
        query_start: int,
        patch_start: int,
        patch_height: int,
        patch_width: int,
        masking_probability: float,
    ) -> torch.Tensor | None:
        if not self.training or masking_probability <= 0:
            return None
        regions = F.interpolate(
            mask_logits.detach().float(),
            size=(patch_height, patch_width),
            mode="bilinear",
            align_corners=False,
        ).flatten(2) > 0
        empty = ~regions.any(dim=-1, keepdim=True)
        regions = regions | empty
        if masking_probability < 1:
            apply_mask = (
                torch.rand(
                    regions.shape[0],
                    regions.shape[1],
                    1,
                    device=regions.device,
                )
                < masking_probability
            )
            regions = regions | ~apply_mask

        allowed = torch.ones(
            regions.shape[0],
            1,
            num_tokens,
            num_tokens,
            dtype=torch.bool,
            device=regions.device,
        )
        allowed[
            :,
            0,
            query_start : query_start + self.num_queries,
            patch_start:,
        ] = regions
        return allowed

    @staticmethod
    def _masked_block_forward(
        block: nn.Module,
        tokens: torch.Tensor,
        rope: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
        allowed_attention: torch.Tensor,
    ) -> torch.Tensor:
        """Run a native DINOv3 block with a query-to-patch SDPA mask."""

        if block.training and getattr(block, "sample_drop_ratio", 0.0) > 0:
            raise RuntimeError(
                "Masked EoMT attention currently requires zero stochastic depth "
                "in the query blocks."
            )
        attention = block.attn
        normalized = block.norm1(tokens)
        batch, num_tokens, channels = normalized.shape
        qkv = attention.qkv(normalized).reshape(
            batch,
            num_tokens,
            3,
            attention.num_heads,
            channels // attention.num_heads,
        )
        query, key, value = torch.unbind(qkv, dim=2)
        query, key, value = [tensor.transpose(1, 2) for tensor in (query, key, value)]
        if rope is not None:
            query, key = attention.apply_rope(query, key, rope)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed_attention,
        )
        attended = attended.transpose(1, 2).reshape(batch, num_tokens, channels)
        attended = attention.proj_drop(attention.proj(attended))
        tokens = tokens + block.ls1(attended)
        return tokens + block.ls2(block.mlp(block.norm2(tokens)))

    def _prediction(
        self,
        tokens: torch.Tensor,
        *,
        patch_height: int,
        patch_width: int,
        special_tokens: int,
        image_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        normalized = self.encoder.norm(tokens)
        query_end = special_tokens + self.num_queries
        query_features = normalized[:, special_tokens:query_end]
        patch_features = normalized[:, query_end:]
        expected_patches = patch_height * patch_width
        if patch_features.shape[1] != expected_patches:
            raise RuntimeError(
                f"Expected {expected_patches} patch tokens, got {patch_features.shape[1]}."
            )

        patch_map = patch_features.reshape(
            patch_features.shape[0], patch_height, patch_width, -1
        ).permute(0, 3, 1, 2)
        target_size = (
            image_size[0] // self.mask_downsample_ratio,
            image_size[1] // self.mask_downsample_ratio,
        )
        pixel_embeddings = self.pixel_embed(
            F.interpolate(
                patch_map,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        )
        query_embeddings = self.mask_embed(query_features)
        masks = (
            torch.einsum("bchw,bqc->bqhw", pixel_embeddings, query_embeddings)
            + self.mask_bias
        )
        return {
            "pred_logits": self.class_embed(query_features),
            "pred_masks": masks,
            "query_features": query_features,
            "patch_features": patch_map,
        }

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        if images.ndim != 4:
            raise ValueError("images must have shape [B, 3, H, W].")
        image_height, image_width = images.shape[-2:]
        tokens, (patch_height, patch_width) = self.encoder.prepare_tokens_with_masks(
            images
        )
        rope = (
            self.encoder.rope_embed(H=patch_height, W=patch_width)
            if self.encoder.rope_embed is not None
            else None
        )

        for block in self.encoder.blocks[: self.query_start_block]:
            tokens = block(tokens, rope)

        special_tokens = int(self.encoder.n_storage_tokens) + 1
        queries = self.segmentation_queries.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat(
            (tokens[:, :special_tokens], queries, tokens[:, special_tokens:]),
            dim=1,
        )

        predictions = []
        query_start = special_tokens
        patch_start = special_tokens + self.num_queries
        for block_index, block in enumerate(
            self.encoder.blocks[self.query_start_block :]
        ):
            # DINOv3 applies RoPE only to the final H*W tokens. The injected
            # queries are therefore treated as additional unrotated prefix
            # tokens without modifying the pretrained block implementation.
            if self.masked_attention:
                intermediate = self._prediction(
                    tokens,
                    patch_height=patch_height,
                    patch_width=patch_width,
                    special_tokens=special_tokens,
                    image_size=(image_height, image_width),
                )
                predictions.append(intermediate)
                allowed_attention = self._query_to_patch_attention_mask(
                    intermediate["pred_masks"],
                    num_tokens=tokens.shape[1],
                    query_start=query_start,
                    patch_start=patch_start,
                    patch_height=patch_height,
                    patch_width=patch_width,
                    masking_probability=self.masking_probabilities[block_index],
                )
                tokens = (
                    block(tokens, rope)
                    if allowed_attention is None
                    else self._masked_block_forward(
                        block, tokens, rope, allowed_attention
                    )
                )
            else:
                tokens = block(tokens, rope)
            if not self.masked_attention:
                predictions.append(
                    self._prediction(
                        tokens,
                        patch_height=patch_height,
                        patch_width=patch_width,
                        special_tokens=special_tokens,
                        image_size=(image_height, image_width),
                    )
                )

        if self.masked_attention:
            predictions.append(
                self._prediction(
                    tokens,
                    patch_height=patch_height,
                    patch_width=patch_width,
                    special_tokens=special_tokens,
                    image_size=(image_height, image_width),
                )
            )

        output = predictions[-1]
        output["aux_outputs"] = [
            {
                "pred_logits": prediction["pred_logits"],
                "pred_masks": prediction["pred_masks"],
            }
            for prediction in predictions[:-1]
        ]
        return output
