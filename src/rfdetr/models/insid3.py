"""Training-free in-context segmentation utilities for DINOv3 features.

This module implements the core INSID3 mechanism as a lightweight diagnostic:
positional-subspace removal, backward correspondence filtering, target-feature
clustering, seed selection, and cluster aggregation.  It intentionally does
not include optional CRF refinement or dataset-specific episode construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class InContextSegmentationResult:
    """Patch-resolution output and diagnostics for one reference/target pair."""

    mask: torch.Tensor
    candidate_mask: torch.Tensor
    cluster_labels: torch.Tensor
    seed_cluster: int
    num_clusters: int
    num_candidate_patches: int
    cluster_scores: torch.Tensor
    score_map: torch.Tensor


def compute_cluster_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return normalized mean features for every integer cluster label."""

    if features.ndim != 2:
        raise ValueError("features must have shape [num_patches, channels].")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must have one entry per feature row.")
    if labels.numel() == 0:
        raise ValueError("at least one patch is required.")

    num_clusters = int(labels.max().item()) + 1
    sums = features.new_zeros((num_clusters, features.shape[1]))
    sums.index_add_(0, labels, features)
    counts = torch.bincount(labels, minlength=num_clusters).to(features.dtype)
    prototypes = sums / counts.clamp_min(1).unsqueeze(1)
    return F.normalize(prototypes, dim=1)


def agglomerative_cluster_labels(
    features: torch.Tensor,
    similarity_threshold: float = 0.6,
) -> torch.Tensor:
    """Cluster normalized patch features with average-link cosine distance."""

    if not -1.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (-1, 1].")
    if features.ndim != 2:
        raise ValueError("features must have shape [num_patches, channels].")

    normalized = F.normalize(features.float(), dim=1)
    distance = (1.0 - normalized @ normalized.T).clamp_min(0)
    distance.fill_diagonal_(0)
    distance_array = distance.cpu().numpy()
    try:
        from sklearn.cluster import AgglomerativeClustering

        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=float(1.0 - similarity_threshold),
        )
        labels = clusterer.fit_predict(distance_array)
    except ModuleNotFoundError:
        # The project's CUDA environment deliberately stays lean but includes
        # SciPy. Its average-link hierarchy is equivalent for this thresholded
        # precomputed cosine-distance use case.
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        hierarchy = linkage(squareform(distance_array, checks=False), method="average")
        labels = fcluster(
            hierarchy,
            t=float(1.0 - similarity_threshold),
            criterion="distance",
        ) - 1
    return torch.as_tensor(labels, dtype=torch.long, device=features.device)


class DinoV3InContextSegmenter(nn.Module):
    """Decoder-free one-shot segmentation on top of a frozen DINOv3 encoder."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @property
    def patch_size(self) -> int:
        patch_size = getattr(self.encoder, "patch_size")
        return int(max(patch_size)) if isinstance(patch_size, tuple) else int(patch_size)

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract normalized last-layer features as ``[B, C, H, W]``."""

        if images.ndim != 4:
            raise ValueError("images must have shape [B, 3, H, W].")
        if images.shape[-2] % self.patch_size or images.shape[-1] % self.patch_size:
            raise ValueError("image height and width must be divisible by patch_size.")

        device_type = images.device.type
        if device_type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = self.encoder.get_intermediate_layers(
                    images,
                    n=1,
                    reshape=True,
                    return_class_token=False,
                )[0]
        else:
            features = self.encoder.get_intermediate_layers(
                images,
                n=1,
                reshape=True,
                return_class_token=False,
            )[0]
        return F.normalize(features.float(), dim=1)

    @torch.no_grad()
    def build_positional_basis(
        self,
        image_size: tuple[int, int],
        max_components: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Estimate the DINOv3 positional subspace from a constant image."""

        if max_components < 0:
            raise ValueError("max_components must be non-negative.")
        height, width = image_size
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size.")
        if max_components == 0:
            embed_dim = int(getattr(self.encoder, "embed_dim"))
            return torch.empty(embed_dim, 0, device=device)

        mean = torch.tensor((0.485, 0.456, 0.406), device=device)[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), device=device)[:, None, None]
        constant_image = ((torch.zeros(1, 3, height, width, device=device) - mean) / std)
        features = self.extract_features(constant_image)[0].flatten(1)
        centered = features - features.mean(dim=1, keepdim=True)
        left_vectors, _, _ = torch.linalg.svd(centered.float(), full_matrices=False)

        # Retaining at least one channel prevents a silent all-zero projection
        # when a Large-model component count is reused with DINOv3-Small.
        component_count = min(
            int(max_components),
            left_vectors.shape[1],
            left_vectors.shape[0] - 1,
        )
        return left_vectors[:, :component_count].contiguous()

    @staticmethod
    def debias_features(
        features: torch.Tensor,
        positional_basis: torch.Tensor | None,
        num_components: int,
    ) -> torch.Tensor:
        """Project ``[C,H,W]`` features off the selected positional subspace."""

        if features.ndim != 3:
            raise ValueError("features must have shape [C, H, W].")
        if num_components < 0:
            raise ValueError("num_components must be non-negative.")
        if num_components == 0:
            return F.normalize(features.float(), dim=0)
        if positional_basis is None or positional_basis.shape[1] < num_components:
            raise ValueError("positional_basis does not contain the requested components.")

        channels, height, width = features.shape
        flat = features.float().reshape(channels, -1)
        basis = positional_basis[:, :num_components].to(flat)
        projected = flat - basis @ (basis.T @ flat)
        return F.normalize(projected.reshape(channels, height, width), dim=0)

    @staticmethod
    def resize_reference_mask(
        reference_mask: torch.Tensor,
        feature_size: tuple[int, int],
    ) -> torch.Tensor:
        """Nearest-neighbor resize a binary reference mask to patch resolution."""

        if reference_mask.ndim == 2:
            reference_mask = reference_mask[None, None]
        elif reference_mask.ndim == 3:
            reference_mask = reference_mask[:, None]
        else:
            raise ValueError("reference_mask must have shape [H,W] or [B,H,W].")
        resized = F.interpolate(reference_mask.float(), size=feature_size, mode="nearest")
        return resized[0, 0] > 0.5

    @torch.no_grad()
    def predict_from_features(
        self,
        reference_features: torch.Tensor,
        reference_mask: torch.Tensor,
        target_features: torch.Tensor,
        *,
        positional_basis: torch.Tensor | None = None,
        num_debias_components: int = 0,
        cluster_similarity_threshold: float = 0.6,
        merge_threshold: float = 0.2,
        cluster_labels: torch.Tensor | None = None,
    ) -> InContextSegmentationResult:
        """Apply INSID3-style matching and aggregation to one feature pair."""

        if reference_features.shape != target_features.shape:
            raise ValueError("reference and target feature maps must have identical shapes.")
        if reference_features.ndim != 3:
            raise ValueError("feature maps must have shape [C, H, W].")

        raw_reference = F.normalize(reference_features.float(), dim=0)
        raw_target = F.normalize(target_features.float(), dim=0)
        debiased_reference = self.debias_features(
            raw_reference, positional_basis, num_debias_components
        )
        debiased_target = self.debias_features(
            raw_target, positional_basis, num_debias_components
        )
        channels, height, width = raw_target.shape
        support = self.resize_reference_mask(reference_mask, (height, width))
        if not support.any():
            raise ValueError("reference mask contains no foreground patch.")

        reference_prototype = F.normalize(
            debiased_reference[:, support].mean(dim=1), dim=0
        )
        forward_similarity = torch.einsum(
            "chw,c->hw", debiased_target, reference_prototype
        )
        forward_mask = forward_similarity > 0
        if not forward_mask.any():
            forward_mask = forward_similarity >= torch.quantile(
                forward_similarity, 0.9
            )

        reference_flat = debiased_reference.flatten(1).T
        target_flat = debiased_target.flatten(1).T
        nearest_reference = (target_flat @ reference_flat.T).argmax(dim=1)
        backward_mask = support.flatten()[nearest_reference].reshape(height, width)
        candidate_mask = forward_mask & backward_mask

        raw_target_flat = raw_target.flatten(1).T
        if cluster_labels is None:
            cluster_labels = agglomerative_cluster_labels(
                raw_target_flat, cluster_similarity_threshold
            )
        cluster_labels = cluster_labels.to(raw_target.device).reshape(-1)
        num_clusters = int(cluster_labels.max().item()) + 1
        label_grid = cluster_labels.reshape(height, width)

        if not candidate_mask.any():
            empty = torch.zeros_like(candidate_mask)
            cluster_scores = raw_target.new_zeros(num_clusters)
            return InContextSegmentationResult(
                mask=empty,
                candidate_mask=candidate_mask,
                cluster_labels=label_grid,
                seed_cluster=-1,
                num_clusters=num_clusters,
                num_candidate_patches=0,
                cluster_scores=cluster_scores,
                score_map=cluster_scores[label_grid],
            )

        candidate_cluster_ids, candidate_counts = torch.unique(
            label_grid[candidate_mask], return_counts=True
        )
        debiased_cluster_prototypes = compute_cluster_prototypes(
            target_flat, cluster_labels
        )
        candidate_cross_similarity = (
            debiased_cluster_prototypes[candidate_cluster_ids]
            @ reference_prototype
        )
        seed_cluster = int(
            candidate_cluster_ids[candidate_cross_similarity.argmax()].item()
        )

        raw_cluster_prototypes = compute_cluster_prototypes(
            raw_target_flat, cluster_labels
        )
        intra_similarity = raw_cluster_prototypes @ raw_cluster_prototypes[seed_cluster]
        cross_similarity = debiased_cluster_prototypes @ reference_prototype

        cluster_areas = torch.bincount(
            cluster_labels, minlength=num_clusters
        ).to(raw_target.dtype)
        candidate_fraction = torch.zeros(
            num_clusters, device=raw_target.device, dtype=raw_target.dtype
        )
        candidate_fraction[candidate_cluster_ids] = (
            candidate_counts.to(raw_target.dtype)
            / cluster_areas[candidate_cluster_ids].clamp_min(1)
        )
        candidate_fraction[seed_cluster] = 1.0
        scores = cross_similarity * intra_similarity * candidate_fraction

        selected = scores > merge_threshold
        selected[seed_cluster] = True
        mask = selected[label_grid] & torch.isin(label_grid, candidate_cluster_ids)
        return InContextSegmentationResult(
            mask=mask,
            candidate_mask=candidate_mask,
            cluster_labels=label_grid,
            seed_cluster=seed_cluster,
            num_clusters=num_clusters,
            num_candidate_patches=int(candidate_mask.sum().item()),
            cluster_scores=scores,
            score_map=scores[label_grid],
        )
