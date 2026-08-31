import torch

from rfdetr.models.insid3 import (
    DinoV3InContextSegmenter,
    agglomerative_cluster_labels,
    compute_cluster_prototypes,
)


def test_compute_cluster_prototypes_returns_normalized_means():
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 1])

    prototypes = compute_cluster_prototypes(features, labels)

    torch.testing.assert_close(prototypes, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


def test_agglomerative_clustering_separates_orthogonal_features():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ]
    )

    labels = agglomerative_cluster_labels(features, similarity_threshold=0.8)

    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_debias_features_removes_selected_subspace():
    features = torch.tensor(
        [
            [[1.0, 1.0]],
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ]
    )
    basis = torch.tensor([[1.0], [0.0], [0.0]])

    projected = DinoV3InContextSegmenter.debias_features(features, basis, 1)

    torch.testing.assert_close(projected[0], torch.zeros_like(projected[0]))
    torch.testing.assert_close(projected[:, 0, 0], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(projected[:, 0, 1], torch.tensor([0.0, 0.0, 1.0]))


def test_predict_from_features_recovers_matching_cluster():
    reference = torch.zeros(4, 2, 2)
    target = torch.zeros(4, 2, 2)
    reference[0, 0] = 1.0
    reference[1, 1] = 1.0
    target.copy_(reference)
    reference_mask = torch.tensor([[1, 1], [0, 0]], dtype=torch.bool)
    cluster_labels = torch.tensor([0, 0, 1, 1])
    segmenter = DinoV3InContextSegmenter(torch.nn.Identity())

    result = segmenter.predict_from_features(
        reference,
        reference_mask,
        target,
        cluster_labels=cluster_labels,
    )

    expected = torch.tensor([[1, 1], [0, 0]], dtype=torch.bool)
    torch.testing.assert_close(result.mask, expected)
    torch.testing.assert_close(result.candidate_mask, expected)
    assert result.seed_cluster == 0
    assert result.num_clusters == 2
    assert result.num_candidate_patches == 2
    assert result.cluster_scores.shape == (2,)
    assert result.score_map.shape == (2, 2)
    assert torch.isfinite(result.score_map).all()
