import numpy as np
import torch

from mnist_wgan.metrics import (
    frechet_distance,
    precision_recall_density_coverage,
    stroke_integrity_metrics,
)


def test_identical_feature_distributions_are_near_perfect():
    rng = np.random.default_rng(112)
    features = rng.normal(size=(80, 12))
    assert frechet_distance(features, features) < 1e-8
    metrics = precision_recall_density_coverage(features, features)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["coverage"] == 1.0


def test_stroke_integrity_detects_a_detached_component():
    reference = torch.full((10, 1, 28, 28), -1.0)
    reference[:, :, 8:20, 12:16] = 1.0
    comparison = reference.clone()
    comparison[0, :, 2:4, 2:4] = 1.0
    metrics = stroke_integrity_metrics(reference, comparison, samples_per_digit=1)
    assert metrics["fragmented_sample_rate"] == 0.1
    assert metrics["fragmented_sample_rate_by_digit"]["0"] == 1.0
    assert metrics["stroke_profile_tail_rate"] > 0


def test_stroke_profiles_detect_blobs_and_weak_lines():
    reference = torch.full((10, 1, 28, 28), -1.0)
    reference[:, :, 5:23, 13:16] = 1.0
    comparison = reference.clone()
    comparison[0, :, 5:23, 9:20] = 1.0
    comparison[1, :, 5:23, 13:16] = -0.5
    metrics = stroke_integrity_metrics(reference, comparison, samples_per_digit=1)
    assert metrics["blob_stroke_rate_by_digit"]["0"] == 1.0
    assert metrics["weak_stroke_rate_by_digit"]["1"] == 1.0
