"""MNIST-specific fidelity, conditioning, and diversity evaluation."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial.distance import cdist
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from mnist_wgan.classifier import MNISTClassifier, classifier_transform, ensure_classifier
from mnist_wgan.losses import stroke_shade_statistics
from mnist_wgan.module import ConditionalWGAN
from mnist_wgan.paths import default_checkpoint_path
from mnist_wgan.sampling import (
    DETACHED_INK_THRESHOLD,
    QUALITY_REJECTION_THRESHOLD,
    STROKE_HALO_THRESHOLDS,
    UNSUPPORTED_INK_THRESHOLD,
    candidate_count,
    select_quality_samples,
)
from mnist_wgan.visualize import image_grid, save_image, seeded_noise


@torch.inference_mode()
def _embed(
    classifier: MNISTClassifier,
    images: Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, predictions, confidences = [], [], []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        logits, batch_features = classifier(batch, return_features=True)
        probabilities = logits.softmax(dim=1)
        confidence, prediction = probabilities.max(dim=1)
        features.append(F.normalize(batch_features.float(), dim=1).cpu().numpy())
        predictions.append(prediction.cpu().numpy())
        confidences.append(confidence.float().cpu().numpy())
    return (
        np.concatenate(features),
        np.concatenate(predictions),
        np.concatenate(confidences),
    )


def _collect_by_class(dataset, samples_per_digit: int) -> tuple[Tensor, Tensor]:
    buckets: dict[int, list[Tensor]] = defaultdict(list)
    for images, labels in DataLoader(dataset, batch_size=512, num_workers=8):
        for digit in range(10):
            needed = samples_per_digit - len(buckets[digit])
            if needed > 0:
                selected = images[labels == digit][:needed]
                buckets[digit].extend(selected.unbind())
        if all(len(buckets[digit]) >= samples_per_digit for digit in range(10)):
            break
    images = torch.stack([image for digit in range(10) for image in buckets[digit]])
    labels = torch.arange(10).repeat_interleave(samples_per_digit)
    return images, labels


def frechet_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Stable Gaussian Fréchet distance using the symmetric Bures formulation."""
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    mean_first, mean_second = first.mean(axis=0), second.mean(axis=0)
    cov_first = np.cov(first, rowvar=False) + np.eye(first.shape[1]) * 1e-6
    cov_second = np.cov(second, rowvar=False) + np.eye(second.shape[1]) * 1e-6
    values, vectors = np.linalg.eigh(cov_first)
    sqrt_first = (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T
    middle = sqrt_first @ cov_second @ sqrt_first
    trace_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh(middle), 0.0, None)).sum()
    distance = (
        np.square(mean_first - mean_second).sum()
        + np.trace(cov_first)
        + np.trace(cov_second)
        - 2.0 * trace_sqrt
    )
    return float(max(distance, 0.0))


def precision_recall_density_coverage(
    real: np.ndarray, generated: np.ndarray, nearest_k: int = 5
) -> dict[str, float]:
    """PRDC manifold metrics from independent classifier embeddings."""
    real_to_real = cdist(real, real)
    generated_to_generated = cdist(generated, generated)
    np.fill_diagonal(real_to_real, np.inf)
    np.fill_diagonal(generated_to_generated, np.inf)
    real_radius = np.partition(real_to_real, nearest_k - 1, axis=1)[:, nearest_k - 1]
    generated_radius = np.partition(generated_to_generated, nearest_k - 1, axis=1)[:, nearest_k - 1]
    distances = cdist(real, generated)
    precision = (distances <= real_radius[:, None]).any(axis=0).mean()
    recall = (distances <= generated_radius[None, :]).any(axis=1).mean()
    density = (distances <= real_radius[:, None]).sum(axis=0).mean() / nearest_k
    coverage = (distances.min(axis=1) <= real_radius).mean()
    # A value of 1 is the real-manifold boundary used by precision. Reporting
    # the continuous margin exposes how severe the rare failures are instead of
    # treating every outlier as equally bad.
    precision_margins = (distances / np.maximum(real_radius[:, None], 1e-8)).min(axis=0)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(coverage),
        "precision_margin_p99": float(np.quantile(precision_margins, 0.99)),
        "manifold_outlier_rate_1_1": float((precision_margins > 1.1).mean()),
    }


def _distribution_metrics(
    real_features: np.ndarray,
    comparison_features: np.ndarray,
    samples_per_digit: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    per_digit = []
    for digit in range(10):
        section = slice(digit * samples_per_digit, (digit + 1) * samples_per_digit)
        real, comparison = real_features[section], comparison_features[section]
        digit_metrics = precision_recall_density_coverage(real, comparison)
        digit_metrics["fmd"] = frechet_distance(real, comparison)
        per_digit.append(digit_metrics)
    result = {
        name: float(np.mean([item[name] for item in per_digit]))
        for name in (
            "fmd",
            "precision",
            "recall",
            "density",
            "coverage",
            "precision_margin_p99",
            "manifold_outlier_rate_1_1",
        )
    }
    result["worst_digit_precision"] = float(min(item["precision"] for item in per_digit))
    return result, per_digit


def _hard_skeleton(foreground: np.ndarray) -> np.ndarray:
    """Morphologically thin a batch without connecting neighboring samples."""
    structure = np.zeros((1, 3, 3), dtype=bool)
    structure[0, 1, :] = True
    structure[0, :, 1] = True
    working = foreground.copy()
    skeleton = np.zeros_like(working)
    while working.any():
        eroded = ndimage.binary_erosion(working, structure=structure)
        opened = ndimage.binary_dilation(eroded, structure=structure)
        skeleton |= working & ~opened
        working = eroded
    return skeleton


def _stroke_profiles(pixels: np.ndarray, threshold: float) -> np.ndarray:
    foreground = pixels >= threshold
    skeleton = _hard_skeleton(foreground)
    foreground_mass = foreground.sum(axis=(1, 2)).clip(min=1)
    skeleton_mass = skeleton.sum(axis=(1, 2)).clip(min=1)
    width = foreground_mass / skeleton_mass
    strength = (pixels * foreground).sum(axis=(1, 2)) / foreground_mass
    strong_fraction = (pixels >= 0.55).sum(axis=(1, 2)) / foreground_mass
    return np.stack((width, strength, strong_fraction, skeleton_mass), axis=1)


def stroke_integrity_metrics(
    reference_images: Tensor,
    comparison_images: Tensor,
    samples_per_digit: int,
    *,
    threshold: float = 0.20,
) -> dict[str, float | dict[str, float]]:
    """Measure detached ink and class-wise foreground-mass calibration.

    A sample is counted as fragmented when more than 2% of thresholded ink lies
    outside its largest 8-connected component. This directly captures detached
    dots and stroke fragments visible in generated MNIST samples.
    """

    def summarize(images: Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pixels = images.detach().float().add(1).mul(0.5).clamp(0, 1).cpu().numpy()[:, 0]
        fragmented = np.zeros(len(pixels), dtype=np.float64)
        ink_mass = pixels.sum(axis=(1, 2), dtype=np.float64)
        unsupported_ink = np.zeros(len(pixels), dtype=np.float64)
        structure = np.ones((3, 3), dtype=np.int8)
        for index, image in enumerate(pixels):
            local_ink = ndimage.uniform_filter(image, size=3, mode="constant")
            unsupported_ink[index] = (image * np.maximum(threshold - local_ink, 0.0)).sum()
            components, _ = ndimage.label(image >= threshold, structure=structure)
            sizes = np.bincount(components.ravel())[1:]
            if sizes.size and (sizes.sum() - sizes.max()) / max(sizes.sum(), 1) > 0.02:
                fragmented[index] = 1.0
        return fragmented, ink_mass, unsupported_ink, _stroke_profiles(pixels, threshold)

    _, reference_ink, _, reference_profiles = summarize(reference_images)
    comparison_fragmented, comparison_ink, comparison_unsupported, comparison_profiles = summarize(
        comparison_images
    )
    artifact_by_digit: dict[str, float] = {}
    ink_error_by_digit: dict[str, float] = {}
    unsupported_by_digit: dict[str, float] = {}
    profile_distance_by_digit: dict[str, float] = {}
    profile_tail_by_digit: dict[str, float] = {}
    blob_by_digit: dict[str, float] = {}
    thin_by_digit: dict[str, float] = {}
    weak_by_digit: dict[str, float] = {}
    for digit in range(10):
        section = slice(digit * samples_per_digit, (digit + 1) * samples_per_digit)
        artifact_by_digit[str(digit)] = float(comparison_fragmented[section].mean())
        reference_mass = reference_ink[section].mean()
        ink_error_by_digit[str(digit)] = float(
            abs(comparison_ink[section].mean() - reference_mass) / max(reference_mass, 1e-8)
        )
        unsupported_by_digit[str(digit)] = float(comparison_unsupported[section].mean())
        reference_profile = reference_profiles[section]
        comparison_profile = comparison_profiles[section]
        lower = np.quantile(reference_profile, 0.01, axis=0)
        upper = np.quantile(reference_profile, 0.99, axis=0)
        profile_tail_by_digit[str(digit)] = float(
            ((comparison_profile < lower) | (comparison_profile > upper)).any(axis=1).mean()
        )
        blob_by_digit[str(digit)] = float((comparison_profile[:, 0] > upper[0]).mean())
        thin_by_digit[str(digit)] = float((comparison_profile[:, 0] < lower[0]).mean())
        weak_by_digit[str(digit)] = float(
            (
                (comparison_profile[:, 1] < lower[1])
                | (comparison_profile[:, 2] < lower[2])
            ).mean()
        )
        quantiles = np.linspace(0.01, 0.99, 99)
        reference_quantiles = np.quantile(reference_profile, quantiles, axis=0)
        comparison_quantiles = np.quantile(comparison_profile, quantiles, axis=0)
        robust_scale = (
            np.quantile(reference_profile, 0.75, axis=0)
            - np.quantile(reference_profile, 0.25, axis=0)
        ).clip(min=1e-3)
        profile_distance_by_digit[str(digit)] = float(
            (np.abs(comparison_quantiles - reference_quantiles) / robust_scale).mean()
        )
    return {
        "fragmented_sample_rate": float(comparison_fragmented.mean()),
        "ink_mass_relative_error": float(np.mean(list(ink_error_by_digit.values()))),
        "unsupported_ink": float(comparison_unsupported.mean()),
        "stroke_profile_distance": float(np.mean(list(profile_distance_by_digit.values()))),
        "stroke_profile_tail_rate": float(np.mean(list(profile_tail_by_digit.values()))),
        "blob_stroke_rate": float(np.mean(list(blob_by_digit.values()))),
        "thin_stroke_rate": float(np.mean(list(thin_by_digit.values()))),
        "weak_stroke_rate": float(np.mean(list(weak_by_digit.values()))),
        "fragmented_sample_rate_by_digit": artifact_by_digit,
        "ink_mass_relative_error_by_digit": ink_error_by_digit,
        "unsupported_ink_by_digit": unsupported_by_digit,
        "stroke_profile_distance_by_digit": profile_distance_by_digit,
        "stroke_profile_tail_rate_by_digit": profile_tail_by_digit,
        "blob_stroke_rate_by_digit": blob_by_digit,
        "thin_stroke_rate_by_digit": thin_by_digit,
        "weak_stroke_rate_by_digit": weak_by_digit,
    }


def stroke_shade_metrics(
    reference_images: Tensor,
    comparison_images: Tensor,
    samples_per_digit: int,
    *,
    batch_size: int = 512,
) -> dict[str, float | dict[str, float]]:
    """Compare low-frequency centerline shade variation with real MNIST."""

    def summarize(images: Tensor) -> np.ndarray:
        batches = [
            stroke_shade_statistics(images[start : start + batch_size]).detach().cpu()
            for start in range(0, len(images), batch_size)
        ]
        return torch.cat(batches).numpy()

    reference = summarize(reference_images)
    comparison = summarize(comparison_images)
    quantiles = np.linspace(0.01, 0.99, 99)
    distance_by_digit: dict[str, float] = {}
    tail_by_digit: dict[str, float] = {}
    for digit in range(10):
        section = slice(digit * samples_per_digit, (digit + 1) * samples_per_digit)
        real_class, comparison_class = reference[section], comparison[section]
        robust_scale = (
            np.quantile(real_class, 0.75, axis=0)
            - np.quantile(real_class, 0.25, axis=0)
        ).clip(min=1e-3)
        distance_by_digit[str(digit)] = float(
            (
                np.abs(
                    np.quantile(comparison_class, quantiles, axis=0)
                    - np.quantile(real_class, quantiles, axis=0)
                )
                / robust_scale
            ).mean()
        )
        upper = np.quantile(real_class, 0.99, axis=0)
        tail_by_digit[str(digit)] = float((comparison_class > upper).any(axis=1).mean())
    return {
        "stroke_shade_distance": float(np.mean(list(distance_by_digit.values()))),
        "stroke_shade_tail_rate": float(np.mean(list(tail_by_digit.values()))),
        "stroke_shade_cv": float(comparison[:, 0].mean()),
        "stroke_shade_dip": float(comparison[:, 1].mean()),
        "stroke_shade_distance_by_digit": distance_by_digit,
        "stroke_shade_tail_rate_by_digit": tail_by_digit,
    }


def stroke_halo_metrics(
    reference_images: Tensor,
    comparison_images: Tensor,
    samples_per_digit: int,
    *,
    low_threshold: float = 0.05,
    core_threshold: float = 0.55,
) -> dict[str, float | dict[str, float]]:
    """Measure pale outer-ring ink beyond normal one-pixel antialiasing."""

    def summarize(images: Tensor) -> np.ndarray:
        ink = images.detach().float().add(1).mul(0.5).clamp(0, 1)
        core = (ink >= core_threshold).float()
        immediate = F.max_pool2d(core, kernel_size=3, stride=1, padding=1) > 0
        extended = F.max_pool2d(core, kernel_size=5, stride=1, padding=2) > 0
        soft = (ink >= low_threshold) & (ink < core_threshold)
        halo_mass = (ink * soft * (extended & ~immediate)).flatten(1).sum(dim=1)
        total_ink = ink.flatten(1).sum(dim=1).clamp_min(1)
        return (halo_mass / total_ink).cpu().numpy()

    reference = summarize(reference_images)
    comparison = summarize(comparison_images)
    quantiles = np.linspace(0.01, 0.99, 99)
    distance_by_digit: dict[str, float] = {}
    tail_by_digit: dict[str, float] = {}
    rate_by_digit: dict[str, float] = {}
    severe_by_digit: dict[str, float] = {}
    for digit in range(10):
        section = slice(digit * samples_per_digit, (digit + 1) * samples_per_digit)
        real_class, comparison_class = reference[section], comparison[section]
        scale = max(float(np.quantile(real_class, 0.99)), 1e-3)
        distance_by_digit[str(digit)] = float(
            np.abs(
                np.quantile(comparison_class, quantiles)
                - np.quantile(real_class, quantiles)
            ).mean()
            / scale
        )
        upper = float(np.quantile(real_class, 0.99))
        tail_by_digit[str(digit)] = float((comparison_class > upper).mean())
        rate_by_digit[str(digit)] = float((comparison_class > 0.001).mean())
        severe_by_digit[str(digit)] = float((comparison_class > 0.005).mean())
    return {
        "stroke_halo_distance": float(np.mean(list(distance_by_digit.values()))),
        "stroke_halo_tail_rate": float(np.mean(list(tail_by_digit.values()))),
        "stroke_halo_rate": float((comparison > 0.001).mean()),
        "stroke_halo_severe_rate": float((comparison > 0.005).mean()),
        "stroke_halo_mean": float(comparison.mean()),
        "stroke_halo_distance_by_digit": distance_by_digit,
        "stroke_halo_tail_rate_by_digit": tail_by_digit,
        "stroke_halo_rate_by_digit": rate_by_digit,
        "stroke_halo_severe_rate_by_digit": severe_by_digit,
    }


def _worst_tail_indices(
    real_features: np.ndarray,
    comparison_features: np.ndarray,
    samples_per_digit: int,
    count: int = 10,
) -> np.ndarray:
    """Return the worst class-conditioned manifold outliers for visual audit."""
    selected = []
    for digit in range(10):
        start = digit * samples_per_digit
        section = slice(start, start + samples_per_digit)
        real = real_features[section]
        comparison = comparison_features[section]
        real_to_real = cdist(real, real)
        np.fill_diagonal(real_to_real, np.inf)
        real_radius = np.partition(real_to_real, 4, axis=1)[:, 4]
        margins = (cdist(real, comparison) / np.maximum(real_radius[:, None], 1e-8)).min(axis=0)
        worst = np.argsort(margins)[-min(count, samples_per_digit) :][::-1]
        selected.extend((start + worst).tolist())
    return np.asarray(selected, dtype=np.int64)


def _score(
    generated: dict[str, float], calibration: dict[str, float], real_accuracy: float
) -> tuple[float, dict[str, float]]:
    label = min(generated["conditional_accuracy"] / max(real_accuracy, 1e-8), 1.0) * 100
    precision = min(generated["precision"] / max(calibration["precision"], 1e-8), 1.0) * 100
    # Coverage is the scored diversity statistic: it directly asks what fraction
    # of the real manifold lies near at least one generated sample. PRDC recall is
    # retained in the report as a useful but radius-sensitive diagnostic.
    coverage = min(generated["coverage"] / max(calibration["coverage"], 1e-8), 1.0) * 100
    fmd_excess = max(generated["fmd"] - calibration["fmd"], 0.0)
    distribution = 100 * math.exp(-fmd_excess / 0.15)
    artifact_excess = max(
        generated["fragmented_sample_rate"] - calibration["fragmented_sample_rate"], 0.0
    )
    stroke_integrity = 100 * math.exp(-artifact_excess / 0.07)
    ink_calibration = 100 * math.exp(-generated["ink_mass_relative_error"] / 0.20)
    components = {
        "conditional_accuracy": label,
        "manifold_precision": precision,
        "manifold_coverage": coverage,
        "frechet_distribution": distribution,
        "stroke_integrity": stroke_integrity,
        "ink_calibration": ink_calibration,
    }
    quality = (
        0.20 * label
        + 0.20 * precision
        + 0.15 * coverage
        + 0.15 * distribution
        + 0.20 * stroke_integrity
        + 0.10 * ink_calibration
    )
    return quality, components


def evaluate_checkpoint(
    checkpoint_path: str | Path | None = None,
    *,
    data_dir: str | Path = "data",
    output_path: str | Path = "artifacts/evaluation.json",
    samples_per_digit: int = 800,
    seed: int = 112,
    device: str | None = None,
    use_ema: bool = True,
    style_strength: float | None = None,
    quality_oversample: float = 1.0,
) -> dict:
    """Evaluate a checkpoint against real-data calibration and write a JSON report."""
    checkpoint_path = Path(checkpoint_path or default_checkpoint_path())
    output_path = Path(output_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if quality_oversample < 1.0:
        raise ValueError("quality_oversample must be at least 1")
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    classifier, real_accuracy = ensure_classifier(data_dir, device=torch_device)
    model = ConditionalWGAN.load_from_checkpoint(checkpoint_path, map_location=torch_device)
    model.to(torch_device).eval()
    if style_strength is not None:
        model.generator.style_strength = style_strength
        model.generator_ema.style_strength = style_strength

    train_set = datasets.MNIST(
        data_dir, train=True, download=True, transform=classifier_transform()
    )
    test_set = datasets.MNIST(
        data_dir, train=False, download=True, transform=classifier_transform()
    )
    reference_images, labels = _collect_by_class(train_set, samples_per_digit)
    calibration_images, _ = _collect_by_class(test_set, samples_per_digit)
    reference_features, _, _ = _embed(classifier, reference_images, torch_device)
    calibration_features, _, _ = _embed(classifier, calibration_images, torch_device)

    generated_batches = []
    with torch.inference_mode():
        for digit in range(10):
            pool_size = (
                candidate_count(samples_per_digit, quality_oversample)
                if quality_oversample > 1.0
                else samples_per_digit
            )
            digit_labels = torch.full((pool_size,), digit, device=torch_device)
            noise = seeded_noise(
                pool_size,
                model.hparams.latent_dim,
                seed + digit,
                torch_device,
            )
            digit_batches = [
                model(
                    noise[start : start + 512],
                    digit_labels[start : start + 512],
                    use_ema=use_ema,
                )
                for start in range(0, pool_size, 512)
            ]
            digit_images = torch.cat(digit_batches)
            if quality_oversample > 1.0:
                digit_images, _ = select_quality_samples(
                    model,
                    digit_images,
                    digit_labels,
                    samples_per_digit,
                )
            generated_batches.append(digit_images[:samples_per_digit].cpu())
    generated_images = torch.cat(generated_batches)
    generated_features, predictions, confidences = _embed(
        classifier, generated_images, torch_device
    )

    calibration, _ = _distribution_metrics(
        reference_features, calibration_features, samples_per_digit
    )
    generated, generated_by_digit = _distribution_metrics(
        reference_features, generated_features, samples_per_digit
    )
    calibration_strokes = stroke_integrity_metrics(
        reference_images, calibration_images, samples_per_digit
    )
    generated_strokes = stroke_integrity_metrics(
        reference_images, generated_images, samples_per_digit
    )
    calibration_shade = stroke_shade_metrics(
        reference_images, calibration_images, samples_per_digit
    )
    generated_shade = stroke_shade_metrics(reference_images, generated_images, samples_per_digit)
    calibration_halo = stroke_halo_metrics(
        reference_images, calibration_images, samples_per_digit
    )
    generated_halo = stroke_halo_metrics(reference_images, generated_images, samples_per_digit)
    calibration.update(
        {name: value for name, value in calibration_strokes.items() if not isinstance(value, dict)}
    )
    generated.update(
        {name: value for name, value in generated_strokes.items() if not isinstance(value, dict)}
    )
    calibration.update(
        {name: value for name, value in calibration_shade.items() if not isinstance(value, dict)}
    )
    generated.update(
        {name: value for name, value in generated_shade.items() if not isinstance(value, dict)}
    )
    calibration.update(
        {name: value for name, value in calibration_halo.items() if not isinstance(value, dict)}
    )
    generated.update(
        {name: value for name, value in generated_halo.items() if not isinstance(value, dict)}
    )
    expected = labels.numpy()
    generated["conditional_accuracy"] = float((predictions == expected).mean())
    generated["classifier_confidence"] = float(confidences.mean())
    per_digit_accuracy = {
        str(digit): float(
            (
                predictions[digit * samples_per_digit : (digit + 1) * samples_per_digit] == digit
            ).mean()
        )
        for digit in range(10)
    }
    quality, components = _score(generated, calibration, real_accuracy)
    report = {
        "quality_score": round(quality, 2),
        "score_weights": {
            "conditional_accuracy": 0.20,
            "manifold_precision": 0.20,
            "manifold_coverage": 0.15,
            "frechet_distribution": 0.15,
            "stroke_integrity": 0.20,
            "ink_calibration": 0.10,
        },
        "component_scores": {name: round(value, 2) for name, value in components.items()},
        "metrics": {name: round(value, 6) for name, value in generated.items()},
        "real_data_calibration": {
            "classifier_accuracy": round(real_accuracy, 6),
            **{name: round(value, 6) for name, value in calibration.items()},
        },
        "conditional_accuracy_by_digit": per_digit_accuracy,
        "precision_by_digit": {
            str(digit): generated_by_digit[digit]["precision"] for digit in range(10)
        },
        "fragmented_sample_rate_by_digit": generated_strokes["fragmented_sample_rate_by_digit"],
        "ink_mass_relative_error_by_digit": generated_strokes["ink_mass_relative_error_by_digit"],
        "unsupported_ink_by_digit": generated_strokes["unsupported_ink_by_digit"],
        "stroke_profile_distance_by_digit": generated_strokes[
            "stroke_profile_distance_by_digit"
        ],
        "stroke_profile_tail_rate_by_digit": generated_strokes[
            "stroke_profile_tail_rate_by_digit"
        ],
        "blob_stroke_rate_by_digit": generated_strokes["blob_stroke_rate_by_digit"],
        "thin_stroke_rate_by_digit": generated_strokes["thin_stroke_rate_by_digit"],
        "weak_stroke_rate_by_digit": generated_strokes["weak_stroke_rate_by_digit"],
        "stroke_shade_distance_by_digit": generated_shade[
            "stroke_shade_distance_by_digit"
        ],
        "stroke_shade_tail_rate_by_digit": generated_shade[
            "stroke_shade_tail_rate_by_digit"
        ],
        "stroke_halo_distance_by_digit": generated_halo["stroke_halo_distance_by_digit"],
        "stroke_halo_tail_rate_by_digit": generated_halo[
            "stroke_halo_tail_rate_by_digit"
        ],
        "stroke_halo_rate_by_digit": generated_halo["stroke_halo_rate_by_digit"],
        "stroke_halo_severe_rate_by_digit": generated_halo[
            "stroke_halo_severe_rate_by_digit"
        ],
        "samples_per_digit": samples_per_digit,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "generator_weights": "ema" if use_ema else "raw",
        "structured_style_strength": model.generator.style_strength,
        "quality_oversample": quality_oversample,
        "quality_gate": (
            {
                "score_threshold": QUALITY_REJECTION_THRESHOLD,
                "detached_ink_threshold": DETACHED_INK_THRESHOLD,
                "unsupported_ink_threshold": UNSUPPORTED_INK_THRESHOLD,
                "stroke_halo_thresholds": STROKE_HALO_THRESHOLDS,
            }
            if quality_oversample > 1.0
            else None
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    preview = image_grid(
        generated_images.reshape(10, samples_per_digit, 1, 28, 28)[:, :10].reshape(-1, 1, 28, 28),
        rows=10,
        columns=10,
        row_labels=[str(i) for i in range(10)],
        scale=2,
    )
    save_image(preview, output_path.with_name("evaluation_grid.png"))
    tail_indices = _worst_tail_indices(
        reference_features, generated_features, samples_per_digit, count=10
    )
    tail_preview = image_grid(
        generated_images[tail_indices],
        rows=10,
        columns=10,
        row_labels=[str(i) for i in range(10)],
        scale=2,
    )
    save_image(tail_preview, output_path.with_name("evaluation_tail_grid.png"))
    return report
