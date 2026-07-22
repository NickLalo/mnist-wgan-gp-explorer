"""Quality-aware sampling for representative generator grids."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from mnist_wgan.module import ConditionalWGAN

DEFAULT_QUALITY_OVERSAMPLE = 1.20
QUALITY_REJECTION_THRESHOLD = 0.15
DETACHED_INK_THRESHOLD = 0.10
UNSUPPORTED_INK_THRESHOLD = 0.03
# Per-digit 99th percentiles from 800 deterministic training-set references.
# Keeping the two coordinates separate matches the evaluation definition: a
# sample is an excessive shade outlier when either centerline statistic crosses
# the real-data envelope.
STROKE_SHADE_CV_THRESHOLDS = (
    0.18009305,
    0.23450953,
    0.17322524,
    0.18218537,
    0.17852868,
    0.18906794,
    0.19157907,
    0.18149434,
    0.18615095,
    0.18855990,
)
STROKE_SHADE_DIP_THRESHOLDS = (
    0.06730614,
    0.08496279,
    0.06908337,
    0.07211228,
    0.07053263,
    0.07226840,
    0.06812958,
    0.07176431,
    0.06746972,
    0.07281310,
)
# Reject only the most severe shade discontinuities. Milder tail samples still
# participate in rank scoring, which avoids sacrificing normal handwriting
# variation merely to optimize one diagnostic.
STROKE_SHADE_REJECTION_MULTIPLIER = 1.50


def candidate_count(requested: int, oversample: float = DEFAULT_QUALITY_OVERSAMPLE) -> int:
    """Return a modest candidate pool with useful headroom for small grids."""
    if requested < 1:
        raise ValueError("requested must be positive")
    if oversample < 1.0:
        raise ValueError("oversample must be at least 1")
    return max(requested, requested + 4, math.ceil(requested * oversample))


def _quality_rank(values: Tensor, *, higher_is_better: bool = True) -> Tensor:
    """Convert arbitrary scores to stable [0, 1] ranks within one digit class."""
    order = values.argsort(descending=higher_is_better, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.linspace(1.0, 0.0, len(values), device=values.device)
    return ranks


@torch.inference_mode()
def select_quality_samples(
    model: ConditionalWGAN,
    images: Tensor,
    labels: Tensor,
    keep_per_class: int,
    rejection_threshold: float = QUALITY_REJECTION_THRESHOLD,
    batch_size: int = 512,
) -> tuple[Tensor, Tensor]:
    """Replace only clear failures with higher-quality backup candidates.

    The score combines requested-class margin, conditional critic score, soft
    stroke-profile centrality, unsupported ink, and disconnected ink. The first
    ``keep_per_class`` samples remain untouched unless they fall below an
    absolute low-quality threshold, are classified as the wrong digit, or have
    a small detached component with little local support. The explicit artifact
    check catches isolated flecks that can otherwise hide beside an excellent
    main stroke. This avoids the diversity loss caused by retaining a fixed top
    percentage.
    """
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    if keep_per_class < 1:
        raise ValueError("keep_per_class must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    device = next(model.parameters()).device
    images = images.to(device)
    labels = labels.to(device)
    critic_batches = []
    unsupported_batches = []
    disconnected_batches = []
    profile_batches = []
    shade_batches = []
    logit_batches = []
    if model.perceptual_encoder is not None:
        model.perceptual_encoder.eval()

    # One-digit requests can contain thousands of images. Keep scoring memory
    # bounded just as generation is, rather than forwarding the whole pool
    # through the critic and classifier at once.
    for start in range(0, len(images), batch_size):
        stop = start + batch_size
        image_batch = images[start:stop]
        label_batch = labels[start:stop]
        critic_batches.append(model.critic(image_batch, label_batch))
        _, unsupported, _, disconnected = model.stroke_loss._statistics(image_batch)
        unsupported_batches.append(unsupported)
        disconnected_batches.append(disconnected)
        profile_batches.append(model.stroke_profile_loss._statistics(image_batch))
        shade_batches.append(model.stroke_shade_loss._statistics(image_batch))
        if model.perceptual_encoder is not None:
            logit_batches.append(model.perceptual_encoder(image_batch))

    critic_scores = torch.cat(critic_batches)
    unsupported = torch.cat(unsupported_batches)
    disconnected = torch.cat(disconnected_batches)
    profiles = torch.cat(profile_batches)
    shade = torch.cat(shade_batches)
    logits = torch.cat(logit_batches) if logit_batches else None
    predicted = None
    if logits is not None:
        predicted = logits.argmax(dim=1)

    selected = []
    selected_scores = []
    for digit in labels.unique(sorted=True):
        indices = torch.where(labels == digit)[0]
        if len(indices) < keep_per_class:
            raise ValueError(f"digit {int(digit)} has fewer candidates than requested")

        score = 0.275 * _quality_rank(critic_scores[indices])
        if logits is not None:
            class_logits = logits[indices]
            requested_logits = class_logits[:, digit]
            alternatives = class_logits.clone()
            alternatives[:, digit] = -torch.inf
            margin = requested_logits - alternatives.max(dim=1).values
            score = score + 0.425 * _quality_rank(margin)
        else:
            score = score + 0.425 * _quality_rank(critic_scores[indices])

        class_profiles = profiles[indices]
        median = class_profiles.median(dim=0).values
        deviation = (class_profiles - median).abs()
        scale = deviation.median(dim=0).values.clamp_min(1e-3)
        profile_outlier = (deviation / scale).mean(dim=1)
        score = score + 0.10 * _quality_rank(profile_outlier, higher_is_better=False)
        score = score + 0.075 * _quality_rank(unsupported[indices], higher_is_better=False)
        score = score + 0.075 * _quality_rank(disconnected[indices], higher_is_better=False)
        shade_level = shade[indices].sum(dim=1)
        score = score + 0.05 * _quality_rank(shade_level, higher_is_better=False)

        base = indices[:keep_per_class]
        extras = indices[keep_per_class:]
        base_bad = score[:keep_per_class] < rejection_threshold
        if predicted is not None:
            base_bad |= predicted[base] != digit
        base_bad |= (
            (disconnected[base] > DETACHED_INK_THRESHOLD)
            & (unsupported[base] > UNSUPPORTED_INK_THRESHOLD)
        )
        shade_cv_threshold = (
            STROKE_SHADE_CV_THRESHOLDS[int(digit)] * STROKE_SHADE_REJECTION_MULTIPLIER
        )
        shade_dip_threshold = (
            STROKE_SHADE_DIP_THRESHOLDS[int(digit)] * STROKE_SHADE_REJECTION_MULTIPLIER
        )
        base_bad |= (shade[base, 0] > shade_cv_threshold) | (
            shade[base, 1] > shade_dip_threshold
        )
        bad_local = torch.where(base_bad)[0]
        replace_count = min(len(bad_local), len(extras))
        if replace_count:
            rejected_local = bad_local[
                score[bad_local].argsort(stable=True)[:replace_count]
            ]
            base_keep = torch.ones(keep_per_class, dtype=torch.bool, device=device)
            base_keep[rejected_local] = False
            replacement_local = (
                score[keep_per_class:].topk(replace_count).indices + keep_per_class
            )
            retained_local = torch.cat((torch.where(base_keep)[0], replacement_local))
            retained = indices[retained_local]
        else:
            retained_local = torch.arange(keep_per_class, device=device)
            retained = base
        selected.append(retained)
        selected_scores.append(score[retained_local])

    selected_indices = torch.cat(selected)
    return images[selected_indices], torch.cat(selected_scores)
