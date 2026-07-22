import torch

from mnist_wgan.losses import (
    BatchStructureLoss,
    StrokeIntegrityLoss,
    StrokeProfileLoss,
    StrokeShadeLoss,
    diversity_matching_loss,
    gradient_penalty,
    mode_seeking_loss,
    perceptual_tail_loss,
    sliced_feature_distance,
)
from mnist_wgan.models import ConditionalCritic, ConditionalGenerator
from mnist_wgan.module import ConditionalWGAN
from mnist_wgan.sampling import candidate_count, select_quality_samples


def test_model_shapes_and_gradients():
    generator = ConditionalGenerator(latent_dim=16, base_channels=8)
    critic = ConditionalCritic(base_channels=8)
    labels = torch.arange(4)
    noise = torch.randn(4, 16)
    generated = generator(noise, labels)
    assert generated.shape == (4, 1, 28, 28)
    assert generated.min() >= -1 and generated.max() <= 1
    scores = critic(generated, labels)
    assert scores.shape == (4,)
    scores.mean().backward()
    assert generator.project.weight.grad is not None


def test_regularizers_are_finite_and_differentiable():
    critic = ConditionalCritic(base_channels=8)
    real = torch.randn(4, 1, 28, 28).tanh()
    fake = torch.randn(4, 1, 28, 28, requires_grad=True).tanh()
    labels = torch.arange(4)
    penalty, norm = gradient_penalty(critic, real, fake, labels)
    histogram, edge, detail, variance = BatchStructureLoss()(real, fake, labels)
    ink, support, footprint, connectivity = StrokeIntegrityLoss()(real, fake, labels)
    stroke_profile = StrokeProfileLoss()(real, fake, labels)
    stroke_shade = StrokeShadeLoss()(real, fake, labels)
    second = torch.randn_like(fake).tanh()
    diversity = mode_seeking_loss(fake, second, torch.randn(4, 16), torch.randn(4, 16))
    diversity_match, _, _ = diversity_matching_loss(real, fake, second, labels)
    feature_distance = sliced_feature_distance(
        torch.randn(4, 12), torch.randn(4, 12, requires_grad=True), labels
    )
    tail_distance = perceptual_tail_loss(
        torch.randn(4, 12), torch.randn(4, 12, requires_grad=True), labels
    )
    total = (
        penalty
        + histogram
        + edge
        + detail
        + variance
        + feature_distance
        + tail_distance
        + ink
        + support
        + footprint
        + connectivity
        + stroke_profile
        + stroke_shade
        + diversity_match
        - 0.1 * diversity
    )
    assert all(
        torch.isfinite(value)
        for value in (
            penalty,
            norm,
            histogram,
            edge,
            detail,
            variance,
            feature_distance,
            tail_distance,
            diversity,
            diversity_match,
            ink,
            support,
            footprint,
            connectivity,
            stroke_profile,
            stroke_shade,
        )
    )
    total.backward()


def test_quality_sampler_keeps_balanced_classes_in_original_order():
    model = ConditionalWGAN(
        latent_dim=16,
        base_channels=8,
        perceptual_tail_weight=1.0,
    ).eval()
    labels = torch.arange(2).repeat_interleave(6)
    images = model(torch.randn(12, 16), labels)
    selected, scores = select_quality_samples(
        model, images, labels, keep_per_class=4, batch_size=3
    )
    assert selected.shape == (8, 1, 28, 28)
    assert scores.shape == (8,)


def test_stroke_shade_gradient_preserves_each_images_mean_tone():
    torch.manual_seed(9)
    real = torch.rand(4, 1, 28, 28).mul(2).sub(1)
    fake = torch.rand(4, 1, 28, 28).mul(2).sub(1).requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1])
    StrokeShadeLoss()(real, fake, labels).backward()
    gradient_means = fake.grad.flatten(1).mean(dim=1)
    torch.testing.assert_close(
        gradient_means,
        torch.zeros_like(gradient_means),
        atol=1e-8,
        rtol=0,
    )
    assert candidate_count(10) == 14
    assert candidate_count(100) == 120
