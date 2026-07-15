"""Regularizers adapted for unpaired conditional image generation."""

from __future__ import annotations

import torch
from torch import Tensor, autograd, nn
from torch.nn import functional as F


def gradient_penalty(
    critic: nn.Module,
    real: Tensor,
    fake: Tensor,
    labels: Tensor,
) -> tuple[Tensor, Tensor]:
    """One-sided interpolation gradient penalty and its mean gradient norm."""
    batch_size = real.shape[0]
    alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)
    interpolated = (real.float() + alpha * (fake.float() - real.float())).requires_grad_(True)
    with torch.autocast(device_type=real.device.type, enabled=False):
        scores = critic(interpolated, labels)
        gradients = autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        norms = gradients.flatten(1).norm(2, dim=1)
        penalty = ((norms - 1.0) ** 2).mean()
    return penalty, norms.mean().detach()


class BatchStructureLoss(nn.Module):
    """Match intensity and high-frequency distributions without pixel pairing."""

    def __init__(self, bins: int = 24, sigma: float = 0.065) -> None:
        super().__init__()
        self.sigma = sigma
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, bins))
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
            / 8.0,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
            / 8.0,
        )
        self.register_buffer(
            "laplacian",
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3),
        )

    def _histogram(self, x: Tensor) -> Tensor:
        values = x.float().flatten().unsqueeze(1)
        distances = (values - self.centers.float().unsqueeze(0)) / self.sigma
        histogram = torch.exp(-0.5 * distances.square()).sum(dim=0)
        return histogram / histogram.sum().clamp_min(1e-8)

    @staticmethod
    def _moments(values: Tensor) -> Tensor:
        per_image = values.flatten(1).mean(dim=1)
        return torch.stack((per_image.mean(), per_image.std(unbiased=False)))

    def forward(
        self, real: Tensor, fake: Tensor, labels: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        real32, fake32 = real.float(), fake.float()
        histogram = F.l1_loss(self._histogram(fake32), self._histogram(real32))

        def edge_energy(x: Tensor) -> Tensor:
            gx = F.conv2d(x, self.sobel_x.float(), padding=1)
            gy = F.conv2d(x, self.sobel_y.float(), padding=1)
            return torch.sqrt(gx.square() + gy.square() + 1e-8)

        real_edges, fake_edges = edge_energy(real32), edge_energy(fake32)
        edge = F.l1_loss(self._moments(fake_edges), self._moments(real_edges))
        real_detail = F.conv2d(real32, self.laplacian.float(), padding=1).abs()
        fake_detail = F.conv2d(fake32, self.laplacian.float(), padding=1).abs()
        detail = F.l1_loss(self._moments(fake_detail), self._moments(real_detail))
        # Match where each class varies across a minibatch. Unlike reconstruction,
        # this rewards changes in stroke position/width without tying a latent draw
        # to one arbitrary real image.
        if labels is None:
            variance = F.l1_loss(
                fake32.std(dim=0, unbiased=False), real32.std(dim=0, unbiased=False)
            )
        else:
            class_losses = []
            for digit in labels.unique():
                mask = labels == digit
                if int(mask.sum()) >= 2:
                    class_losses.append(
                        F.l1_loss(
                            fake32[mask].std(dim=0, unbiased=False),
                            real32[mask].std(dim=0, unbiased=False),
                        )
                    )
            variance = (
                torch.stack(class_losses).mean()
                if class_losses
                else F.l1_loss(fake32.std(dim=0, unbiased=False), real32.std(dim=0, unbiased=False))
            )
        return histogram, edge, detail, variance


def mode_seeking_loss(first: Tensor, second: Tensor, z_first: Tensor, z_second: Tensor) -> Tensor:
    """Normalized output change per unit latent change; maximize this quantity."""
    image_distance = (first.float() - second.float()).abs().flatten(1).mean(dim=1)
    latent_distance = (z_first.float() - z_second.float()).abs().mean(dim=1)
    return (image_distance / latent_distance.clamp_min(1e-6)).mean()


def _classwise_moment_loss(real_values: Tensor, fake_values: Tensor, labels: Tensor) -> Tensor:
    """Match per-class means and standard deviations for per-sample statistics."""
    losses = []
    for digit in labels.unique():
        mask = labels == digit
        real_class, fake_class = real_values[mask], fake_values[mask]
        losses.append(
            F.l1_loss(fake_class.mean(), real_class.mean())
            + F.l1_loss(
                fake_class.std(unbiased=False),
                real_class.std(unbiased=False),
            )
        )
    return torch.stack(losses).mean()


def _classwise_quantile_loss(real_values: Tensor, fake_values: Tensor, labels: Tensor) -> Tensor:
    """Match complete per-class scalar distributions with a 1D Wasserstein loss."""
    losses = []
    for digit in labels.unique():
        mask = labels == digit
        losses.append(
            F.l1_loss(
                fake_values[mask].sort(dim=0).values,
                real_values[mask].detach().sort(dim=0).values,
            )
        )
    return torch.stack(losses).mean()


def _soft_erode(images: Tensor) -> Tensor:
    return -F.max_pool2d(-images, kernel_size=3, stride=1, padding=1)


def _soft_open(images: Tensor) -> Tensor:
    eroded = _soft_erode(images)
    return F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)


def _soft_skeleton(images: Tensor, iterations: int = 10) -> Tensor:
    """Differentiable morphological skeleton adapted from soft-clDice."""
    opened = _soft_open(images)
    skeleton = F.relu(images - opened)
    working = images
    for _ in range(iterations):
        working = _soft_erode(working)
        delta = F.relu(working - _soft_open(working))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


class StrokeProfileLoss(nn.Module):
    """Match per-class stroke-width, strength, and centerline distributions.

    Sorting each per-sample statistic produces an unpaired one-dimensional
    Wasserstein objective. Unlike mean/variance matching, malformed samples in
    either tail remain visible to the loss; unlike reconstruction, no generated
    handwriting style is tied to an arbitrary real image.
    """

    def __init__(self, threshold: float = 0.20, temperature: float = 0.04) -> None:
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature

    def _statistics(self, images: Tensor) -> Tensor:
        ink = images.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
        occupancy = torch.sigmoid((ink - self.threshold) / self.temperature)
        strong_occupancy = torch.sigmoid((ink - 0.55) / 0.06)
        skeleton = _soft_skeleton(occupancy)
        occupancy_mass = occupancy.flatten(1).sum(dim=1).clamp_min(1.0)
        skeleton_mass = skeleton.flatten(1).sum(dim=1).clamp_min(1.0)

        # Typical MNIST strokes are a few pixels wide. Scaling keeps all four
        # profile coordinates numerically comparable during fine-tuning.
        width = occupancy_mass / skeleton_mass / 4.0
        strength = (ink * occupancy).flatten(1).sum(dim=1) / occupancy_mass
        strong_fraction = strong_occupancy.flatten(1).sum(dim=1) / occupancy_mass
        centerline_extent = skeleton_mass / 28.0
        return torch.stack((width, strength, strong_fraction, centerline_extent), dim=1)

    def forward(self, real: Tensor, fake: Tensor, labels: Tensor) -> Tensor:
        return _classwise_quantile_loss(
            self._statistics(real),
            self._statistics(fake),
            labels,
        )


class StrokeIntegrityLoss(nn.Module):
    """Match MNIST ink amount, local stroke support, and class footprint.

    The local-support term makes isolated bright pixels and tiny detached
    components expensive without imposing a paired reconstruction target.
    All statistics are matched to real samples from the same class so thin
    legitimate strokes are retained.
    """

    def __init__(self, threshold: float = 0.20, temperature: float = 0.04) -> None:
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature

    def _statistics(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ink = images.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
        occupancy = torch.sigmoid((ink - self.threshold) / self.temperature)
        # Match the same soft edge-residue statistic used by evaluation. A dim
        # pixel with little local ink is characteristic of the tiny hooks and
        # flecks that remain visible around otherwise good generated strokes.
        local_ink = F.avg_pool2d(ink, kernel_size=3, stride=1, padding=1)
        unsupported = ink * F.relu(self.threshold - local_ink)
        ink_mass = ink.flatten(1).sum(dim=1) / 100.0
        unsupported_mass = unsupported.flatten(1).sum(dim=1)
        # Differentiable morphological reconstruction: grow the portion of the
        # stroke that intersects MNIST's central writing region, constrained by
        # foreground occupancy. Ink that cannot be reached is a detached piece.
        foreground = ((occupancy - 0.02) / 0.98).clamp(0.0, 1.0)
        seed_mask = torch.zeros_like(foreground)
        seed_mask[:, :, 8:20, 8:20] = 1.0
        reachable = foreground * seed_mask
        for _ in range(14):
            reachable = torch.minimum(
                foreground,
                F.max_pool2d(reachable, kernel_size=3, stride=1, padding=1),
            )
        disconnected_mass = F.relu(foreground - reachable).flatten(1).sum(dim=1) / 10.0
        return ink_mass, unsupported_mass, occupancy, disconnected_mass

    def forward(
        self, real: Tensor, fake: Tensor, labels: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        real_ink, real_unsupported, real_occupancy, real_disconnected = self._statistics(real)
        fake_ink, fake_unsupported, fake_occupancy, fake_disconnected = self._statistics(fake)
        ink = _classwise_moment_loss(real_ink, fake_ink, labels)
        support = _classwise_moment_loss(real_unsupported, fake_unsupported, labels)
        connectivity = _classwise_moment_loss(real_disconnected, fake_disconnected, labels)

        footprint_losses = []
        for digit in labels.unique():
            mask = labels == digit
            footprint_losses.append(
                F.l1_loss(
                    fake_occupancy[mask].mean(dim=0),
                    real_occupancy[mask].mean(dim=0),
                )
            )
        footprint = torch.stack(footprint_losses).mean()
        return ink, support, footprint, connectivity


def diversity_matching_loss(
    real: Tensor,
    fake_first: Tensor,
    fake_second: Tensor,
    labels: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Match, rather than maximize, within-class low-frequency variation.

    An unbounded pixel-space mode-seeking reward can obtain cheap diversity
    through specks and malformed stroke tips. Matching real pairwise distances
    preserves stochastic variation while also penalizing excessive variation.
    """
    real_low = F.avg_pool2d(real.float(), kernel_size=2)
    first_low = F.avg_pool2d(fake_first.float(), kernel_size=2)
    second_low = F.avg_pool2d(fake_second.float(), kernel_size=2)
    real_distances = torch.zeros(labels.shape[0], device=real.device)
    for digit in labels.unique():
        indices = torch.where(labels == digit)[0]
        paired = indices.roll(1)
        real_distances[indices] = (
            (real_low[indices] - real_low[paired]).abs().flatten(1).mean(dim=1)
        )
    fake_distances = (first_low - second_low).abs().flatten(1).mean(dim=1)
    loss = _classwise_moment_loss(real_distances, fake_distances, labels)
    return loss, real_distances.mean().detach(), fake_distances.mean().detach()


def sliced_feature_distance(
    real_features: Tensor,
    fake_features: Tensor,
    labels: Tensor,
    projections: int = 32,
) -> Tensor:
    """Class-conditional sliced Wasserstein distance in a learned feature space."""
    real_features = F.normalize(real_features.float(), dim=1)
    fake_features = F.normalize(fake_features.float(), dim=1)
    directions = torch.randn(
        real_features.shape[1], projections, device=real_features.device, dtype=torch.float32
    )
    directions = F.normalize(directions, dim=0)
    losses = []
    for digit in labels.unique():
        mask = labels == digit
        real_projection = real_features[mask] @ directions
        fake_projection = fake_features[mask] @ directions
        losses.append(
            F.l1_loss(
                fake_projection.sort(dim=0).values,
                real_projection.sort(dim=0).values,
            )
        )
    return torch.stack(losses).mean()


def perceptual_tail_loss(
    real_features: Tensor,
    fake_features: Tensor,
    labels: Tensor,
    tail_fraction: float = 0.25,
) -> Tensor:
    """Pull the worst class-conditioned perceptual outliers toward real MNIST.

    Distribution matching is dominated by typical samples. This complementary
    objective only follows the most distant generated embeddings in each class,
    giving rare malformed digits a useful gradient without pairing images.
    """
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    real_features = F.normalize(real_features.detach().float(), dim=1)
    fake_features = F.normalize(fake_features.float(), dim=1)
    losses = []
    for digit in labels.unique():
        mask = labels == digit
        nearest_real = torch.cdist(fake_features[mask], real_features[mask]).min(dim=1).values
        tail_size = max(1, int(nearest_real.shape[0] * tail_fraction + 0.999))
        losses.append(nearest_real.topk(tail_size, largest=True).values.mean())
    return torch.stack(losses).mean()
