"""Label-conditioned generator and projection critic."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
        nn.init.orthogonal_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        residual = F.interpolate(x, scale_factor=2, mode="nearest")
        residual = self.skip(residual)
        x = F.relu(self.bn1(x), inplace=True)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.conv2(F.relu(self.bn2(x), inplace=True))
        return x + residual


class ConditionalGenerator(nn.Module):
    """Map a latent vector and digit label to a 28x28 image in [-1, 1]."""

    def __init__(
        self, latent_dim: int = 96, base_channels: int = 64, style_strength: float = 0.0
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.style_strength = style_strength
        self.label_embedding = nn.Embedding(10, 32)
        self.project = nn.Linear(latent_dim + 32, base_channels * 4 * 7 * 7)
        self.block1 = GeneratorBlock(base_channels * 4, base_channels * 2)
        self.block2 = GeneratorBlock(base_channels * 2, base_channels)
        self.bn = nn.BatchNorm2d(base_channels)
        self.output = nn.Conv2d(base_channels, 1, 3, padding=1)
        self.apply(_init_weights)

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        embedded = self.label_embedding(labels)
        x = self.project(torch.cat((z, embedded), dim=1))
        x = x.view(z.shape[0], -1, 7, 7)
        x = self.block2(self.block1(x))
        images = torch.tanh(self.output(F.relu(self.bn(x), inplace=True)))
        return self._apply_structured_style(images, z)

    def _apply_structured_style(self, images: Tensor, z: Tensor) -> Tensor:
        """Use four latent axes for smooth rotation, scale, and translation."""
        if z.shape[1] < 4:
            return images
        strength = self.style_strength
        angle = torch.tanh(z[:, 0].float()) * 0.30 * strength
        scale = torch.exp(torch.tanh(z[:, 1].float()) * 0.13 * strength)
        translate_x = torch.tanh(z[:, 2].float()) * 0.13 * strength
        translate_y = torch.tanh(z[:, 3].float()) * 0.13 * strength
        cosine, sine = torch.cos(angle) * scale, torch.sin(angle) * scale
        theta = torch.zeros(z.shape[0], 2, 3, device=z.device, dtype=torch.float32)
        theta[:, 0, 0] = cosine
        theta[:, 0, 1] = -sine
        theta[:, 1, 0] = sine
        theta[:, 1, 1] = cosine
        theta[:, 0, 2] = translate_x
        theta[:, 1, 2] = translate_y
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        transformed = F.grid_sample(
            images.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return transformed.to(images.dtype)


class CriticBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, downsample: bool = True) -> None:
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def _down(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(x, 2) if self.downsample else x

    def forward(self, x: Tensor) -> Tensor:
        residual = self._down(self.skip(x))
        # The critic must support second-order derivatives for WGAN-GP.
        x = self.conv1(F.leaky_relu(x, 0.2, inplace=False))
        x = self.conv2(F.leaky_relu(x, 0.2, inplace=False))
        return self._down(x) + residual


class ConditionalCritic(nn.Module):
    """Projection critic: image realism plus learned image/label compatibility."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        self.input = nn.Conv2d(1, base_channels, 3, padding=1)
        self.block1 = CriticBlock(base_channels, base_channels * 2)
        self.block2 = CriticBlock(base_channels * 2, base_channels * 4)
        self.block3 = CriticBlock(base_channels * 4, base_channels * 4, downsample=False)
        feature_dim = base_channels * 4
        self.score = nn.Linear(feature_dim, 1)
        self.label_embedding = nn.Embedding(10, feature_dim)
        self.apply(_init_weights)

    def features(self, images: Tensor) -> Tensor:
        x = self.input(images)
        x = self.block3(self.block2(self.block1(x)))
        return F.leaky_relu(x, 0.2, inplace=False).sum(dim=(2, 3))

    def forward(self, images: Tensor, labels: Tensor) -> Tensor:
        features = self.features(images)
        unconditional = self.score(features).squeeze(1)
        projection = (features * self.label_embedding(labels)).sum(dim=1)
        return unconditional + projection
