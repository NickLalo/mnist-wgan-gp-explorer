"""PyTorch Lightning module for conditional WGAN-GP training."""

from __future__ import annotations

import copy

import lightning as L
import torch
from torch import Tensor
from torch.nn import functional as F

from mnist_wgan.classifier import MNISTClassifier
from mnist_wgan.losses import (
    BatchStructureLoss,
    StrokeHaloLoss,
    StrokeIntegrityLoss,
    StrokeProfileLoss,
    StrokeShadeLoss,
    diversity_matching_loss,
    gradient_penalty,
    perceptual_tail_loss,
    sliced_feature_distance,
)
from mnist_wgan.models import ConditionalCritic, ConditionalGenerator


class ConditionalWGAN(L.LightningModule):
    """Conditional WGAN with manual optimization and EMA inference weights."""

    def __init__(
        self,
        latent_dim: int = 96,
        base_channels: int = 32,
        structured_style_strength: float = 0.0,
        generator_lr: float = 1.0e-4,
        critic_lr: float = 2.0e-4,
        critic_steps: int = 3,
        gradient_penalty_weight: float = 10.0,
        drift_weight: float = 1.0e-3,
        label_consistency_weight: float = 0.5,
        label_margin: float = 1.0,
        histogram_weight: float = 5.0,
        edge_weight: float = 0.5,
        detail_weight: float = 0.25,
        variance_weight: float = 10.0,
        feature_distribution_weight: float = 300.0,
        perceptual_distribution_weight: float = 0.0,
        perceptual_tail_weight: float = 0.0,
        perceptual_tail_fraction: float = 0.25,
        diversity_weight: float = 3.0,
        diversity_final_weight: float = 2.0,
        diversity_decay_epochs: int = 30,
        distribution_start_epoch: int = 15,
        distribution_ramp_epochs: int = 10,
        ink_weight: float = 10.0,
        stroke_support_weight: float = 2.0,
        class_footprint_weight: float = 5.0,
        connectivity_weight: float = 5.0,
        stroke_profile_weight: float = 10.0,
        stroke_shade_weight: float = 0.0,
        stroke_shade_tail_fraction: float = 0.25,
        stroke_halo_weight: float = 0.0,
        stroke_halo_tail_fraction: float = 0.50,
        ema_decay: float = 0.995,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Permits continuing an older checkpoint before the optional frozen
        # perceptual encoder was introduced.
        self.strict_loading = False
        self.automatic_optimization = False
        self.generator = ConditionalGenerator(latent_dim, base_channels, structured_style_strength)
        self.critic = ConditionalCritic(base_channels)
        self.generator_ema = copy.deepcopy(self.generator).requires_grad_(False)
        self.generator_ema.eval()
        self.structure_loss = BatchStructureLoss()
        self.stroke_loss = StrokeIntegrityLoss()
        self.stroke_profile_loss = StrokeProfileLoss()
        self.stroke_shade_loss = StrokeShadeLoss(stroke_shade_tail_fraction)
        self.stroke_halo_loss = StrokeHaloLoss(stroke_halo_tail_fraction)
        self.perceptual_encoder = (
            MNISTClassifier().requires_grad_(False)
            if perceptual_distribution_weight > 0 or perceptual_tail_weight > 0
            else None
        )
        if self.perceptual_encoder is not None:
            self.perceptual_encoder.eval()

    def set_perceptual_encoder(self, encoder: MNISTClassifier) -> None:
        if self.perceptual_encoder is None:
            raise RuntimeError("a perceptual loss weight must be positive")
        self.perceptual_encoder.load_state_dict(encoder.state_dict())
        self.perceptual_encoder.requires_grad_(False).eval()

    def forward(self, z: Tensor, labels: Tensor, *, use_ema: bool = True) -> Tensor:
        return (self.generator_ema if use_ema else self.generator)(z, labels)

    @property
    def current_diversity_weight(self) -> float:
        progress = min(self.current_epoch / max(self.hparams.diversity_decay_epochs, 1), 1.0)
        start = self.hparams.diversity_weight
        return start + progress * (self.hparams.diversity_final_weight - start)

    @property
    def distribution_scale(self) -> float:
        progress = (self.current_epoch - self.hparams.distribution_start_epoch) / max(
            self.hparams.distribution_ramp_epochs, 1
        )
        return min(max(progress, 0.0), 1.0)

    @staticmethod
    def _wrong_labels(labels: Tensor) -> Tensor:
        offsets = torch.randint(1, 10, labels.shape, device=labels.device)
        return (labels + offsets) % 10

    @torch.no_grad()
    def _update_ema(self) -> None:
        decay = self.hparams.ema_decay
        for ema, current in zip(
            self.generator_ema.parameters(), self.generator.parameters(), strict=True
        ):
            ema.lerp_(current, 1.0 - decay)
        for ema, current in zip(
            self.generator_ema.buffers(), self.generator.buffers(), strict=True
        ):
            ema.copy_(current)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        real, labels = batch
        generator_optimizer, critic_optimizer = self.optimizers()
        batch_size = real.shape[0]

        critic_loss_total = torch.zeros((), device=self.device)
        wasserstein_total = torch.zeros((), device=self.device)
        gp_total = torch.zeros((), device=self.device)
        gradient_norm_total = torch.zeros((), device=self.device)

        for _ in range(self.hparams.critic_steps):
            self.toggle_optimizer(critic_optimizer)
            z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
            with torch.no_grad():
                fake = self.generator(z, labels)
            real_scores = self.critic(real, labels)
            fake_scores = self.critic(fake, labels)
            wrong_scores = self.critic(real, self._wrong_labels(labels))
            # The Wasserstein objective compares the real and generated joint
            # distributions. A linear wrong-label score here is unbounded and
            # previously overwhelmed image realism; use a bounded margin instead.
            wasserstein = fake_scores.mean() - real_scores.mean()
            label_consistency = F.relu(
                self.hparams.label_margin - (real_scores - wrong_scores)
            ).mean()
            gp, gradient_norm = gradient_penalty(self.critic, real, fake, labels)
            drift = self.hparams.drift_weight * real_scores.square().mean()
            critic_loss = (
                wasserstein
                + self.hparams.label_consistency_weight * label_consistency
                + self.hparams.gradient_penalty_weight * gp
                + drift
            )
            critic_optimizer.zero_grad(set_to_none=True)
            self.manual_backward(critic_loss)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            critic_optimizer.step()
            self.untoggle_optimizer(critic_optimizer)
            critic_loss_total += critic_loss.detach()
            wasserstein_total += wasserstein.detach()
            gp_total += gp.detach()
            gradient_norm_total += gradient_norm

        self.toggle_optimizer(generator_optimizer)
        z_first = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        z_second = torch.randn_like(z_first)
        fake_first = self.generator(z_first, labels)
        fake_second = self.generator(z_second, labels)
        fake_scores = self.critic(fake_first, labels)
        adversarial = -fake_scores.mean()
        histogram, edge, detail, variance = self.structure_loss(real, fake_first, labels)
        ink, stroke_support, class_footprint, connectivity = self.stroke_loss(
            real, fake_first, labels
        )
        stroke_profile = self.stroke_profile_loss(real, fake_first, labels)
        stroke_shade = self.stroke_shade_loss(real, fake_first, labels)
        stroke_halo = self.stroke_halo_loss(real, fake_first, labels)
        real_features = self.critic.features(real).detach()
        fake_features = self.critic.features(fake_first)
        feature_distribution = sliced_feature_distance(real_features, fake_features, labels)
        if self.perceptual_encoder is not None:
            self.perceptual_encoder.eval()
            with torch.no_grad():
                _, real_perceptual = self.perceptual_encoder(real, return_features=True)
            _, fake_perceptual = self.perceptual_encoder(fake_first, return_features=True)
            perceptual_distribution = sliced_feature_distance(
                real_perceptual, fake_perceptual, labels, projections=64
            )
            perceptual_tail = perceptual_tail_loss(
                real_perceptual,
                fake_perceptual,
                labels,
                self.hparams.perceptual_tail_fraction,
            )
        else:
            perceptual_distribution = torch.zeros((), device=self.device)
            perceptual_tail = torch.zeros((), device=self.device)
        diversity, real_pair_distance, fake_pair_distance = diversity_matching_loss(
            real, fake_first, fake_second, labels
        )
        distribution_scale = self.distribution_scale
        diversity_weight = 0.2 + distribution_scale * (self.current_diversity_weight - 0.2)
        generator_loss = (
            adversarial
            + self.hparams.histogram_weight * histogram
            + self.hparams.edge_weight * edge
            + self.hparams.detail_weight * detail
            + distribution_scale * self.hparams.variance_weight * variance
            + distribution_scale * self.hparams.feature_distribution_weight * feature_distribution
            + distribution_scale
            * self.hparams.perceptual_distribution_weight
            * perceptual_distribution
            + distribution_scale * self.hparams.perceptual_tail_weight * perceptual_tail
            + self.hparams.ink_weight * ink
            + self.hparams.stroke_support_weight * stroke_support
            + self.hparams.class_footprint_weight * class_footprint
            + self.hparams.connectivity_weight * connectivity
            + distribution_scale * self.hparams.stroke_profile_weight * stroke_profile
            + distribution_scale * self.hparams.stroke_shade_weight * stroke_shade
            + distribution_scale * self.hparams.stroke_halo_weight * stroke_halo
            + diversity_weight * diversity
        )
        generator_optimizer.zero_grad(set_to_none=True)
        self.manual_backward(generator_loss)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 10.0)
        generator_optimizer.step()
        self.untoggle_optimizer(generator_optimizer)
        self._update_ema()

        critic_steps = float(self.hparams.critic_steps)
        metrics = {
            "train/generator_loss": generator_loss.detach(),
            "train/critic_loss": critic_loss_total / critic_steps,
            "train/wasserstein": wasserstein_total / critic_steps,
            "train/gradient_penalty": gp_total / critic_steps,
            "train/gradient_norm": gradient_norm_total / critic_steps,
            "train/adversarial": adversarial.detach(),
            "train/histogram": histogram.detach(),
            "train/edge": edge.detach(),
            "train/detail": detail.detach(),
            "train/variance": variance.detach(),
            "train/feature_distribution": feature_distribution.detach(),
            "train/perceptual_distribution": perceptual_distribution.detach(),
            "train/perceptual_tail": perceptual_tail.detach(),
            "train/diversity": diversity.detach(),
            "train/real_pair_distance": real_pair_distance,
            "train/fake_pair_distance": fake_pair_distance,
            "train/ink": ink.detach(),
            "train/stroke_support": stroke_support.detach(),
            "train/class_footprint": class_footprint.detach(),
            "train/connectivity": connectivity.detach(),
            "train/stroke_profile": stroke_profile.detach(),
            "train/stroke_shade": stroke_shade.detach(),
            "train/stroke_halo": stroke_halo.detach(),
            "train/distribution_scale": torch.tensor(distribution_scale, device=self.device),
        }
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("g_loss", generator_loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "critic_gap",
            -wasserstein_total / critic_steps,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.generator_lr,
            betas=(0.0, 0.9),
            eps=1e-8,
        )
        critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.hparams.critic_lr,
            betas=(0.0, 0.9),
            eps=1e-8,
        )
        return [generator_optimizer, critic_optimizer]

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        real, labels = batch
        noise = torch.randn(real.shape[0], self.hparams.latent_dim, device=self.device)
        with torch.no_grad():
            fake = self.generator_ema(noise, labels)
            gap = self.critic(real, labels).mean() - self.critic(fake, labels).mean()
            wrong_gap = (
                self.critic(real, labels).mean()
                - self.critic(real, self._wrong_labels(labels)).mean()
            )
        self.log_dict(
            {"validation/critic_gap": gap, "validation/label_gap": wrong_gap},
            on_step=False,
            on_epoch=True,
            batch_size=real.shape[0],
        )
