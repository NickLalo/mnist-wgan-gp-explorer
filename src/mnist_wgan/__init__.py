"""Conditional WGAN-GP for MNIST."""

from mnist_wgan.models import ConditionalCritic, ConditionalGenerator
from mnist_wgan.module import ConditionalWGAN

__all__ = ["ConditionalCritic", "ConditionalGenerator", "ConditionalWGAN"]
__version__ = "0.1.0"
