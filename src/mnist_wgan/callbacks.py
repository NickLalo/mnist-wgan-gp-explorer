"""Training visualization callback."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch

from mnist_wgan.visualize import image_grid, save_image, seeded_noise


class SampleGridCallback(L.Callback):
    def __init__(self, output_dir: str | Path = "artifacts/samples", every_n_epochs: int = 1):
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs

    @torch.inference_mode()
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module) -> None:
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs:
            return
        device = pl_module.device
        labels = torch.arange(10, device=device).repeat_interleave(8)
        noise = seeded_noise(80, pl_module.hparams.latent_dim, 112, device)
        was_training = pl_module.generator_ema.training
        pl_module.generator_ema.eval()
        generated = pl_module(noise, labels, use_ema=True)
        pl_module.generator_ema.train(was_training)
        grid = image_grid(
            generated, rows=10, columns=8, row_labels=[str(i) for i in range(10)], scale=2
        )
        save_image(grid, self.output_dir / f"epoch_{epoch:03d}.png")
