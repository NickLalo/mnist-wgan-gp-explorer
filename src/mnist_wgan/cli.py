"""Console entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from mnist_wgan.callbacks import SampleGridCallback
from mnist_wgan.classifier import ensure_classifier
from mnist_wgan.data import MNISTDataModule
from mnist_wgan.metrics import evaluate_checkpoint
from mnist_wgan.module import ConditionalWGAN
from mnist_wgan.paths import default_checkpoint_path


def train() -> None:
    parser = argparse.ArgumentParser(description="Train the conditional MNIST WGAN-GP")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--critic-steps", type=int, default=3)
    parser.add_argument("--generator-lr", type=float, default=5e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--diversity-weight", type=float, default=3.0)
    parser.add_argument("--diversity-final-weight", type=float, default=2.0)
    parser.add_argument("--diversity-decay-epochs", type=int)
    parser.add_argument("--variance-weight", type=float, default=5.0)
    parser.add_argument("--feature-distribution-weight", type=float, default=0.0)
    parser.add_argument("--perceptual-distribution-weight", type=float, default=50.0)
    parser.add_argument("--perceptual-tail-weight", type=float, default=7.0)
    parser.add_argument("--perceptual-tail-fraction", type=float, default=0.20)
    parser.add_argument("--distribution-start-epoch", type=int, default=0)
    parser.add_argument("--distribution-ramp-epochs", type=int, default=5)
    parser.add_argument(
        "--perceptual-checkpoint", type=Path, default=Path("artifacts/perceptual_classifier.pt")
    )
    parser.add_argument("--style-strength", type=float, default=0.0)
    parser.add_argument("--gradient-penalty-weight", type=float, default=10.0)
    parser.add_argument("--label-consistency-weight", type=float, default=0.5)
    parser.add_argument("--ink-weight", type=float, default=10.0)
    parser.add_argument("--stroke-support-weight", type=float, default=10.0)
    parser.add_argument("--class-footprint-weight", type=float, default=5.0)
    parser.add_argument("--connectivity-weight", type=float, default=5.0)
    parser.add_argument("--stroke-profile-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-generator-from",
        type=Path,
        help="Warm-start generator and EMA weights while training a fresh critic",
    )
    parser.add_argument("--precision", default="16-mixed", choices=("16-mixed", "32-true"))
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    args = parser.parse_args()
    if args.resume and args.init_generator_from:
        parser.error("--resume and --init-generator-from are mutually exclusive")

    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    artifacts = Path(args.artifacts_dir)
    checkpoint_dir = artifacts / "checkpoints"
    data = MNISTDataModule(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
    )
    model = ConditionalWGAN(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        structured_style_strength=args.style_strength,
        generator_lr=args.generator_lr,
        critic_lr=args.critic_lr,
        critic_steps=args.critic_steps,
        gradient_penalty_weight=args.gradient_penalty_weight,
        label_consistency_weight=args.label_consistency_weight,
        diversity_weight=args.diversity_weight,
        diversity_final_weight=args.diversity_final_weight,
        diversity_decay_epochs=args.diversity_decay_epochs or max(int(args.epochs * 0.75), 1),
        variance_weight=args.variance_weight,
        feature_distribution_weight=args.feature_distribution_weight,
        perceptual_distribution_weight=args.perceptual_distribution_weight,
        perceptual_tail_weight=args.perceptual_tail_weight,
        perceptual_tail_fraction=args.perceptual_tail_fraction,
        distribution_start_epoch=args.distribution_start_epoch,
        distribution_ramp_epochs=args.distribution_ramp_epochs,
        ink_weight=args.ink_weight,
        stroke_support_weight=args.stroke_support_weight,
        class_footprint_weight=args.class_footprint_weight,
        connectivity_weight=args.connectivity_weight,
        stroke_profile_weight=args.stroke_profile_weight,
    )
    if args.perceptual_distribution_weight > 0 or args.perceptual_tail_weight > 0:
        perceptual_encoder, perceptual_accuracy = ensure_classifier(
            args.data_dir,
            args.perceptual_checkpoint,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        model.set_perceptual_encoder(perceptual_encoder)
        del perceptual_encoder
        print(f"perceptual encoder accuracy={perceptual_accuracy:.4%}")
    if args.init_generator_from:
        source = ConditionalWGAN.load_from_checkpoint(args.init_generator_from, map_location="cpu")
        model.generator.load_state_dict(source.generator.state_dict())
        model.generator_ema.load_state_dict(source.generator_ema.state_dict())
        print(f"initialized generator weights from {args.init_generator_from}")
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch-{epoch:03d}",
            every_n_epochs=5,
            save_top_k=-1,
            save_last=True,
        ),
        SampleGridCallback(artifacts / "samples"),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = TensorBoardLogger(save_dir=artifacts / "logs", name="conditional_wgan")
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        benchmark=torch.cuda.is_available(),
        limit_train_batches=args.limit_train_batches,
        enable_checkpointing=True,
    )
    trainer.fit(model, datamodule=data, ckpt_path=args.resume)
    final_path = checkpoint_dir / "last.ckpt"
    # ModelCheckpoint only refreshes ``last.ckpt`` on its periodic trigger.
    # Always persist the actual final epoch so short fine-tuning runs are not
    # silently evaluated from an earlier checkpoint.
    trainer.save_checkpoint(final_path)
    print(f"training complete: {final_path}")
    print("next: uv run mnist-wgan-evaluate")


def evaluate() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained conditional MNIST WGAN")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint_path())
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    parser.add_argument("--samples-per-digit", type=int, default=800)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument("--device")
    parser.add_argument("--style-strength", type=float)
    parser.add_argument(
        "--quality-oversample",
        type=float,
        default=1.0,
        help=(
            "Audit UI-like selective resampling by generating this multiple of the "
            "requested candidates per class (raw checkpoint evaluation: 1.0)"
        ),
    )
    parser.add_argument(
        "--raw-generator", action="store_true", help="Evaluate raw rather than EMA weights"
    )
    args = parser.parse_args()
    report = evaluate_checkpoint(
        args.checkpoint,
        data_dir=args.data_dir,
        output_path=args.output,
        samples_per_digit=args.samples_per_digit,
        seed=args.seed,
        device=args.device,
        use_ema=not args.raw_generator,
        style_strength=args.style_strength,
        quality_oversample=args.quality_oversample,
    )
    print(json.dumps(report, indent=2))


def serve() -> None:
    parser = argparse.ArgumentParser(description="Launch the MNIST WGAN-GP Explorer")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint_path())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device")
    args = parser.parse_args()
    import uvicorn

    from mnist_wgan.app import create_app

    uvicorn.run(create_app(args.checkpoint, args.device), host=args.host, port=args.port)
