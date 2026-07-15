"""Repository paths shared by the command-line and inference APIs."""

from pathlib import Path

LOCAL_CHECKPOINT = Path("artifacts/checkpoints/last.ckpt")
BUNDLED_CHECKPOINT = Path("checkpoints/mnist-wgan-gp-inference.ckpt")


def default_checkpoint_path() -> Path:
    """Prefer the latest local training result, then the bundled inference model."""
    return LOCAL_CHECKPOINT if LOCAL_CHECKPOINT.exists() else BUNDLED_CHECKPOINT
