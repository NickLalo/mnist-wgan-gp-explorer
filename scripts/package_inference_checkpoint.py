"""Strip trainer-only state from a Lightning checkpoint for bundled inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mnist_wgan.module import ConditionalWGAN

INFERENCE_KEYS = (
    "epoch",
    "global_step",
    "pytorch-lightning_version",
    "state_dict",
    "hyper_parameters",
)


def package_checkpoint(source: Path, destination: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    missing = [key for key in INFERENCE_KEYS if key not in checkpoint]
    if missing:
        raise ValueError(f"source checkpoint is missing required keys: {missing}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: checkpoint[key] for key in INFERENCE_KEYS}, destination)
    ConditionalWGAN.load_from_checkpoint(destination, map_location="cpu").eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    package_checkpoint(args.source, args.destination)
    print(f"Packaged inference checkpoint at {args.destination}")


if __name__ == "__main__":
    main()
