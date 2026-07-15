"""Export the bundled checkpoint for private, in-browser ONNX inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mnist_wgan.module import ConditionalWGAN
from mnist_wgan.paths import default_checkpoint_path
from mnist_wgan.sampling import (
    DEFAULT_QUALITY_OVERSAMPLE,
    DETACHED_INK_THRESHOLD,
    QUALITY_REJECTION_THRESHOLD,
    UNSUPPORTED_INK_THRESHOLD,
)
from mnist_wgan.visualize import INK_COLOR, PAPER_COLOR


class BrowserGenerator(nn.Module):
    """EMA generator graph without the checkpoint's disabled identity warp."""

    def __init__(self, model: ConditionalWGAN) -> None:
        super().__init__()
        self.generator = model.generator_ema
        if float(self.generator.style_strength) != 0.0:
            raise ValueError("browser export currently requires structured_style_strength=0")

    def forward(self, noise: Tensor, label_one_hot: Tensor) -> Tensor:
        generator = self.generator
        embedded = label_one_hot @ generator.label_embedding.weight
        features = generator.project(torch.cat((noise, embedded), dim=1))
        features = features.view(noise.shape[0], -1, 7, 7)
        features = generator.block2(generator.block1(features))
        return torch.tanh(generator.output(F.relu(generator.bn(features), inplace=False)))


class BrowserQualityScorer(nn.Module):
    """Combine every tensor needed by the UI's selective resampling pass."""

    def __init__(self, model: ConditionalWGAN) -> None:
        super().__init__()
        if model.perceptual_encoder is None:
            raise ValueError("checkpoint does not contain the UI quality classifier")
        self.critic = model.critic
        self.classifier = model.perceptual_encoder
        self.stroke_loss = model.stroke_loss
        self.stroke_profile_loss = model.stroke_profile_loss

    def forward(self, images: Tensor, label_one_hot: Tensor) -> tuple[Tensor, ...]:
        critic_features = self.critic.features(images)
        unconditional = self.critic.score(critic_features).squeeze(1)
        embedded = label_one_hot @ self.critic.label_embedding.weight
        critic_scores = unconditional + (critic_features * embedded).sum(dim=1)
        logits = self.classifier(images)
        _, unsupported, _, disconnected = self.stroke_loss._statistics(images)
        profiles = self.stroke_profile_loss._statistics(images)
        return critic_scores, logits, unsupported, disconnected, profiles


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _export(
    module: nn.Module,
    inputs: tuple[Tensor, ...],
    output_path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
) -> None:
    batch = torch.export.Dim("batch", min=1)
    dynamic_shapes = tuple({0: batch} for _ in inputs)
    program = torch.onnx.export(
        module.eval(),
        inputs,
        input_names=input_names,
        output_names=output_names,
        opset_version=20,
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
        external_data=False,
        optimize=True,
    )
    program.save(output_path, external_data=False)
    onnx.checker.check_model(onnx.load(output_path))


def _verify(
    module: nn.Module,
    inputs: tuple[Tensor, ...],
    output_path: Path,
    output_names: list[str],
) -> dict[str, float]:
    with torch.inference_mode():
        expected = module(*inputs)
    if isinstance(expected, Tensor):
        expected = (expected,)

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    feeds = {
        onnx_input.name: tensor.detach().cpu().numpy()
        for onnx_input, tensor in zip(session.get_inputs(), inputs, strict=True)
    }
    actual = session.run(output_names, feeds)
    errors: dict[str, float] = {}
    for name, torch_output, onnx_output in zip(
        output_names, expected, actual, strict=True
    ):
        reference = torch_output.detach().cpu().numpy()
        np.testing.assert_allclose(onnx_output, reference, rtol=2e-4, atol=2e-5)
        errors[name] = float(np.max(np.abs(onnx_output - reference)))
    return errors


def export_browser_models(checkpoint: Path, output_dir: Path, *, verify: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = ConditionalWGAN.load_from_checkpoint(checkpoint, map_location="cpu").eval()
    latent_dim = int(model.hparams.latent_dim)

    generator = BrowserGenerator(model).eval()
    labels = torch.arange(7, dtype=torch.int64) % 10
    generator_inputs = (
        torch.randn(7, latent_dim, generator=torch.Generator().manual_seed(112)),
        F.one_hot(labels, num_classes=10).float(),
    )
    generator_path = output_dir / "generator.onnx"
    _export(
        generator,
        generator_inputs,
        generator_path,
        input_names=["noise", "label_one_hot"],
        output_names=["images"],
    )

    scorer = BrowserQualityScorer(model).eval()
    with torch.inference_mode():
        scoring_images = generator(*generator_inputs)
    scorer_inputs = (scoring_images, generator_inputs[1])
    scorer_outputs = ["critic_scores", "logits", "unsupported", "disconnected", "profiles"]
    scorer_path = output_dir / "quality-scorer.onnx"
    _export(
        scorer,
        scorer_inputs,
        scorer_path,
        input_names=["images", "label_one_hot"],
        output_names=scorer_outputs,
    )

    verification: dict[str, dict[str, float]] = {}
    if verify:
        verification["generator"] = _verify(
            generator, generator_inputs, generator_path, ["images"]
        )
        verification["quality_scorer"] = _verify(
            scorer, scorer_inputs, scorer_path, scorer_outputs
        )

    manifest = {
        "format_version": 1,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "latent_dim": latent_dim,
        "models": {
            "generator": {
                "path": generator_path.name,
                "bytes": generator_path.stat().st_size,
                "sha256": _sha256(generator_path),
            },
            "quality_scorer": {
                "path": scorer_path.name,
                "bytes": scorer_path.stat().st_size,
                "sha256": _sha256(scorer_path),
            },
        },
        "sampling": {
            "quality_oversample": DEFAULT_QUALITY_OVERSAMPLE,
            "quality_rejection_threshold": QUALITY_REJECTION_THRESHOLD,
            "detached_ink_threshold": DETACHED_INK_THRESHOLD,
            "unsupported_ink_threshold": UNSUPPORTED_INK_THRESHOLD,
        },
        "rendering": {"paper_color": PAPER_COLOR, "ink_color": INK_COLOR},
        "verification_max_absolute_error": verification,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint_path())
    parser.add_argument(
        "--output-dir", type=Path, default=Path("browser/public/models")
    )
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    export_browser_models(args.checkpoint, args.output_dir, verify=not args.skip_verify)
    print(f"Exported browser models to {args.output_dir}")


if __name__ == "__main__":
    main()
