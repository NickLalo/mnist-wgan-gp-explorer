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
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from torch import Tensor, nn
from torch.nn import functional as F

from mnist_wgan.module import ConditionalWGAN
from mnist_wgan.paths import default_checkpoint_path
from mnist_wgan.sampling import (
    DEFAULT_QUALITY_OVERSAMPLE,
    DETACHED_INK_THRESHOLD,
    QUALITY_REJECTION_THRESHOLD,
    STROKE_SHADE_CV_THRESHOLDS,
    STROKE_SHADE_DIP_THRESHOLDS,
    STROKE_SHADE_REJECTION_MULTIPLIER,
    UNSUPPORTED_INK_THRESHOLD,
)
from mnist_wgan.visualize import INK_COLOR, PAPER_COLOR


class _CalibrationReader(CalibrationDataReader):
    def __init__(self, rows: list[dict[str, np.ndarray]]) -> None:
        self._rows = iter(rows)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._rows, None)


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
        self.stroke_shade_loss = model.stroke_shade_loss

    def forward(self, images: Tensor, label_one_hot: Tensor) -> tuple[Tensor, ...]:
        critic_features = self.critic.features(images)
        unconditional = self.critic.score(critic_features).squeeze(1)
        embedded = label_one_hot @ self.critic.label_embedding.weight
        critic_scores = unconditional + (critic_features * embedded).sum(dim=1)
        logits = self.classifier(images)
        _, unsupported, _, disconnected = self.stroke_loss._statistics(images)
        profiles = self.stroke_profile_loss._statistics(images)
        shade = self.stroke_shade_loss._statistics(images)
        return critic_scores, logits, unsupported, disconnected, profiles, shade


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


def _quantize_uint8(
    source: Path,
    destination: Path,
    calibration_rows: list[dict[str, np.ndarray]],
) -> None:
    """Create the WASM-oriented static uint8 graph recommended for browser CPUs."""
    quantize_static(
        str(source),
        str(destination),
        _CalibrationReader(calibration_rows),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["Conv", "Gemm", "MatMul"],
        per_channel=True,
    )
    onnx.checker.check_model(onnx.load(destination))


def _quantized_errors(
    module: nn.Module,
    inputs: tuple[Tensor, ...],
    output_path: Path,
    output_names: list[str],
) -> dict[str, dict[str, float]]:
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
    errors = {}
    for name, torch_output, onnx_output in zip(
        output_names, expected, actual, strict=True
    ):
        difference = np.abs(onnx_output - torch_output.detach().cpu().numpy())
        errors[name] = {
            "mean_absolute_error": float(difference.mean()),
            "max_absolute_error": float(difference.max()),
        }
    return errors


def _quantized_quality_agreement(
    module: BrowserQualityScorer,
    inputs: tuple[Tensor, Tensor],
    output_path: Path,
) -> dict[str, float]:
    with torch.inference_mode():
        expected_critic, expected_logits, *_ = module(*inputs)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    feeds = {
        onnx_input.name: tensor.detach().cpu().numpy()
        for onnx_input, tensor in zip(session.get_inputs(), inputs, strict=True)
    }
    actual_critic, actual_logits = session.run(["critic_scores", "logits"], feeds)
    expected_critic_array = expected_critic.detach().cpu().numpy()

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        result = np.empty_like(values, dtype=np.float64)
        result[order] = np.arange(len(values), dtype=np.float64)
        return result

    critic_rank_correlation = float(
        np.corrcoef(ranks(expected_critic_array), ranks(actual_critic))[0, 1]
    )
    expected_classes = expected_logits.detach().cpu().numpy().argmax(axis=1)
    classifier_agreement = float(
        np.mean(expected_classes == actual_logits.argmax(axis=1))
    )
    return {
        "critic_rank_correlation": critic_rank_correlation,
        "classifier_agreement": classifier_agreement,
    }


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
    scorer_outputs = [
        "critic_scores",
        "logits",
        "unsupported",
        "disconnected",
        "profiles",
        "shade",
    ]
    scorer_path = output_dir / "quality-scorer.onnx"
    _export(
        scorer,
        scorer_inputs,
        scorer_path,
        input_names=["images", "label_one_hot"],
        output_names=scorer_outputs,
    )

    calibration_generator_rows: list[dict[str, np.ndarray]] = []
    calibration_scorer_rows: list[dict[str, np.ndarray]] = []
    calibration_rng = torch.Generator().manual_seed(737)
    with torch.inference_mode():
        for batch_index in range(12):
            calibration_labels = (torch.arange(64) + batch_index) % 10
            calibration_one_hot = F.one_hot(calibration_labels, num_classes=10).float()
            calibration_noise = torch.randn(64, latent_dim, generator=calibration_rng)
            calibration_generator_rows.append(
                {
                    "noise": calibration_noise.numpy(),
                    "label_one_hot": calibration_one_hot.numpy(),
                }
            )
            calibration_scorer_rows.append(
                {
                    "images": generator(calibration_noise, calibration_one_hot).numpy(),
                    "label_one_hot": calibration_one_hot.numpy(),
                }
            )

    quantized_generator_path = output_dir / "generator-uint8.onnx"
    quantized_scorer_path = output_dir / "quality-scorer-uint8.onnx"
    _quantize_uint8(generator_path, quantized_generator_path, calibration_generator_rows)
    _quantize_uint8(scorer_path, quantized_scorer_path, calibration_scorer_rows)

    verification: dict[str, dict[str, float]] = {}
    if verify:
        verification["generator"] = _verify(
            generator, generator_inputs, generator_path, ["images"]
        )
        verification["quality_scorer"] = _verify(
            scorer, scorer_inputs, scorer_path, scorer_outputs
        )
        verification["generator_uint8"] = _quantized_errors(
            generator, generator_inputs, quantized_generator_path, ["images"]
        )
        verification["quality_scorer_uint8"] = _quantized_errors(
            scorer, scorer_inputs, quantized_scorer_path, scorer_outputs
        )
        agreement_inputs = (
            torch.from_numpy(np.concatenate([row["images"] for row in calibration_scorer_rows])),
            torch.from_numpy(
                np.concatenate([row["label_one_hot"] for row in calibration_scorer_rows])
            ),
        )
        verification["quality_scorer_uint8_agreement"] = _quantized_quality_agreement(
            scorer,
            agreement_inputs,
            quantized_scorer_path,
        )
        if verification["generator_uint8"]["images"]["mean_absolute_error"] > 0.015:
            raise AssertionError("quantized generator exceeded the 0.015 mean-error budget")
        if verification["quality_scorer_uint8"]["logits"]["mean_absolute_error"] > 0.20:
            raise AssertionError("quantized quality scorer exceeded the 0.20 logit-error budget")
        if (
            verification["quality_scorer_uint8_agreement"]["critic_rank_correlation"]
            < 0.98
        ):
            raise AssertionError("quantized critic rank correlation fell below 0.98")
        if verification["quality_scorer_uint8_agreement"]["classifier_agreement"] < 0.995:
            raise AssertionError("quantized classifier agreement fell below 99.5%")

    manifest = {
        "format_version": 2,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "latent_dim": latent_dim,
        "models": {
            "generator": {
                "path": generator_path.name,
                "bytes": generator_path.stat().st_size,
                "sha256": _sha256(generator_path),
                "wasm_path": quantized_generator_path.name,
                "wasm_bytes": quantized_generator_path.stat().st_size,
                "wasm_sha256": _sha256(quantized_generator_path),
            },
            "quality_scorer": {
                "path": scorer_path.name,
                "bytes": scorer_path.stat().st_size,
                "sha256": _sha256(scorer_path),
                "wasm_path": quantized_scorer_path.name,
                "wasm_bytes": quantized_scorer_path.stat().st_size,
                "wasm_sha256": _sha256(quantized_scorer_path),
            },
        },
        "sampling": {
            "quality_oversample": DEFAULT_QUALITY_OVERSAMPLE,
            "quality_rejection_threshold": QUALITY_REJECTION_THRESHOLD,
            "detached_ink_threshold": DETACHED_INK_THRESHOLD,
            "unsupported_ink_threshold": UNSUPPORTED_INK_THRESHOLD,
            "stroke_shade_cv_thresholds": STROKE_SHADE_CV_THRESHOLDS,
            "stroke_shade_dip_thresholds": STROKE_SHADE_DIP_THRESHOLDS,
            "stroke_shade_rejection_multiplier": STROKE_SHADE_REJECTION_MULTIPLIER,
        },
        "rendering": {"paper_color": PAPER_COLOR, "ink_color": INK_COLOR},
        "wasm_quantization": {
            "format": "QOperator",
            "activations": "uint8",
            "weights": "uint8",
            "per_channel": True,
        },
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
