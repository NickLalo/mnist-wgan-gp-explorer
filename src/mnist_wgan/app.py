"""Inference service for the interactive MNIST WGAN-GP explorer."""

from __future__ import annotations

from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from mnist_wgan.module import ConditionalWGAN
from mnist_wgan.paths import default_checkpoint_path
from mnist_wgan.sampling import candidate_count, select_quality_samples
from mnist_wgan.visualize import (
    image_grid,
    latent_plane,
    seeded_noise,
    single_digit_grid,
    tensor_to_image,
    to_png_bytes,
)


class InferenceEngine:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ConditionalWGAN.load_from_checkpoint(
            self.checkpoint_path, map_location=self.device
        ).to(self.device)
        self.model.eval()
        self.latent_dim = int(self.model.hparams.latent_dim)

    def _generate(
        self, noise: torch.Tensor, labels: torch.Tensor, *, to_cpu: bool = True
    ) -> torch.Tensor:
        """Generate large UI requests in bounded batches."""
        images = torch.cat(
            [
                self.model(noise[start : start + 512], labels[start : start + 512])
                for start in range(0, len(labels), 512)
            ]
        )
        return images.cpu() if to_cpu else images

    def _quality_generate(
        self, labels: torch.Tensor, requested_per_class: int, seed: int
    ) -> torch.Tensor:
        classes = labels.unique(sorted=True)
        candidates_per_class = candidate_count(requested_per_class)
        candidate_labels = classes.repeat_interleave(candidates_per_class)
        noise = seeded_noise(len(candidate_labels), self.latent_dim, seed, self.device)
        candidates = self._generate(noise, candidate_labels, to_cpu=False)
        selected, _ = select_quality_samples(
            self.model,
            candidates,
            candidate_labels,
            requested_per_class,
        )
        return selected.cpu()

    @torch.inference_mode()
    def all_digits(self, samples: int, seed: int) -> bytes:
        labels = torch.arange(10, device=self.device).repeat_interleave(samples)
        images = self._quality_generate(labels, samples, seed)
        grid = image_grid(
            images,
            rows=10,
            columns=samples,
            row_labels=[str(i) for i in range(10)],
            scale=3,
        )
        return to_png_bytes(grid)

    @torch.inference_mode()
    def one_digit(self, digit: int, samples: int, seed: int, scale: float = 2) -> bytes:
        labels = torch.full((samples,), digit, device=self.device)
        images = self._quality_generate(labels, samples, seed)
        # Keep the encoded canvas around the same width at every zoom level.
        # Smaller tiles therefore create more columns and dramatically fewer
        # rows, allowing thousands of samples to remain practical to inspect.
        columns = max(1, round(48 / scale))
        return to_png_bytes(
            single_digit_grid(images, columns=columns, scale=scale, gap=2 * scale)
        )

    @torch.inference_mode()
    def explore(self, digit: int, x: float, y: float, seed: int) -> bytes:
        base, first, second = latent_plane(seed, self.latent_dim, self.device)
        noise = (base + x * first + y * second).unsqueeze(0)
        label = torch.tensor([digit], device=self.device)
        image = tensor_to_image(self.model(noise, label)[0], scale=10).convert("RGB")
        return to_png_bytes(image)


def create_app(
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> FastAPI:
    engine = InferenceEngine(checkpoint_path or default_checkpoint_path(), device)
    app = FastAPI(title="MNIST WGAN-GP Explorer", version="0.1.0")
    static_path = Path(__file__).parent / "static"
    html_path = static_path / "index.html"
    favicon_path = static_path / "favicon.svg"
    favicon_ico_path = static_path / "favicon.ico"

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return html_path.read_text()

    @app.get("/favicon.svg", include_in_schema=False)
    def favicon() -> Response:
        return Response(
            favicon_path.read_bytes(),
            media_type="image/svg+xml",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon_ico() -> Response:
        return Response(
            favicon_ico_path.read_bytes(),
            media_type="image/x-icon",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/info")
    def info() -> dict:
        return {
            "api_version": 2,
            "checkpoint": engine.checkpoint_path.name,
            "device": str(engine.device),
            "latent_dim": engine.latent_dim,
            "limits": {
                "single_digit_samples": 5000,
                "latent_coordinate": 10.0,
            },
        }

    @app.get("/api/all")
    def all_digits(
        samples: int = Query(20, ge=1, le=100), seed: int = Query(112, ge=0, le=2**31 - 1)
    ) -> Response:
        return Response(engine.all_digits(samples, seed), media_type="image/png")

    @app.get("/api/digit")
    def one_digit(
        digit: int = Query(3, ge=0, le=9),
        samples: int = Query(240, ge=1, le=5000),
        seed: int = Query(112, ge=0, le=2**31 - 1),
        scale: float = Query(2, ge=0.4, le=4),
    ) -> Response:
        return Response(engine.one_digit(digit, samples, seed, scale), media_type="image/png")

    @app.get("/api/explore")
    def explore(
        digit: int = Query(3, ge=0, le=9),
        x: float = Query(0.0, ge=-10.0, le=10.0),
        y: float = Query(0.0, ge=-10.0, le=10.0),
        seed: int = Query(112, ge=0, le=2**31 - 1),
    ) -> Response:
        if not math_is_finite(x, y):
            raise HTTPException(400, "Coordinates must be finite")
        return Response(engine.explore(digit, x, y, seed), media_type="image/png")

    return app


def math_is_finite(*values: float) -> bool:
    return all(torch.isfinite(torch.tensor(value)).item() for value in values)
