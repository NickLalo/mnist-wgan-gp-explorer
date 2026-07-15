"""Tensor-to-image utilities shared by training, evaluation, and the UI."""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch import Tensor

PAPER_COLOR = "#e8e5da"
INK_COLOR = "#082a22"


def tensor_to_image(image: Tensor, scale: int = 3) -> Image.Image:
    pixels = image.detach().float().squeeze().clamp(-1, 1).add(1).mul(127.5)
    pil = Image.fromarray(pixels.byte().cpu().numpy(), mode="L")
    colored = ImageOps.colorize(pil, black=PAPER_COLOR, white=INK_COLOR)
    return colored.resize((28 * scale, 28 * scale), Image.Resampling.NEAREST)


def image_grid(
    images: Tensor,
    *,
    rows: int,
    columns: int,
    row_labels: list[str] | None = None,
    scale: int = 3,
    gap: int = 8,
) -> Image.Image:
    tile = 28 * scale
    label_width = 42 if row_labels else 0
    width = label_width + columns * tile + max(columns - 1, 0) * gap
    height = rows * tile + max(rows - 1, 0) * gap
    canvas = Image.new("RGB", (width, height), PAPER_COLOR)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(12, scale * 6))
    for index, image in enumerate(images[: rows * columns]):
        row, column = divmod(index, columns)
        x = label_width + column * (tile + gap)
        y = row * (tile + gap)
        canvas.paste(tensor_to_image(image, scale).convert("RGB"), (x, y))
    if row_labels:
        for row, label in enumerate(row_labels[:rows]):
            box = draw.textbbox((0, 0), label, font=font)
            x = (label_width - (box[2] - box[0])) // 2
            y = row * (tile + gap) + (tile - (box[3] - box[1])) // 2
            draw.text((x, y), label, fill=INK_COLOR, font=font)
    return canvas


def single_digit_grid(
    images: Tensor, *, columns: int = 10, scale: int = 3, gap: int = 8
) -> Image.Image:
    columns = min(columns, len(images))
    rows = math.ceil(len(images) / columns)
    return image_grid(images, rows=rows, columns=columns, scale=scale, gap=gap)


def to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def save_image(image: Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def seeded_noise(count: int, latent_dim: int, seed: int, device: torch.device) -> Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(count, latent_dim, generator=generator, device=device)


def latent_plane(seed: int, latent_dim: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    vectors = seeded_noise(3, latent_dim, seed, device)
    base = vectors[0]
    first = vectors[1] - torch.dot(vectors[1], base) / base.square().sum() * base
    first = first / first.norm().clamp_min(1e-8)
    second = vectors[2]
    second = second - torch.dot(second, base) / base.square().sum() * base
    second = second - torch.dot(second, first) * first
    second = second / second.norm().clamp_min(1e-8)
    # A radius of sqrt(latent_dim) gives each UI unit a perceptible but smooth effect.
    radius = float(np.sqrt(latent_dim))
    return base, first * radius, second * radius
