from pathlib import Path

import torch
from PIL import Image

from mnist_wgan.visualize import (
    image_grid,
    latent_plane,
    single_digit_grid,
    tensor_to_image,
    to_png_bytes,
)


def test_grids_and_latent_plane():
    images = torch.zeros(20, 1, 28, 28)
    grid = image_grid(images, rows=2, columns=10, row_labels=["0", "1"], scale=2)
    assert grid.width > grid.height
    one = single_digit_grid(images, columns=10, scale=2)
    assert to_png_bytes(one).startswith(b"\x89PNG")
    assert one.getpixel((56, 0)) == (232, 229, 218)
    base, first, second = latent_plane(112, 16, torch.device("cpu"))
    assert base.shape == first.shape == second.shape == (16,)
    assert abs(float(torch.dot(first, second))) < 1e-4


def test_tensor_to_image_uses_paper_palette():
    background = tensor_to_image(torch.full((1, 28, 28), -1.0), scale=1)
    ink = tensor_to_image(torch.ones(1, 28, 28), scale=1)
    assert background.getpixel((0, 0)) == (232, 229, 218)
    assert ink.getpixel((0, 0)) == (8, 42, 34)


def test_single_digit_grid_keeps_samples_beyond_one_hundred():
    hundred = single_digit_grid(torch.zeros(100, 1, 28, 28), columns=10, scale=1)
    hundred_and_one = single_digit_grid(torch.zeros(101, 1, 28, 28), columns=10, scale=1)
    assert hundred_and_one.width == hundred.width
    assert hundred_and_one.height > hundred.height


def test_single_digit_grid_supports_dense_zoom():
    images = torch.zeros(250, 1, 28, 28)
    full_size = single_digit_grid(images, columns=12, scale=4, gap=8)
    zoomed_out = single_digit_grid(images, columns=48, scale=1, gap=2)
    assert abs(zoomed_out.width - full_size.width) <= 8
    assert zoomed_out.height < full_size.height


def test_chrome_favicon_fallback_is_packaged():
    static = Path(__file__).parents[1] / "src" / "mnist_wgan" / "static"
    html = (static / "index.html").read_text()
    assert '/favicon.ico?v=2' in html
    assert html.count('class="digit-input"') == 2
    assert html.count('maxlength="1"') == 2
    assert "function bindSingleDigitInput" in html
    with Image.open(static / "favicon.ico") as icon:
        assert icon.format == "ICO"
        assert {(16, 16), (32, 32), (48, 48), (64, 64)} <= icon.info["sizes"]
