from pathlib import Path

import mnist_wgan.app as app_module


class FakeInferenceEngine:
    checkpoint_path = Path("fake.ckpt")
    device = "cpu"
    latent_dim = 128


def test_single_digit_limit_is_ten_thousand(monkeypatch):
    monkeypatch.setattr(
        app_module, "InferenceEngine", lambda *args, **kwargs: FakeInferenceEngine()
    )
    app = app_module.create_app()
    info_route = next(route for route in app.routes if route.path == "/api/info")
    digit_route = next(route for route in app.routes if route.path == "/api/digit")
    samples = next(field for field in digit_route.dependant.query_params if field.name == "samples")
    maximum = next(item.le for item in samples.field_info.metadata if hasattr(item, "le"))

    assert info_route.endpoint()["limits"]["single_digit_samples"] == 10_000
    assert maximum == 10_000
