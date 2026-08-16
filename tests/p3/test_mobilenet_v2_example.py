from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def _example_module():
    root = Path(__file__).resolve().parents[2]
    source = root / "examples/mobilenet_v2_cifar10/run.py"
    spec = importlib.util.spec_from_file_location("bakenn_mobilenet_example", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_representative_model_is_quarter_width_cifar10_mobilenet_v2() -> None:
    module = _example_module()
    model = module.mobilenet_v2_quarter().eval()
    output = model(torch.zeros(1, 3, 32, 32))

    assert tuple(output.shape) == (1, 10)
    assert model.features[0][0].out_channels == 8
    assert model.last_channel == 1280
    assert sum(parameter.numel() for parameter in model.parameters()) < 500_000
