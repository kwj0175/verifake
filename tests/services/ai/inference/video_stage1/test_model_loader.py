# pyright: reportMissingImports=false

from __future__ import annotations

from pathlib import Path

import pytest

from services.ai.inference.video_stage1.model_loader import (
    load_detector_config,
    load_efficientnet_b4_detector,
)
from services.ai.pipelines.video_stage1.exceptions import Stage1UnavailableError


def test_load_detector_config_reads_local_efficientnet_yaml() -> None:
    config = load_detector_config()

    assert config["model_name"] == "efficientnetb4"
    assert config["resolution"] == 256
    assert config["mean"] == [0.5, 0.5, 0.5]
    assert config["std"] == [0.5, 0.5, 0.5]


def test_load_efficientnet_b4_detector_requires_existing_checkpoint() -> None:
    with pytest.raises(Stage1UnavailableError, match="checkpoint"):
        load_efficientnet_b4_detector(
            weights_path="/tmp/does-not-exist/model.pth",
        )


def test_load_efficientnet_b4_detector_loads_public_checkpoint_when_present() -> None:
    checkpoint_path = Path("services/ai/checkpoints/video/effnb4_best.pth")
    if not checkpoint_path.exists():
        pytest.skip("public checkpoint not downloaded")

    model, device, config = load_efficientnet_b4_detector(
        weights_path=checkpoint_path,
        config_path="services/ai/deepfakebench/training/config/detector/efficientnetb4.yaml",
        device="cpu",
    )

    assert type(model).__name__
    assert str(device) == "cpu"
    assert config["model_name"] == "efficientnetb4"
