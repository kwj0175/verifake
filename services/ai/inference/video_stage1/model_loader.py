# pyright: reportMissingImports=false, reportMissingModuleSource=false

"""Model loading helpers for Stage 1 B EfficientNet-B4 inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from services.ai.pipelines.video_stage1.exceptions import Stage1UnavailableError


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "deepfakebench"
    / "training"
    / "config"
    / "detector"
    / "efficientnetb4.yaml"
)


def _load_yaml_runtime():
    try:
        import yaml
    except ImportError as exc:
        raise Stage1UnavailableError(
            "Stage1 detection requires PyYAML. Install services/ai/requirements.txt "
            "and services/backend/requirements-ai-stage1.txt before enabling real inference."
        ) from exc

    return yaml


def load_detector_config(config_path: str | Path | None = None) -> dict[str, Any]:
    yaml = _load_yaml_runtime()
    resolved_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not resolved_path.exists():
        raise Stage1UnavailableError(
            f"DeepfakeBench detector config not found: {resolved_path}"
        )

    with resolved_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _load_torch_runtime():
    try:
        import torch
    except ImportError as exc:
        raise Stage1UnavailableError(
            "Stage1 detection requires torch runtime dependencies. Install "
            "services/backend/requirements-ai-stage1.txt before enabling real inference."
        ) from exc

    return torch


def resolve_runtime_device(device: str = "auto"):
    torch = _load_torch_runtime()

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_runtime_inference_model(detector_config: dict[str, Any]):
    torch = _load_torch_runtime()

    from services.ai.deepfakebench.training.networks.efficientnetb4 import (
        EfficientNetB4,
    )

    class EfficientNetB4InferenceRuntime(torch.nn.Module):
        def __init__(self, runtime_config: dict[str, Any]) -> None:
            super().__init__()
            backbone_config = dict(runtime_config["backbone_config"])
            backbone_config["pretrained"] = False
            self.backbone = EfficientNetB4(backbone_config)

        def forward(self, data_dict: dict[str, Any], inference: bool = False) -> dict[str, Any]:
            del inference
            logits = self.backbone(data_dict["image"])
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            return {"cls": logits, "prob": probabilities}

    return EfficientNetB4InferenceRuntime(detector_config)


def load_efficientnet_b4_detector(
    weights_path: str | Path,
    config_path: str | Path | None = None,
    device: str = "auto",
):
    resolved_weights_path = Path(weights_path)
    if not resolved_weights_path.exists():
        raise Stage1UnavailableError(
            f"EfficientNet-B4 checkpoint not found: {resolved_weights_path}"
        )

    config = load_detector_config(config_path)
    torch = _load_torch_runtime()
    runtime_device = resolve_runtime_device(device)

    model = _build_runtime_inference_model(config).to(runtime_device)
    checkpoint = torch.load(resolved_weights_path, map_location=runtime_device)
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    model.load_state_dict(state_dict or checkpoint, strict=True)
    model.eval()

    return model, runtime_device, config
