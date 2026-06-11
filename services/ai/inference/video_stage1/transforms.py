# pyright: reportMissingImports=false

"""Image preprocessing helpers for Stage 1 B EfficientNet-B4 inference."""

from __future__ import annotations

from pathlib import Path

from services.ai.pipelines.video_stage1.exceptions import Stage1UnavailableError


def _load_transform_runtime():
    try:
        import cv2
        import numpy as np
        import torch
    except ImportError as exc:
        raise Stage1UnavailableError(
            "Stage1 detection requires cv2/numpy/torch runtime dependencies. "
            "Install services/backend/requirements-ai-stage1.txt before enabling real inference."
        ) from exc

    return cv2, np, torch


def preprocess_face_crop(
    crop_path: str | Path,
    resolution: int,
    mean: list[float],
    std: list[float],
):
    file_path = Path(crop_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Face crop not found: {file_path}")

    cv2, np, torch = _load_transform_runtime()

    image = cv2.imread(str(file_path))
    if image is None:
        raise ValueError(f"Failed to read face crop: {file_path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(rgb_image, (resolution, resolution))
    float_image = resized_image.astype(np.float32) / 255.0
    normalized_image = (float_image - np.array(mean, dtype=np.float32)) / np.array(
        std,
        dtype=np.float32,
    )
    tensor = torch.from_numpy(normalized_image).permute(2, 0, 1).float()
    return tensor


def stack_face_tensors(face_tensors: list[object]):
    _, _, torch = _load_transform_runtime()
    return torch.stack(face_tensors, dim=0)
