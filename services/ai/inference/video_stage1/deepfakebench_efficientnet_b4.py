"""Mock-first EfficientNet-B4 inference wrapper for video Stage 1."""

from __future__ import annotations

from typing import Any

from services.ai.inference.video_stage1.model_loader import (
    load_efficientnet_b4_detector,
)
from services.ai.inference.video_stage1.transforms import (
    preprocess_face_crop,
    stack_face_tensors,
)


def predict_face_crops(
    face_items: list[dict[str, Any]],
    batch_size: int = 16,
    use_mock: bool = True,
    weights_path: str | None = None,
    device: str = "auto",
    config_path: str | None = None,
) -> list[dict[str, Any]]:
    """Return mock face-level fake scores for Stage 1 B.

    The `batch_size` argument is intentionally accepted now so the function
    signature already matches the later real-model integration point.
    """

    if use_mock:
        del batch_size, weights_path, device, config_path

        results: list[dict[str, Any]] = []
        for item in face_items:
            results.append(
                {
                    "face_id": item["face_id"],
                    "frame_index": item["frame_index"],
                    "timestamp_sec": item["timestamp_sec"],
                    "crop_path": item["crop_path"],
                    "raw_fake_score": 0.5,
                    "inference_success": True,
                }
            )
        return results

    model, runtime_device, detector_config = load_efficientnet_b4_detector(
        weights_path=weights_path or "",
        config_path=config_path,
        device=device,
    )

    resolution = int(detector_config.get("resolution", 256))
    mean = list(detector_config.get("mean", [0.5, 0.5, 0.5]))
    std = list(detector_config.get("std", [0.5, 0.5, 0.5]))

    results: list[dict[str, Any]] = []
    for batch_start in range(0, len(face_items), batch_size):
        batch_items = face_items[batch_start:batch_start + batch_size]
        batch_tensors = [
            preprocess_face_crop(
                item["crop_path"],
                resolution=resolution,
                mean=mean,
                std=std,
            )
            for item in batch_items
        ]
        batch_tensor = stack_face_tensors(batch_tensors).to(runtime_device)
        predictions = model({"image": batch_tensor}, inference=True)
        probabilities = predictions["prob"].detach().cpu().tolist()

        for item, probability in zip(batch_items, probabilities):
            results.append(
                {
                    "face_id": item["face_id"],
                    "frame_index": item["frame_index"],
                    "timestamp_sec": item["timestamp_sec"],
                    "crop_path": item["crop_path"],
                    "raw_fake_score": float(probability),
                    "inference_success": True,
                }
            )

    return results
