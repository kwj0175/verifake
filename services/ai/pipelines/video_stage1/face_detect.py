from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from services.ai.pipelines.video_stage1.exceptions import Stage1UnavailableError


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise Stage1UnavailableError(
            "Stage1 preprocessing requires the optional AI runtime. "
            "Install services/backend/requirements-ai-stage1.txt before running face detection."
        ) from exc

    return cv2


def _load_retinaface():
    try:
        from retinaface import RetinaFace
    except Exception as exc:
        raise Stage1UnavailableError(
            "Stage1 preprocessing requires a working retina-face/TensorFlow runtime. "
            "Install services/backend/requirements-ai-stage1.txt and set TF_USE_LEGACY_KERAS=1 before starting the API."
        ) from exc

    return RetinaFace


def _should_use_retinaface() -> bool:
    detector = os.getenv("VERIFAKE_FACE_DETECTOR", "opencv").strip().lower()
    return detector in {"retinaface", "retina-face"}


def _run_retinaface_detect_faces(RetinaFace, image: object, confidence_threshold: float):
    return RetinaFace.detect_faces(image, threshold=confidence_threshold)


def _detect_faces_with_haar(cv2, image: object) -> list[dict[str, Any]]:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(str(cascade_path))
    detections = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(48, 48),
    )
    return [
        {
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "score": 0.5,
        }
        for x, y, w, h in detections
    ]


def _center_crop_detection(width: int, height: int) -> dict[str, Any]:
    crop_size = int(min(width, height) * 0.8)
    x1 = max(0, (width - crop_size) // 2)
    y1 = max(0, (height - crop_size) // 2)
    return {
        "bbox": [x1, y1, min(width, x1 + crop_size), min(height, y1 + crop_size)],
        "score": 0.0,
    }


def _normalize_retinaface_results(detections: object) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    if isinstance(detections, dict):
        if "facial_area" in detections:
            candidate_items = [detections]
        else:
            candidate_items = [value for value in detections.values() if isinstance(value, dict)]
    elif isinstance(detections, list):
        candidate_items = [value for value in detections if isinstance(value, dict)]
    else:
        candidate_items = []

    for item in candidate_items:
        facial_area = item.get("facial_area") or item.get("bbox")
        if not facial_area or len(facial_area) < 4:
            continue

        normalized.append(
            {
                "bbox": [int(facial_area[0]), int(facial_area[1]), int(facial_area[2]), int(facial_area[3])],
                "score": float(item.get("score") or item.get("confidence") or 0.0),
            }
        )

    return normalized


def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[int] | None:
    x1 = max(0, min(width - 1, bbox[0]))
    y1 = max(0, min(height - 1, bbox[1]))
    x2 = max(0, min(width, bbox[2]))
    y2 = max(0, min(height, bbox[3]))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def detect_and_crop_faces(
    frames: list[dict[str, Any]],
    faces_dir: str | Path,
    confidence_threshold: float = 0.9,
    max_faces_per_frame: int = 5,
) -> list[dict[str, Any]]:
    cv2 = _load_cv2()
    if _should_use_retinaface():
        try:
            RetinaFace = _load_retinaface()
        except Stage1UnavailableError:
            RetinaFace = None
    else:
        RetinaFace = None
    output_dir = Path(faces_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        image = cv2.imread(frame["frame_path"])
        if image is None:
            frame["face_count"] = 0
            frame["faces"] = []
            continue

        frame_height, frame_width = image.shape[:2]
        detections: list[dict[str, Any]] = []
        if RetinaFace is not None:
            try:
                raw_detections = _run_retinaface_detect_faces(RetinaFace, image, confidence_threshold)
                detections = _normalize_retinaface_results(raw_detections)
            except Exception:
                RetinaFace = None

        if not detections:
            detections = _detect_faces_with_haar(cv2, image)
        if not detections:
            detections = [_center_crop_detection(frame_width, frame_height)]

        detections.sort(
            key=lambda detection: (
                (detection["bbox"][2] - detection["bbox"][0])
                * (detection["bbox"][3] - detection["bbox"][1])
            ),
            reverse=True,
        )

        face_entries: list[dict[str, Any]] = []
        for face_index, detection in enumerate(detections[:max_faces_per_frame]):
            bbox = _clamp_bbox(detection["bbox"], frame_width, frame_height)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_path = output_dir / f"frame_{frame['frame_index']:06d}_face_{face_index:02d}.jpg"
            written = cv2.imwrite(str(crop_path), crop)
            if not written:
                raise ValueError(f"Failed to write face crop: {crop_path}")

            bbox_area_ratio = ((x2 - x1) * (y2 - y1)) / float(frame_width * frame_height)
            face_entries.append(
                {
                    "face_id": f"face_{frame['frame_index']:06d}_{face_index:02d}",
                    "face_index": face_index,
                    "detected": True,
                    "bbox": bbox,
                    "bbox_area_ratio": round(max(0.0, min(1.0, bbox_area_ratio)), 4),
                    "detection_confidence": round(max(0.0, min(1.0, detection["score"])), 4),
                    "crop_path": crop_path.as_posix(),
                }
            )

        frame["face_count"] = len(face_entries)
        frame["faces"] = face_entries

    return frames
