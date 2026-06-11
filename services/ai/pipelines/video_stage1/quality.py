from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _read_frame_image(frame_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(frame_path))
    if image is None:
        raise ValueError(f"Failed to read frame image: {frame_path}")
    return image


def _calculate_frame_blur_score(gray_frame: np.ndarray, reference: float) -> float:
    variance = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    normalized = 1.0 - min(1.0, variance / max(reference, 1e-6))
    return _clamp_unit(float(normalized))


def _calculate_compression_artifact_score(gray_frame: np.ndarray, block_size: int) -> float:
    if gray_frame.shape[0] <= block_size or gray_frame.shape[1] <= block_size:
        return 0.0

    vertical_left = gray_frame[:, block_size - 1 :: block_size].astype(np.float32)
    vertical_right = gray_frame[:, block_size :: block_size].astype(np.float32)
    vertical_count = min(vertical_left.shape[1], vertical_right.shape[1])
    vertical_edges = (
        np.abs(vertical_left[:, :vertical_count] - vertical_right[:, :vertical_count])
        if vertical_count > 0
        else np.array([], dtype=np.float32)
    )

    horizontal_top = gray_frame[block_size - 1 :: block_size, :].astype(np.float32)
    horizontal_bottom = gray_frame[block_size :: block_size, :].astype(np.float32)
    horizontal_count = min(horizontal_top.shape[0], horizontal_bottom.shape[0])
    horizontal_edges = (
        np.abs(horizontal_top[:horizontal_count, :] - horizontal_bottom[:horizontal_count, :])
        if horizontal_count > 0
        else np.array([], dtype=np.float32)
    )

    vertical_score = float(vertical_edges.mean() / 255.0) if vertical_edges.size else 0.0
    horizontal_score = float(horizontal_edges.mean() / 255.0) if horizontal_edges.size else 0.0
    return _clamp_unit((vertical_score + horizontal_score) / 2.0)


def calculate_quality_metrics(
    video_metadata: dict[str, Any],
    frames: list[dict[str, Any]],
    visibility_min_confidence: float = 0.9,
    visibility_min_area_ratio: float = 0.05,
    blur_variance_reference: float = 400.0,
    motion_blur_threshold: float = 0.6,
    dark_frame_mean_threshold: float = 40.0,
    compression_block_size: int = 8,
) -> dict[str, float]:
    total_frames = len(frames)
    if total_frames == 0:
        return {
            "face_detect_ratio": 0.0,
            "face_visibility_ratio": 0.0,
            "avg_face_size_ratio": 0.0,
            "min_face_size_ratio": 0.0,
            "max_face_size_ratio": 0.0,
            "blur_score": 0.0,
            "motion_blur_ratio": 0.0,
            "dark_frame_ratio": 0.0,
            "compression_artifact_score": 0.0,
        }

    detected_frames = 0
    visible_frames = 0
    dark_frames = 0
    motion_blur_frames = 0
    blur_scores: list[float] = []
    compression_scores: list[float] = []
    face_area_ratios: list[float] = []

    for frame in frames:
        faces = frame.get("faces", [])
        if faces:
            detected_frames += 1

        if any(
            face["detection_confidence"] >= visibility_min_confidence
            and face["bbox_area_ratio"] >= visibility_min_area_ratio
            for face in faces
        ):
            visible_frames += 1

        face_area_ratios.extend(face["bbox_area_ratio"] for face in faces)

        image = _read_frame_image(frame["frame_path"])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = _calculate_frame_blur_score(gray, blur_variance_reference)
        blur_scores.append(blur_score)
        compression_scores.append(
            _calculate_compression_artifact_score(gray, compression_block_size)
        )

        if blur_score >= motion_blur_threshold:
            motion_blur_frames += 1

        if float(gray.mean()) <= dark_frame_mean_threshold:
            dark_frames += 1

    avg_face_size_ratio = float(np.mean(face_area_ratios)) if face_area_ratios else 0.0
    min_face_size_ratio = float(np.min(face_area_ratios)) if face_area_ratios else 0.0
    max_face_size_ratio = float(np.max(face_area_ratios)) if face_area_ratios else 0.0

    return {
        "face_detect_ratio": round(_clamp_unit(_safe_ratio(detected_frames, total_frames)), 4),
        "face_visibility_ratio": round(_clamp_unit(_safe_ratio(visible_frames, total_frames)), 4),
        "avg_face_size_ratio": round(_clamp_unit(avg_face_size_ratio), 4),
        "min_face_size_ratio": round(_clamp_unit(min_face_size_ratio), 4),
        "max_face_size_ratio": round(_clamp_unit(max_face_size_ratio), 4),
        "blur_score": round(_clamp_unit(float(np.mean(blur_scores))), 4),
        "motion_blur_ratio": round(_clamp_unit(_safe_ratio(motion_blur_frames, total_frames)), 4),
        "dark_frame_ratio": round(_clamp_unit(_safe_ratio(dark_frames, total_frames)), 4),
        "compression_artifact_score": round(_clamp_unit(float(np.mean(compression_scores))), 4),
    }


def calculate_face_summary(
    frames: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> dict[str, bool | int | float]:
    total_frames = len(frames)
    face_counts = [len(frame.get("faces", [])) for frame in frames]
    human_face_detected = any(count > 0 for count in face_counts)
    failed_frame_count = sum(1 for count in face_counts if count == 0)
    max_face_count = max(face_counts, default=0)
    avg_face_count = _safe_ratio(sum(face_counts), total_frames)

    frame_diagonal = sqrt((frame_width ** 2) + (frame_height ** 2)) if frame_width > 0 and frame_height > 0 else 0.0
    tracked_faces = []
    for frame in frames:
        faces = frame.get("faces", [])
        if not faces:
            continue
        tracked_faces.append(max(faces, key=lambda face: face["bbox_area_ratio"]))

    if not tracked_faces:
        stability = 0.0
    elif len(tracked_faces) == 1:
        stability = 1.0
    else:
        penalties: list[float] = []
        for previous, current in zip(tracked_faces, tracked_faces[1:]):
            px1, py1, px2, py2 = previous["bbox"]
            cx1, cy1, cx2, cy2 = current["bbox"]
            previous_center = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)
            current_center = ((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0)
            movement = sqrt(
                (previous_center[0] - current_center[0]) ** 2
                + (previous_center[1] - current_center[1]) ** 2
            )
            movement_penalty = _safe_ratio(movement, frame_diagonal)
            size_penalty = abs(
                previous["bbox_area_ratio"] - current["bbox_area_ratio"]
            )
            penalties.append(min(1.0, movement_penalty + size_penalty))

        stability = 1.0 - float(np.mean(penalties))

    return {
        "human_face_detected": human_face_detected,
        "face_detect_failed_frame_count": failed_frame_count,
        "max_face_count_per_frame": max_face_count,
        "avg_face_count_per_frame": round(avg_face_count, 4),
        "multi_face_flag": max_face_count >= 2,
        "face_track_stability": round(_clamp_unit(stability), 4),
    }
