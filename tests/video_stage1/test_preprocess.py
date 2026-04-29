from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from services.ai.common.json_io import read_json


cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
preprocess = importlib.import_module("services.ai.pipelines.video_stage1.preprocess")


def test_run_video_stage1_preprocess_writes_preprocessing_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "sample.mov"
    input_path.write_bytes(b"fake-input")
    storage_root = tmp_path / "storage" / "jobs"

    def fake_load_config() -> dict[str, Any]:
        return {
            "schema_version": "video-stage1-v1",
            "pipeline_stage": 1,
            "internal_video_format": "mp4",
            "sample_fps": 3.0,
            "max_video_duration_sec": 300,
            "storage_root": str(storage_root),
            "face_detection": {
                "confidence_threshold": 0.9,
                "max_faces_per_frame": 5,
            },
            "quality": {
                "visibility_min_confidence": 0.9,
                "visibility_min_area_ratio": 0.05,
                "blur_variance_reference": 400.0,
                "motion_blur_threshold": 0.6,
                "dark_frame_mean_threshold": 40.0,
                "compression_block_size": 8,
            },
        }

    def fake_normalize_video(source: Path, destination: Path) -> None:
        destination.write_bytes(source.read_bytes() + b"-normalized")

    def fake_probe_video(video_path: Path) -> dict[str, Any]:
        assert video_path.exists()
        return {
            "width": 100,
            "height": 100,
            "fps": 30.0,
            "duration_sec": 2.0,
            "total_frame_count": 60,
        }

    def fake_sample_frames(
        video_path: Path,
        frames_dir: Path,
        sample_fps: float,
    ) -> list[dict[str, Any]]:
        frame_path = frames_dir / "frame_000000.jpg"
        image = np.full((100, 100, 3), 180, dtype=np.uint8)
        cv2.imwrite(str(frame_path), image)
        return [
            {
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "frame_path": frame_path.as_posix(),
                "face_count": 0,
                "faces": [],
            }
        ]

    def fake_detect_and_crop_faces(
        frames: list[dict[str, Any]],
        faces_dir: Path,
        confidence_threshold: float,
        max_faces_per_frame: int,
    ) -> list[dict[str, Any]]:
        crop_path = faces_dir / "frame_000000_face_00.jpg"
        image = np.full((32, 32, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(crop_path), image)
        frames[0]["faces"] = [
            {
                "face_id": "face_000000_00",
                "face_index": 0,
                "detected": True,
                "bbox": [10, 10, 60, 60],
                "bbox_area_ratio": 0.25,
                "detection_confidence": 0.98,
                "crop_path": crop_path.as_posix(),
            }
        ]
        frames[0]["face_count"] = 1
        return frames

    def fake_calculate_face_summary(
        frames: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> dict[str, Any]:
        return {
            "human_face_detected": True,
            "face_detect_failed_frame_count": 0,
            "max_face_count_per_frame": 1,
            "avg_face_count_per_frame": 1.0,
            "multi_face_flag": False,
            "face_track_stability": 1.0,
        }

    def fake_calculate_quality_metrics(
        video_metadata: dict[str, Any],
        frames: list[dict[str, Any]],
        visibility_min_confidence: float,
        visibility_min_area_ratio: float,
        blur_variance_reference: float,
        motion_blur_threshold: float,
        dark_frame_mean_threshold: float,
        compression_block_size: int,
    ) -> dict[str, float]:
        return {
            "face_detect_ratio": 1.0,
            "face_visibility_ratio": 1.0,
            "avg_face_size_ratio": 0.25,
            "min_face_size_ratio": 0.25,
            "max_face_size_ratio": 0.25,
            "blur_score": 0.1,
            "motion_blur_ratio": 0.0,
            "dark_frame_ratio": 0.0,
            "compression_artifact_score": 0.1,
        }

    monkeypatch.setattr(preprocess, "load_stage1_config", fake_load_config)
    monkeypatch.setattr(preprocess, "_normalize_video", fake_normalize_video)
    monkeypatch.setattr(
        preprocess,
        "_load_stage1_runtime_components",
        lambda: (
            fake_probe_video,
            fake_detect_and_crop_faces,
            fake_sample_frames,
            fake_calculate_face_summary,
            fake_calculate_quality_metrics,
        ),
    )

    result = preprocess.run_video_stage1_preprocess(str(input_path), job_id="job_test_002")
    output_path = storage_root / "job_test_002" / "metadata" / "preprocessing.json"
    saved = read_json(output_path)

    assert result["job_id"] == "job_test_002"
    assert result["status"] == "success"
    assert output_path.exists()
    assert saved["job_id"] == "job_test_002"
    assert saved["frames"][0]["faces"][0]["crop_path"].endswith("frame_000000_face_00.jpg")
