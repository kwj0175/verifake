# pyright: reportMissingImports=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from services.ai.common.json_io import read_json
from services.ai.pipelines.video_stage1 import detect
from services.ai.pipelines.video_stage1 import preprocess
from services.ai.pipelines.video_stage1.detect import run_video_stage1_detection


cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def test_stage1_a_output_can_feed_stage1_b_detection(
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
            "score_range": [0.0, 1.0],
            "storage_root": str(storage_root),
            "preprocessing_output": "storage/jobs/{job_id}/metadata/preprocessing.json",
            "detection_output": "storage/jobs/{job_id}/output/detection.json",
            "final_output": "storage/jobs/{job_id}/output/result.json",
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
            "detector": {
                "framework": "DeepfakeBench",
                "model_name": "EfficientNet-B4",
                "score_field": "raw_fake_score",
                "batch_size": 16,
                "use_mock": True,
            },
            "segment_merge": {
                "score_threshold": 0.6,
                "max_gap_sec": 1.0,
                "min_segment_duration_sec": 0.5,
                "top_k": 5,
            },
            "video_score": {
                "aggregation_method": "topk_mean",
                "topk_frame_count": 10,
            },
        }

    def fake_normalize_video(source: Path, destination: Path) -> None:
        destination.write_bytes(source.read_bytes() + b"-normalized")

    def fake_probe_video(video_path: Path) -> dict[str, Any]:
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
    monkeypatch.setattr(detect, "load_stage1_config", fake_load_config)
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

    preprocessing = preprocess.run_video_stage1_preprocess(
        str(input_path),
        job_id="job_test_integration",
    )
    preprocessing_path = storage_root / "job_test_integration" / "metadata" / "preprocessing.json"

    detection = run_video_stage1_detection(str(preprocessing_path))
    result_path = storage_root / "job_test_integration" / "output" / "result.json"

    saved_preprocessing = read_json(preprocessing_path)
    saved_result = json.loads(result_path.read_text(encoding="utf-8"))

    assert preprocessing["job_id"] == "job_test_integration"
    assert saved_preprocessing["frames"][0]["faces"][0]["crop_path"].endswith("frame_000000_face_00.jpg")
    assert saved_preprocessing["input"]["internal_format"] == "mp4"
    assert saved_preprocessing["input"]["normalized_video_path"].endswith("normalized.mp4")
    assert detection["job_id"] == "job_test_integration"
    assert saved_result["job_id"] == "job_test_integration"
    assert 0.0 <= saved_result["detection"]["video_score"]["final_fake_score"] <= 1.0
