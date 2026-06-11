from __future__ import annotations

from services.ai.pipelines.video_stage1 import detect
from services.ai.pipelines.video_stage1.config import load_stage1_config as load_stage1_config_from_config
from services.ai.pipelines.video_stage1.schemas import (
    DetectionOutput,
    ResultOutput,
    load_stage1_config,
)


def test_load_stage1_config_returns_stage1_b_defaults() -> None:
    config = load_stage1_config()

    assert config["schema_version"] == "video-stage1-v1"
    assert config["pipeline_stage"] == 1
    assert config["detector"]["framework"] == "DeepfakeBench"
    assert config["detector"]["model_name"] == "EfficientNet-B4"
    assert config["detector"]["batch_size"] == 16
    assert config["segment_merge"]["score_threshold"] == 0.6
    assert config["segment_merge"]["max_gap_sec"] == 1.0
    assert config["segment_merge"]["min_segment_duration_sec"] == 0.5
    assert config["segment_merge"]["top_k"] == 5
    assert config["video_score"]["aggregation_method"] == "topk_mean"
    assert config["video_score"]["topk_frame_count"] == 10


def test_detect_uses_shared_stage1_config_loader() -> None:
    assert detect.load_stage1_config is load_stage1_config_from_config


def test_detection_output_accepts_minimal_valid_payload() -> None:
    payload = {
        "schema_version": "video-stage1-v1",
        "job_id": "mock_job_001",
        "pipeline_stage": 1,
        "status": "success",
        "created_at": "2026-04-28T10:00:00+09:00",
        "detector": {
            "framework": "DeepfakeBench",
            "model_name": "EfficientNet-B4",
            "detector_type": "face_artifact_detector",
            "score_type": "fake_raw_score",
            "score_range": [0.0, 1.0],
        },
        "inference_summary": {
            "analyzed_face_crop_count": 1,
            "analyzed_frame_count": 1,
            "skipped_frame_count": 0,
            "inference_status": "success",
        },
        "face_scores": [
            {
                "face_id": "face_000000_00",
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "crop_path": "storage/jobs/mock_job_001/faces/frame_000000_face_00.jpg",
                "raw_fake_score": 0.5,
                "inference_success": True,
            }
        ],
        "frame_scores": [
            {
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "face_count": 1,
                "max_fake_score": 0.5,
                "avg_fake_score": 0.5,
                "score_source": "face_scores",
            }
        ],
        "segment_scores": [],
        "top_segments": [],
        "video_score": {
            "max_fake_score": 0.5,
            "topk_mean_fake_score": 0.5,
            "avg_fake_score": 0.5,
            "final_fake_score": 0.5,
            "aggregation_method": "topk_mean",
        },
        "errors": [],
    }

    result = DetectionOutput.model_validate(payload)

    assert result.job_id == "mock_job_001"
    assert result.face_scores[0].raw_fake_score == 0.5
    assert result.video_score.final_fake_score == 0.5


def test_detection_output_rejects_out_of_range_scores() -> None:
    payload = {
        "schema_version": "video-stage1-v1",
        "job_id": "mock_job_001",
        "pipeline_stage": 1,
        "status": "success",
        "created_at": "2026-04-28T10:00:00+09:00",
        "detector": {
            "framework": "DeepfakeBench",
            "model_name": "EfficientNet-B4",
            "detector_type": "face_artifact_detector",
            "score_type": "fake_raw_score",
            "score_range": [0.0, 1.0],
        },
        "inference_summary": {
            "analyzed_face_crop_count": 1,
            "analyzed_frame_count": 1,
            "skipped_frame_count": 0,
            "inference_status": "success",
        },
        "face_scores": [
            {
                "face_id": "face_000000_00",
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "crop_path": "storage/jobs/mock_job_001/faces/frame_000000_face_00.jpg",
                "raw_fake_score": 1.5,
                "inference_success": True,
            }
        ],
        "frame_scores": [],
        "segment_scores": [],
        "top_segments": [],
        "video_score": {
            "max_fake_score": 0.5,
            "topk_mean_fake_score": 0.5,
            "avg_fake_score": 0.5,
            "final_fake_score": 0.5,
            "aggregation_method": "topk_mean",
        },
        "errors": [],
    }

    try:
        DetectionOutput.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        assert "raw_fake_score" in str(exc)
    else:
        raise AssertionError("expected score validation failure")


def test_result_output_accepts_minimal_valid_payload() -> None:
    payload = {
        "schema_version": "video-stage1-v1",
        "job_id": "mock_job_001",
        "pipeline_stage": 1,
        "status": "success",
        "input": {
            "normalized_video_path": "storage/jobs/mock_job_001/input/normalized.mp4"
        },
        "video_metadata": {
            "width": 1280,
            "height": 720,
            "fps": 30.0,
            "duration_sec": 10.0,
            "sample_fps": 3.0,
        },
        "quality_metrics": {
            "face_detect_ratio": 0.9,
            "face_visibility_ratio": 0.88,
            "avg_face_size_ratio": 0.16,
            "blur_score": 0.18,
            "motion_blur_ratio": 0.05,
            "dark_frame_ratio": 0.02,
            "compression_artifact_score": 0.14,
        },
        "face_summary": {
            "human_face_detected": True,
            "multi_face_flag": False,
            "face_track_stability": 0.84,
        },
        "detection": {
            "detector": "DeepfakeBench + EfficientNet-B4",
            "video_score": {
                "final_fake_score": 0.5,
                "max_fake_score": 0.5,
                "aggregation_method": "topk_mean",
            },
            "top_segments": [],
        },
        "stage1_note": "quality_metrics are used for reliability/context, not as direct fake evidence.",
    }

    result = ResultOutput.model_validate(payload)

    assert result.job_id == "mock_job_001"
    assert result.detection.video_score.final_fake_score == 0.5
