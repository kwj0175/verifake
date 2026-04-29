from __future__ import annotations

from math import isclose

from services.ai.pipelines.video_stage1.summarize import (
    build_final_result,
    build_video_score,
)


def test_build_video_score_uses_topk_mean() -> None:
    frame_scores = [
        {"frame_index": 0, "timestamp_sec": 0.0, "face_count": 1, "max_fake_score": 0.1, "avg_fake_score": 0.1, "score_source": "face_scores"},
        {"frame_index": 1, "timestamp_sec": 0.333, "face_count": 1, "max_fake_score": 0.8, "avg_fake_score": 0.8, "score_source": "face_scores"},
        {"frame_index": 2, "timestamp_sec": 0.666, "face_count": 1, "max_fake_score": 0.5, "avg_fake_score": 0.5, "score_source": "face_scores"},
    ]

    result = build_video_score(frame_scores, [], topk_frame_count=2)

    assert result["max_fake_score"] == 0.8
    assert result["topk_mean_fake_score"] == 0.65
    assert isclose(result["avg_fake_score"], 0.4666666666666666)
    assert result["final_fake_score"] == 0.65
    assert result["aggregation_method"] == "topk_mean"


def test_build_video_score_returns_zeroes_for_empty_frame_scores() -> None:
    result = build_video_score([], [], topk_frame_count=10)

    assert result == {
        "max_fake_score": 0.0,
        "topk_mean_fake_score": 0.0,
        "avg_fake_score": 0.0,
        "final_fake_score": 0.0,
        "aggregation_method": "topk_mean",
    }


def test_build_final_result_projects_preprocessing_and_detection() -> None:
    preprocessing = {
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
    }
    detection = {
        "status": "success",
        "video_score": {
            "final_fake_score": 0.65,
            "max_fake_score": 0.8,
            "aggregation_method": "topk_mean",
        },
        "top_segments": [
            {
                "rank": 1,
                "start_sec": 3.0,
                "end_sec": 4.0,
                "segment_score": 0.8,
                "representative_frame_path": "frames/frame_000010.jpg",
            }
        ],
    }

    result = build_final_result(preprocessing, detection)

    assert result == {
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
                "final_fake_score": 0.65,
                "max_fake_score": 0.8,
                "aggregation_method": "topk_mean",
            },
            "top_segments": [
                {
                    "rank": 1,
                    "start_sec": 3.0,
                    "end_sec": 4.0,
                    "segment_score": 0.8,
                    "representative_frame_path": "frames/frame_000010.jpg",
                }
            ],
        },
        "stage1_note": "quality_metrics are used for reliability/context, not as direct fake evidence.",
    }
