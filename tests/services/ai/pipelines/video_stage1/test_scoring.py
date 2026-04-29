from __future__ import annotations

from services.ai.pipelines.video_stage1.scoring import (
    aggregate_face_scores_to_frame_scores,
)


def test_aggregate_face_scores_to_frame_scores_groups_by_frame() -> None:
    face_scores = [
        {
            "face_id": "face_000001_00",
            "frame_index": 1,
            "timestamp_sec": 0.333,
            "crop_path": "faces/frame_000001_face_00.jpg",
            "raw_fake_score": 0.2,
            "inference_success": True,
        },
        {
            "face_id": "face_000001_01",
            "frame_index": 1,
            "timestamp_sec": 0.333,
            "crop_path": "faces/frame_000001_face_01.jpg",
            "raw_fake_score": 0.8,
            "inference_success": True,
        },
        {
            "face_id": "face_000002_00",
            "frame_index": 2,
            "timestamp_sec": 0.666,
            "crop_path": "faces/frame_000002_face_00.jpg",
            "raw_fake_score": 0.4,
            "inference_success": True,
        },
    ]

    result = aggregate_face_scores_to_frame_scores(face_scores)

    assert result == [
        {
            "frame_index": 1,
            "timestamp_sec": 0.333,
            "face_count": 2,
            "max_fake_score": 0.8,
            "avg_fake_score": 0.5,
            "score_source": "face_scores",
        },
        {
            "frame_index": 2,
            "timestamp_sec": 0.666,
            "face_count": 1,
            "max_fake_score": 0.4,
            "avg_fake_score": 0.4,
            "score_source": "face_scores",
        },
    ]


def test_aggregate_face_scores_to_frame_scores_returns_empty_list_for_empty_input() -> None:
    assert aggregate_face_scores_to_frame_scores([]) == []
