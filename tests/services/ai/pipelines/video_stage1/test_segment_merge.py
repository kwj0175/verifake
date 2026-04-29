from __future__ import annotations

from services.ai.pipelines.video_stage1.segment_merge import (
    merge_suspicious_frames,
    select_top_segments,
)


def test_merge_suspicious_frames_merges_frames_with_small_timestamp_gaps() -> None:
    frame_scores = [
        {"frame_index": 10, "timestamp_sec": 4.0, "face_count": 1, "max_fake_score": 0.7, "avg_fake_score": 0.7, "score_source": "face_scores"},
        {"frame_index": 11, "timestamp_sec": 4.5, "face_count": 1, "max_fake_score": 0.9, "avg_fake_score": 0.9, "score_source": "face_scores"},
        {"frame_index": 20, "timestamp_sec": 8.0, "face_count": 1, "max_fake_score": 0.8, "avg_fake_score": 0.8, "score_source": "face_scores"},
    ]

    segments = merge_suspicious_frames(
        frame_scores,
        threshold=0.6,
        max_gap_sec=1.0,
        min_segment_duration_sec=0.0,
    )

    assert segments == [
        {
            "segment_id": "seg_0001",
            "start_sec": 4.0,
            "end_sec": 4.5,
            "duration_sec": 0.5,
            "frame_count": 2,
            "max_fake_score": 0.9,
            "avg_fake_score": 0.8,
            "representative_frame_index": 11,
            "representative_frame_path": None,
        },
        {
            "segment_id": "seg_0002",
            "start_sec": 8.0,
            "end_sec": 8.0,
            "duration_sec": 0.0,
            "frame_count": 1,
            "max_fake_score": 0.8,
            "avg_fake_score": 0.8,
            "representative_frame_index": 20,
            "representative_frame_path": None,
        },
    ]


def test_merge_suspicious_frames_drops_too_short_segments() -> None:
    frame_scores = [
        {"frame_index": 1, "timestamp_sec": 1.0, "face_count": 1, "max_fake_score": 0.9, "avg_fake_score": 0.9, "score_source": "face_scores"},
        {"frame_index": 2, "timestamp_sec": 3.0, "face_count": 1, "max_fake_score": 0.95, "avg_fake_score": 0.95, "score_source": "face_scores"},
    ]

    segments = merge_suspicious_frames(
        frame_scores,
        threshold=0.6,
        max_gap_sec=1.0,
        min_segment_duration_sec=0.5,
    )

    assert segments == []


def test_select_top_segments_orders_by_segment_score_and_limits_count() -> None:
    segment_scores = [
        {"segment_id": "seg_0001", "start_sec": 1.0, "end_sec": 2.0, "duration_sec": 1.0, "frame_count": 2, "max_fake_score": 0.7, "avg_fake_score": 0.65, "representative_frame_index": 1, "representative_frame_path": "frames/frame_000001.jpg"},
        {"segment_id": "seg_0002", "start_sec": 3.0, "end_sec": 4.0, "duration_sec": 1.0, "frame_count": 2, "max_fake_score": 0.95, "avg_fake_score": 0.9, "representative_frame_index": 2, "representative_frame_path": "frames/frame_000002.jpg"},
        {"segment_id": "seg_0003", "start_sec": 5.0, "end_sec": 6.0, "duration_sec": 1.0, "frame_count": 2, "max_fake_score": 0.8, "avg_fake_score": 0.75, "representative_frame_index": 3, "representative_frame_path": "frames/frame_000003.jpg"},
    ]

    top_segments = select_top_segments(segment_scores, top_k=2)

    assert top_segments == [
        {
            "rank": 1,
            "segment_id": "seg_0002",
            "start_sec": 3.0,
            "end_sec": 4.0,
            "segment_score": 0.95,
            "reason": "high consecutive face artifact scores",
            "representative_frame_path": "frames/frame_000002.jpg",
        },
        {
            "rank": 2,
            "segment_id": "seg_0003",
            "start_sec": 5.0,
            "end_sec": 6.0,
            "segment_score": 0.8,
            "reason": "high consecutive face artifact scores",
            "representative_frame_path": "frames/frame_000003.jpg",
        },
    ]
