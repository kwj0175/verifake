"""Suspicious frame merge helpers for video Stage 1 B."""

from __future__ import annotations

from typing import Any


def merge_suspicious_frames(
    frame_scores: list[dict[str, Any]],
    threshold: float = 0.6,
    max_gap_sec: float = 1.0,
    min_segment_duration_sec: float = 0.5,
) -> list[dict[str, Any]]:
    suspicious_frames = [
        frame_score
        for frame_score in sorted(frame_scores, key=lambda item: item["timestamp_sec"])
        if frame_score["max_fake_score"] >= threshold
    ]
    if not suspicious_frames:
        return []

    segments: list[list[dict[str, Any]]] = [[suspicious_frames[0]]]
    for frame_score in suspicious_frames[1:]:
        previous_frame = segments[-1][-1]
        if frame_score["timestamp_sec"] - previous_frame["timestamp_sec"] <= max_gap_sec:
            segments[-1].append(frame_score)
        else:
            segments.append([frame_score])

    merged_segments: list[dict[str, Any]] = []
    for index, segment_frames in enumerate(segments, start=1):
        start_sec = segment_frames[0]["timestamp_sec"]
        end_sec = segment_frames[-1]["timestamp_sec"]
        duration_sec = end_sec - start_sec
        if duration_sec < min_segment_duration_sec:
            continue

        representative_frame = max(
            segment_frames,
            key=lambda item: item["max_fake_score"],
        )
        scores = [item["max_fake_score"] for item in segment_frames]
        merged_segments.append(
            {
                "segment_id": f"seg_{index:04d}",
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration_sec,
                "frame_count": len(segment_frames),
                "max_fake_score": max(scores),
                "avg_fake_score": sum(scores) / len(scores),
                "representative_frame_index": representative_frame["frame_index"],
                "representative_frame_path": representative_frame.get(
                    "frame_path"
                ),
            }
        )

    return merged_segments


def select_top_segments(
    segment_scores: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    ranked_segments = sorted(
        segment_scores,
        key=lambda item: item["max_fake_score"],
        reverse=True,
    )[:top_k]

    top_segments: list[dict[str, Any]] = []
    for rank, segment in enumerate(ranked_segments, start=1):
        top_segments.append(
            {
                "rank": rank,
                "segment_id": segment["segment_id"],
                "start_sec": segment["start_sec"],
                "end_sec": segment["end_sec"],
                "segment_score": segment["max_fake_score"],
                "reason": "high consecutive face artifact scores",
                "representative_frame_path": segment.get(
                    "representative_frame_path"
                ),
            }
        )

    return top_segments
