"""Score aggregation helpers for video Stage 1 B."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def aggregate_face_scores_to_frame_scores(
    face_scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_scores: dict[tuple[int, float], list[float]] = defaultdict(list)
    for face_score in face_scores:
        key = (face_score["frame_index"], face_score["timestamp_sec"])
        grouped_scores[key].append(face_score["raw_fake_score"])

    frame_scores: list[dict[str, Any]] = []
    for (frame_index, timestamp_sec), scores in sorted(grouped_scores.items()):
        frame_scores.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "face_count": len(scores),
                "max_fake_score": max(scores),
                "avg_fake_score": sum(scores) / len(scores),
                "score_source": "face_scores",
            }
        )

    return frame_scores
