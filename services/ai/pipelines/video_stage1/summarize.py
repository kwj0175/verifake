"""Video-level summary builders for Stage 1 B."""

from __future__ import annotations

from typing import Any


def build_video_score(
    frame_scores: list[dict[str, Any]],
    segment_scores: list[dict[str, Any]],
    topk_frame_count: int = 10,
) -> dict[str, Any]:
    del segment_scores

    if not frame_scores:
        return {
            "max_fake_score": 0.0,
            "topk_mean_fake_score": 0.0,
            "avg_fake_score": 0.0,
            "final_fake_score": 0.0,
            "aggregation_method": "topk_mean",
        }

    ordered_scores = sorted(
        (item["max_fake_score"] for item in frame_scores),
        reverse=True,
    )
    top_scores = ordered_scores[:topk_frame_count]
    average_score = sum(ordered_scores) / len(ordered_scores)
    topk_mean_score = sum(top_scores) / len(top_scores)

    return {
        "max_fake_score": ordered_scores[0],
        "topk_mean_fake_score": topk_mean_score,
        "avg_fake_score": average_score,
        "final_fake_score": topk_mean_score,
        "aggregation_method": "topk_mean",
    }


def build_final_result(
    preprocessing: dict[str, Any],
    detection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": preprocessing["schema_version"],
        "job_id": preprocessing["job_id"],
        "pipeline_stage": preprocessing["pipeline_stage"],
        "status": detection["status"],
        "input": {
            "normalized_video_path": preprocessing["input"][
                "normalized_video_path"
            ]
        },
        "video_metadata": preprocessing["video_metadata"],
        "quality_metrics": preprocessing["quality_metrics"],
        "face_summary": preprocessing["face_summary"],
        "detection": {
            "detector": "DeepfakeBench + EfficientNet-B4",
            "video_score": detection["video_score"],
            "top_segments": detection["top_segments"],
        },
        "stage1_note": "quality_metrics are used for reliability/context, not as direct fake evidence.",
    }
