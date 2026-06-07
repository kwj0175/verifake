"""Video-level summary builders for Stage 1 B."""

from __future__ import annotations

from math import sqrt
from typing import Any


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float], average: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return sqrt(variance)


def _select_final_score(
    aggregation_method: str,
    *,
    max_score: float,
    topk_mean_score: float,
    average_score: float,
) -> float:
    if aggregation_method == "topk_mean":
        return topk_mean_score
    if aggregation_method == "avg":
        return average_score
    if aggregation_method == "max":
        return max_score
    raise ValueError(
        "Unsupported video score aggregation_method: "
        f"{aggregation_method!r}. Expected one of: topk_mean, avg, max."
    )


def build_video_score(
    frame_scores: list[dict[str, Any]],
    segment_scores: list[dict[str, Any]],
    topk_frame_count: int = 10,
    aggregation_method: str = "topk_mean",
    score_threshold: float = 0.6,
) -> dict[str, Any]:
    del segment_scores

    if topk_frame_count < 1:
        raise ValueError("topk_frame_count must be greater than or equal to 1.")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between 0.0 and 1.0.")

    if not frame_scores:
        final_score = _select_final_score(
            aggregation_method,
            max_score=0.0,
            topk_mean_score=0.0,
            average_score=0.0,
        )
        return {
            "max_fake_score": 0.0,
            "topk_mean_fake_score": 0.0,
            "avg_fake_score": 0.0,
            "final_fake_score": final_score,
            "aggregation_method": aggregation_method,
            "score_threshold": score_threshold,
            "analyzed_frame_count": 0,
            "suspicious_frame_count": 0,
            "suspicious_frame_ratio": 0.0,
            "score_std": 0.0,
        }

    ordered_scores = sorted(
        (float(item["max_fake_score"]) for item in frame_scores),
        reverse=True,
    )
    top_scores = ordered_scores[:topk_frame_count]
    average_score = _mean(ordered_scores)
    topk_mean_score = _mean(top_scores)
    max_score = ordered_scores[0]
    final_score = _select_final_score(
        aggregation_method,
        max_score=max_score,
        topk_mean_score=topk_mean_score,
        average_score=average_score,
    )
    suspicious_frame_count = sum(
        1 for score in ordered_scores if score >= score_threshold
    )

    return {
        "max_fake_score": max_score,
        "topk_mean_fake_score": topk_mean_score,
        "avg_fake_score": average_score,
        "final_fake_score": final_score,
        "aggregation_method": aggregation_method,
        "score_threshold": score_threshold,
        "analyzed_frame_count": len(ordered_scores),
        "suspicious_frame_count": suspicious_frame_count,
        "suspicious_frame_ratio": suspicious_frame_count / len(ordered_scores),
        "score_std": _std(ordered_scores, average_score),
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
