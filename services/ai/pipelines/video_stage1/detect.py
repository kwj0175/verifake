"""CLI and orchestration entrypoint for video Stage 1 B detection."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from services.ai.inference.video_stage1.deepfakebench_efficientnet_b4 import (
    predict_face_crops,
)
from services.ai.pipelines.video_stage1.config import load_stage1_config
from services.ai.pipelines.video_stage1.scoring import (
    aggregate_face_scores_to_frame_scores,
)
from services.ai.pipelines.video_stage1.schemas import (
    DetectionOutput,
    ResultOutput,
)
from services.ai.pipelines.video_stage1.segment_merge import (
    merge_suspicious_frames,
    select_top_segments,
)
from services.ai.pipelines.video_stage1.summarize import (
    build_final_result,
    build_video_score,
)


KST = timezone(timedelta(hours=9))


def _load_json(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _write_json(file_path: Path, payload: dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)


def _extract_face_items(preprocessing: dict[str, Any]) -> list[dict[str, Any]]:
    face_items: list[dict[str, Any]] = []
    for frame in preprocessing.get("frames", []):
        for face in frame.get("faces", []):
            face_items.append(
                {
                    "face_id": face["face_id"],
                    "frame_index": frame["frame_index"],
                    "timestamp_sec": frame["timestamp_sec"],
                    "crop_path": face["crop_path"],
                }
            )
    return face_items


def run_video_stage1_detection(preprocessing_json_path: str) -> dict[str, Any]:
    config = load_stage1_config()
    preprocessing_path = Path(preprocessing_json_path)
    preprocessing = _load_json(preprocessing_path)
    job_id = preprocessing["job_id"]

    face_items = _extract_face_items(preprocessing)
    face_scores = predict_face_crops(
        face_items,
        batch_size=config["detector"]["batch_size"],
        use_mock=bool(config["detector"].get("use_mock", True)),
        weights_path=config["detector"].get("weights_path"),
        device=str(config["detector"].get("device", "auto")),
        config_path=config["detector"].get("config_path"),
    )

    frame_scores = aggregate_face_scores_to_frame_scores(face_scores)
    frame_path_by_index = {
        frame["frame_index"]: frame["frame_path"]
        for frame in preprocessing.get("frames", [])
    }
    for frame_score in frame_scores:
        frame_score["frame_path"] = frame_path_by_index.get(frame_score["frame_index"])

    segment_scores = merge_suspicious_frames(
        frame_scores,
        threshold=config["segment_merge"]["score_threshold"],
        max_gap_sec=config["segment_merge"]["max_gap_sec"],
        min_segment_duration_sec=config["segment_merge"]["min_segment_duration_sec"],
    )
    top_segments = select_top_segments(
        segment_scores,
        top_k=config["segment_merge"]["top_k"],
    )
    video_score = build_video_score(
        frame_scores,
        segment_scores,
        topk_frame_count=config["video_score"]["topk_frame_count"],
    )

    analyzed_frame_count = len({item["frame_index"] for item in face_scores})
    total_frame_count = len(preprocessing.get("frames", []))
    detection = {
        "schema_version": config["schema_version"],
        "job_id": job_id,
        "pipeline_stage": config["pipeline_stage"],
        "status": "success",
        "created_at": datetime.now(KST).isoformat(),
        "detector": {
            "framework": config["detector"]["framework"],
            "model_name": config["detector"]["model_name"],
            "detector_type": "face_artifact_detector",
            "score_type": "fake_raw_score",
            "score_range": config["score_range"],
        },
        "inference_summary": {
            "analyzed_face_crop_count": len(face_scores),
            "analyzed_frame_count": analyzed_frame_count,
            "skipped_frame_count": total_frame_count - analyzed_frame_count,
            "inference_status": "success",
        },
        "face_scores": face_scores,
        "frame_scores": frame_scores,
        "segment_scores": segment_scores,
        "top_segments": top_segments,
        "video_score": video_score,
        "errors": [],
    }
    detection = DetectionOutput.model_validate(detection).model_dump(mode="json")

    final_result = build_final_result(preprocessing, detection)
    final_result = ResultOutput.model_validate(final_result).model_dump(mode="json")

    job_root = preprocessing_path.parent.parent
    detection_output_path = job_root / "output" / "detection.json"
    result_output_path = job_root / "output" / "result.json"
    log_output_path = job_root / "logs" / "detection.log"

    _write_json(detection_output_path, detection)
    _write_json(result_output_path, final_result)
    log_output_path.parent.mkdir(parents=True, exist_ok=True)
    log_output_path.write_text("video stage1 detection completed\n", encoding="utf-8")

    return detection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessing-json", required=True)
    run_video_stage1_detection(parser.parse_args().preprocessing_json)


if __name__ == "__main__":
    main()
