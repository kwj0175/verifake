from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import static_ffmpeg

from services.ai.common.job_paths import create_job_dirs
from services.ai.common.json_io import write_json
from services.ai.pipelines.video_stage1.config import load_stage1_config
from services.ai.pipelines.video_stage1.exceptions import Stage1UnavailableError
from services.ai.pipelines.video_stage1.schema import (
    FaceSummary,
    FrameEntry,
    PipelineStatus,
    PreprocessingDocument,
    QualityMetrics,
    Stage1InputInfo,
    VideoMetadata,
)
static_ffmpeg.add_paths()


def _load_stage1_runtime_components() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from services.ai.common.video_probe import probe_video
        from services.ai.pipelines.video_stage1.face_detect import detect_and_crop_faces
        from services.ai.pipelines.video_stage1.frame_sampler import sample_frames
        from services.ai.pipelines.video_stage1.quality import (
            calculate_face_summary,
            calculate_quality_metrics,
        )
    except ImportError as exc:
        raise Stage1UnavailableError(
            "Stage1 preprocessing requires optional AI runtime dependencies. "
            "Install services/backend/requirements-ai-stage1.txt before calling this pipeline."
        ) from exc

    return (
        probe_video,
        detect_and_crop_faces,
        sample_frames,
        calculate_face_summary,
        calculate_quality_metrics,
    )
def _generate_job_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"job_{timestamp}_{uuid.uuid4().hex[:6]}"


def _create_logger(log_path: Path, job_id: str) -> logging.Logger:
    logger = logging.getLogger(f"video_stage1_preprocess.{job_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def _normalize_video(input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def _copy_original_video(input_path: Path, destination_without_suffix: Path) -> Path:
    destination = destination_without_suffix.with_suffix(input_path.suffix.lower())
    shutil.copy2(input_path, destination)
    return destination


def _build_status(face_detected: bool, errors: list[str]) -> PipelineStatus:
    if errors:
        return "failed"
    if face_detected:
        return "success"
    return "partial_success"


def run_video_stage1_preprocess(input_path: str, job_id: str | None = None) -> dict[str, Any]:
    config = load_stage1_config()
    source_path = Path(input_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Input video not found: {source_path}")
    (
        probe_video,
        detect_and_crop_faces,
        sample_frames,
        calculate_face_summary,
        calculate_quality_metrics,
    ) = _load_stage1_runtime_components()

    resolved_job_id = job_id or _generate_job_id()
    paths = create_job_dirs(resolved_job_id, storage_root=config["storage_root"])
    logger = _create_logger(paths["preprocess_log_path"], resolved_job_id)

    logger.info("Stage1 A preprocess started: %s", source_path)

    original_video_path = _copy_original_video(source_path, paths["original_video_path"])
    normalized_video_path = paths["normalized_video_path"]

    logger.info("Normalizing video to mp4")
    _normalize_video(original_video_path, normalized_video_path)

    logger.info("Probing normalized video")
    probed_metadata = probe_video(normalized_video_path)
    max_duration = float(config["max_video_duration_sec"])
    if probed_metadata["duration_sec"] > max_duration:
        raise ValueError(
            f"Video duration exceeds max_video_duration_sec: {probed_metadata['duration_sec']} > {max_duration}"
        )

    logger.info("Sampling frames at %s fps", config["sample_fps"])
    frames = sample_frames(
        video_path=normalized_video_path,
        frames_dir=paths["frames_dir"],
        sample_fps=float(config["sample_fps"]),
    )

    logger.info("Running face detection with RetinaFace")
    frames = detect_and_crop_faces(
        frames=frames,
        faces_dir=paths["faces_dir"],
        confidence_threshold=float(config["face_detection"]["confidence_threshold"]),
        max_faces_per_frame=int(config["face_detection"]["max_faces_per_frame"]),
    )

    quality_config = config["quality"]
    logger.info("Calculating quality metrics")
    quality_metrics_payload = calculate_quality_metrics(
        video_metadata=probed_metadata,
        frames=frames,
        visibility_min_confidence=float(quality_config["visibility_min_confidence"]),
        visibility_min_area_ratio=float(quality_config["visibility_min_area_ratio"]),
        blur_variance_reference=float(quality_config["blur_variance_reference"]),
        motion_blur_threshold=float(quality_config["motion_blur_threshold"]),
        dark_frame_mean_threshold=float(quality_config["dark_frame_mean_threshold"]),
        compression_block_size=int(quality_config["compression_block_size"]),
    )
    face_summary_payload = calculate_face_summary(
        frames=frames,
        frame_width=int(probed_metadata["width"]),
        frame_height=int(probed_metadata["height"]),
    )

    errors: list[str] = []
    status = _build_status(bool(face_summary_payload["human_face_detected"]), errors)
    input_info = Stage1InputInfo(
        original_video_path=original_video_path.as_posix(),
        normalized_video_path=normalized_video_path.as_posix(),
        original_extension=source_path.suffix.lower().lstrip("."),
        internal_format=str(config["internal_video_format"]),
    )
    video_metadata = VideoMetadata(
        width=int(probed_metadata["width"]),
        height=int(probed_metadata["height"]),
        fps=float(probed_metadata["fps"]),
        duration_sec=float(probed_metadata["duration_sec"]),
        total_frame_count=int(probed_metadata["total_frame_count"]),
        sample_fps=float(config["sample_fps"]),
        sampled_frame_count=len(frames),
    )
    quality_metrics = QualityMetrics(**quality_metrics_payload)
    face_summary = FaceSummary(**face_summary_payload)
    frame_entries = [FrameEntry(**frame) for frame in frames]
    document = PreprocessingDocument(
        schema_version=config["schema_version"],
        job_id=resolved_job_id,
        pipeline_stage=int(config["pipeline_stage"]),
        status=status,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        input=input_info,
        video_metadata=video_metadata,
        quality_metrics=quality_metrics,
        face_summary=face_summary,
        frames=frame_entries,
        errors=errors,
    )

    payload = document.model_dump(mode="json")
    write_json(paths["preprocessing_json_path"], payload)
    logger.info("Preprocessing output written: %s", paths["preprocessing_json_path"])
    return payload


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VeriFake Stage1 A video preprocessing")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--job-id", required=False, help="Optional job id")
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    result = run_video_stage1_preprocess(input_path=args.input, job_id=args.job_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
