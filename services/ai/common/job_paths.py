from __future__ import annotations

from pathlib import Path


def build_job_paths(job_id: str, storage_root: str | Path = "storage/jobs") -> dict[str, Path]:
    if not job_id.strip():
        raise ValueError("job_id must not be empty")

    root = Path(storage_root)
    job_root = root / job_id

    return {
        "storage_root": root,
        "job_root": job_root,
        "input_dir": job_root / "input",
        "frames_dir": job_root / "frames",
        "faces_dir": job_root / "faces",
        "metadata_dir": job_root / "metadata",
        "output_dir": job_root / "output",
        "logs_dir": job_root / "logs",
        "original_video_path": job_root / "input" / "original",
        "normalized_video_path": job_root / "input" / "normalized.mp4",
        "preprocessing_json_path": job_root / "metadata" / "preprocessing.json",
        "detection_json_path": job_root / "output" / "detection.json",
        "result_json_path": job_root / "output" / "result.json",
        "preprocess_log_path": job_root / "logs" / "preprocess.log",
        "detection_log_path": job_root / "logs" / "detection.log",
    }


def create_job_dirs(job_id: str, storage_root: str | Path = "storage/jobs") -> dict[str, Path]:
    paths = build_job_paths(job_id=job_id, storage_root=storage_root)

    for key in ("job_root", "input_dir", "frames_dir", "faces_dir", "metadata_dir", "output_dir", "logs_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    return paths
