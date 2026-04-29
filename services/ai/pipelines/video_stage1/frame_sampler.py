from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


def sample_frames(
    video_path: str | Path,
    frames_dir: str | Path,
    sample_fps: float = 3.0,
) -> list[dict[str, Any]]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than 0")

    source_path = Path(video_path)
    output_dir = Path(frames_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video for sampling: {source_path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0:
        capture.release()
        raise ValueError(f"Invalid source FPS for video: {source_path}")

    interval_sec = 1.0 / sample_fps
    frame_duration = 1.0 / native_fps
    next_sample_sec = 0.0
    source_frame_index = 0
    sampled_frame_index = 0
    sampled_frames: list[dict[str, Any]] = []

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            timestamp_sec = source_frame_index / native_fps
            if timestamp_sec + (frame_duration / 2.0) >= next_sample_sec:
                frame_path = output_dir / f"frame_{sampled_frame_index:06d}.jpg"
                written = cv2.imwrite(str(frame_path), frame)
                if not written:
                    raise ValueError(f"Failed to write sampled frame: {frame_path}")

                sampled_frames.append(
                    {
                        "frame_index": sampled_frame_index,
                        "timestamp_sec": round(timestamp_sec, 3),
                        "frame_path": frame_path.as_posix(),
                        "face_count": 0,
                        "faces": [],
                    }
                )

                sampled_frame_index += 1
                next_sample_sec += interval_sec

            source_frame_index += 1
    finally:
        capture.release()

    if not sampled_frames:
        raise ValueError(f"No frames were sampled from video: {source_path}")

    return sampled_frames
