from __future__ import annotations

import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import static_ffmpeg


static_ffmpeg.add_paths()


def _parse_frame_rate(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    return float(Fraction(value))


def _probe_video_ffprobe(video_path: Path) -> dict[str, float | int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found: {video_path}")

    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _parse_frame_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    duration = float(payload.get("format", {}).get("duration") or 0.0)
    frame_count_raw = stream.get("nb_frames")

    if frame_count_raw in (None, "N/A", ""):
        frame_count = int(round(duration * fps)) if fps > 0 else 0
    else:
        frame_count = int(frame_count_raw)

    if width <= 0 or height <= 0 or fps <= 0 or duration < 0 or frame_count < 0:
        raise ValueError(f"Invalid ffprobe metadata for {video_path}")

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration_sec": duration,
        "total_frame_count": frame_count,
    }


def _probe_video_opencv(video_path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video with OpenCV: {video_path}")

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0

        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError(f"Invalid OpenCV metadata for {video_path}")

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "duration_sec": duration,
            "total_frame_count": frame_count,
        }
    finally:
        capture.release()


def probe_video(video_path: str | Path) -> dict[str, float | int]:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    try:
        return _probe_video_ffprobe(path)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, json.JSONDecodeError):
        metadata = _probe_video_opencv(path)
        metadata["duration_sec"] = math.floor(metadata["duration_sec"] * 1000) / 1000
        return metadata
