# pyright: reportMissingImports=false

import subprocess
from pathlib import Path
from typing import Any

import static_ffmpeg


static_ffmpeg.add_paths()

VIDEO_DIR = Path("storage/video")
AUDIO_DIR = Path("storage/audio")
TMP_DIR = Path("storage/tmp")

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


def separate_streams(input_file: Path, job_id: str):
    video_out = VIDEO_DIR / f"{job_id}_video.mp4"
    audio_out = AUDIO_DIR / f"{job_id}_audio.wav"

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(input_file),
        "-an",
        "-c:v", "copy",
        str(video_out)
    ], check=True)

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(input_file),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_out)
    ], check=True)

    return str(video_out), str(audio_out)


def save_and_split(task_id: str, filename: str, content: bytes):
    dest_dir = TMP_DIR / task_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    dest_path.write_bytes(content)
    video_path, audio_path = separate_streams(dest_path, task_id)
    return str(dest_dir), video_path, audio_path


def run_video_stage1_preprocess_job(input_file: Path, job_id: str | None = None) -> dict[str, Any]:
    from services.ai.pipelines.video_stage1.preprocess import run_video_stage1_preprocess

    return run_video_stage1_preprocess(input_path=str(input_file), job_id=job_id)
