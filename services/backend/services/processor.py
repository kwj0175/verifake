# pyright: reportMissingImports=false

import subprocess
from pathlib import Path

import static_ffmpeg


_ = static_ffmpeg.add_paths()

VIDEO_DIR = Path("storage/video")
AUDIO_DIR = Path("storage/audio")
TMP_DIR = Path("storage/tmp")
MEDIA_SPLIT_TIMEOUT_SEC = 10 * 60

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


def separate_streams(input_file: Path, job_id: str):
    video_out = VIDEO_DIR / f"{job_id}_video.mp4"
    audio_out = AUDIO_DIR / f"{job_id}_audio.wav"

    _ = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-an",
            "-c:v", "copy",
            str(video_out),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=MEDIA_SPLIT_TIMEOUT_SEC,
    )

    _ = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_out),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=MEDIA_SPLIT_TIMEOUT_SEC,
    )

    return str(video_out), str(audio_out)


def _safe_upload_filename(filename: str) -> str:
    safe_name = Path(filename).name.strip()
    if safe_name:
        return safe_name
    return "upload.mp4"


def save_and_split(task_id: str, filename: str, content: bytes):
    dest_dir = TMP_DIR / task_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / _safe_upload_filename(filename)
    _ = dest_path.write_bytes(content)
    video_path, audio_path = separate_streams(dest_path, task_id)
    return str(dest_dir), video_path, audio_path


# 버그 수정: extract_thumbnail이 호출하는 _run_ffmpeg_command가 이 파일에 없었음 → 추가
def _run_ffmpeg_command(command: list[str], description: str) -> None:
    """FFmpeg 명령어 실행

    Raises:
        RuntimeError: FFmpeg 실행 실패 시
    """
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg {description} 실패: {stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg를 찾을 수 없습니다. static_ffmpeg이 제대로 초기화되었는지 확인하세요.")


def extract_thumbnail(video_path: Path, task_id: str) -> str:
    thumb_path = VIDEO_DIR / f"{task_id}_thumb.jpg"
    _run_ffmpeg_command([
        "ffmpeg", "-y", "-i", str(video_path),
        "-ss", "00:00:01", "-vframes", "1", str(thumb_path)
    ], "썸네일 추출")
    return str(thumb_path)
