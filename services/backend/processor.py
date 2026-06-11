from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import static_ffmpeg


static_ffmpeg.add_paths()

# 결과 저장 폴더
VIDEO_DIR = Path("storage/video")
AUDIO_DIR = Path("storage/audio")

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _run_ffmpeg_command(command: list[str], description: str) -> None:
    """FFmpeg 명령어 실행
    
    Args:
        command: FFmpeg 명령어 리스트
        description: 작업 설명 (에러 메시지용)
        
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


def separate_streams(input_file: Path, job_id: str) -> tuple[str, str]:
    """영상과 음성 스트림 분리
    
    Args:
        input_file: 입력 미디어 파일 경로
        job_id: 작업 ID
        
    Returns:
        (video_path, audio_path) 튜플
        
    Raises:
        FileNotFoundError: 입력 파일이 없을 때
        RuntimeError: FFmpeg 실행 실패 시
    """
    if not input_file.exists():
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {input_file}")
    
    video_out = VIDEO_DIR / f"{job_id}_video.mp4"
    audio_out = AUDIO_DIR / f"{job_id}_audio.wav"

    # 영상만 추출 (비디오 코덱 복사, 음성 제거)
    _run_ffmpeg_command(
        [
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-an",
            "-c:v", "copy",
            str(video_out),
        ],
        description="영상 추출",
    )

    # 음성만 추출 (PCM 16-bit, 16kHz, mono)
    _run_ffmpeg_command(
        [
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_out),
        ],
        description="음성 추출",
    )

    return str(video_out), str(audio_out)


def run_video_stage1_preprocess_job(
    input_file: Path,
    job_id: str | None = None,
) -> dict[str, Any]:
    """video_stage1 전처리 작업 실행
    
    Args:
        input_file: 입력 영상 파일 경로
        job_id: 작업 ID (선택사항)
        
    Returns:
        전처리 결과 딕셔너리
    """
    from services.ai.pipelines.video_stage1.preprocess import run_video_stage1_preprocess

    return run_video_stage1_preprocess(
        input_path=str(input_file),
        job_id=job_id,
    )


def run_video_stage1_result_explainer_job(
    video_result_json_path: Path,
    audio_result_json_path: Path,
) -> dict[str, Any]:
    """video_stage1 결과 분석 작업 실행
    
    Args:
        video_result_json_path: 영상 분석 결과 JSON 경로
        audio_result_json_path: 음성 분석 결과 JSON 경로
        
    Returns:
        분석 결과 딕셔너리
    """
    from services.ai.pipelines.video_stage1.result_explainer import (
        run_video_stage1_result_explainer,
    )

    return run_video_stage1_result_explainer(
        video_result_json_path=str(video_result_json_path),
        audio_result_json_path=str(audio_result_json_path),
    )