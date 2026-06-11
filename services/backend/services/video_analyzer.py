from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from services.backend.tasks import get_video_detect_job, update_video_detect_job

# result.json에서 verdict / deepfake_score 를 파싱하는 공통 유틸
def parse_result_json(result_path: Path) -> tuple[str, float]:
    """result.json을 파싱해 (verdict, deepfake_score) 반환.

    Args:
        result_path: video stage1 result.json 파일 경로

    Returns:
        verdict: "FAKE" | "REAL"
        deepfake_score: 0.0 ~ 100.0 범위 퍼센트 점수
    """
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    detection = raw.get("detection") or {}
    video_score = detection.get("video_score") or {}
    raw_score = video_score.get("final_fake_score", 0.0)
    score = round(max(0.0, min(1.0, float(raw_score))) * 100.0, 1)
    verdict = "FAKE" if score >= 50.0 else "REAL"
    return verdict, score


VIDEO_DETECT_TIMEOUT_SEC = 30 * 60
DETECTION_FILENAME = "detection.json"
RESULT_FILENAME = "result.json"
LOG_LIMIT_CHARS = 16000


def _truncate_log(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    return text[-LOG_LIMIT_CHARS:]


def get_video_ai_python() -> Path:
    raw_path = os.getenv("VERIFAKE_VIDEO_AI_PYTHON")
    if not raw_path:
        raw_path = sys.executable
        project_root = Path(__file__).resolve().parents[3]
        env_name = ".venv-antideepfake"
        exe_name = "python.exe" if os.name == "nt" else "python"
        candidate = project_root / env_name / "Scripts" / exe_name
        if candidate.exists():
            raw_path = str(candidate)

    python_path = Path(raw_path).expanduser()
    if not python_path.is_absolute():
        python_path = Path.cwd() / python_path
    python_path = python_path.absolute()
    if not python_path.exists():
        raise RuntimeError(f"VERIFAKE_VIDEO_AI_PYTHON 경로가 존재하지 않습니다: {python_path}")
    if not python_path.is_file():
        raise RuntimeError(f"VERIFAKE_VIDEO_AI_PYTHON 경로가 실행 파일이 아닙니다: {python_path}")
    if not os.access(python_path, os.X_OK):
        raise RuntimeError(f"VERIFAKE_VIDEO_AI_PYTHON 실행 권한이 없습니다: {python_path}")
    return python_path


def validate_video_ai_python() -> Path:
    return get_video_ai_python()


def build_video_detect_command(
    *,
    python_executable: Path,
    preprocessing_json: Path,
) -> list[str]:
    return [
        str(python_executable),
        "-m",
        "services.ai.pipelines.video_stage1.detect",
        "--preprocessing-json",
        str(preprocessing_json),
    ]


def run_video_detect_job(task_id: str, preprocessing_json: Path) -> None:
    started_at = datetime.now().isoformat()
    try:
        python_executable = get_video_ai_python()
        output_dir = Path("storage/jobs") / task_id / "output"
        detection_path = output_dir / DETECTION_FILENAME
        result_path = output_dir / RESULT_FILENAME

        update_video_detect_job(
            task_id,
            status="ANALYZING",
            stage="video_stage1_detect",
            preprocessing_json=str(preprocessing_json.resolve()),
            detection_path=str(detection_path),
            result_path=str(result_path),
            started_at=started_at,
        )

        command = build_video_detect_command(
            python_executable=python_executable,
            preprocessing_json=preprocessing_json.resolve(),
        )

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=VIDEO_DETECT_TIMEOUT_SEC,
            cwd=Path(__file__).resolve().parents[3],
        )

        stdout = _truncate_log(completed.stdout)
        stderr = _truncate_log(completed.stderr)

        if completed.returncode != 0:
            update_video_detect_job(
                task_id,
                status="FAILED",
                stage="video_stage1_detect",
                stdout=stdout,
                stderr=stderr,
                returncode=completed.returncode,
                error=stderr or f"video_stage1 detect subprocess failed with return code {completed.returncode}",
                finished_at=datetime.now().isoformat(),
            )
            return

        if not result_path.exists():
            update_video_detect_job(
                task_id,
                status="FAILED",
                stage="video_stage1_detect",
                stdout=stdout,
                stderr=stderr,
                returncode=completed.returncode,
                error=f"video_stage1 detect 결과 파일이 생성되지 않았습니다: {result_path}",
                finished_at=datetime.now().isoformat(),
            )
            return

        try:
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            update_video_detect_job(
                task_id,
                status="FAILED",
                stage="video_stage1_detect",
                stdout=stdout,
                stderr=stderr,
                returncode=completed.returncode,
                error=f"video_stage1 detect 결과 파일을 읽을 수 없습니다: {result_path}",
                finished_at=datetime.now().isoformat(),
            )
            return

        update_video_detect_job(
            task_id,
            status="SUCCEEDED",
            stage="video_stage1_detect",
            stdout=stdout,
            stderr=stderr,
            returncode=completed.returncode,
            result=result_payload,
            finished_at=datetime.now().isoformat(),
        )
    except subprocess.TimeoutExpired as exc:
        update_video_detect_job(
            task_id,
            status="TIMED_OUT",
            stage="video_stage1_detect",
            stdout=_truncate_log(exc.stdout),
            stderr=_truncate_log(exc.stderr),
            error=f"video_stage1 detect subprocess timeout after {VIDEO_DETECT_TIMEOUT_SEC} seconds",
            finished_at=datetime.now().isoformat(),
        )
    except Exception as exc:
        existing_job = get_video_detect_job(task_id) or {}
        update_video_detect_job(
            task_id,
            status="FAILED",
            stage=existing_job.get("stage", "video_stage1_detect"),
            error=str(exc),
            finished_at=datetime.now().isoformat(),
        )
