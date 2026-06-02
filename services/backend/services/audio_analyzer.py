from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from services.backend.tasks import (
    fail_audio_job,
    get_audio_job,
    start_audio_job,
    succeed_audio_job,
    timeout_audio_job,
)
from services.ai.common.runtime_probe import resolve_torch_device


AUDIO_STAGE1_TIMEOUT_SEC = 30 * 60
RESULT_FILENAME = "audio_stage1_result.json"
LOG_LIMIT_CHARS = 16000
DEFAULT_AI_DEVICE = "cpu"


def _truncate_log(value: str | bytes | None) -> str:
    """로그 크기 제한"""
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    return text[-LOG_LIMIT_CHARS:]


def get_audio_python() -> Path:
    """AI 파이썬 인터프리터 경로 조회"""
    raw_path = os.getenv("VERIFAKE_AI_PYTHON")
    if not raw_path:
        raise RuntimeError("VERIFAKE_AI_PYTHON 환경변수가 설정되지 않았습니다.")

    python_path = Path(raw_path).expanduser()
    if not python_path.is_absolute():
        python_path = Path.cwd() / python_path
    python_path = python_path.absolute()
    if not python_path.exists():
        raise RuntimeError(f"VERIFAKE_AI_PYTHON 경로가 존재하지 않습니다: {python_path}")
    if not python_path.is_file():
        raise RuntimeError(f"VERIFAKE_AI_PYTHON 경로가 실행 파일이 아닙니다: {python_path}")
    if not os.access(python_path, os.X_OK):
        raise RuntimeError(f"VERIFAKE_AI_PYTHON 실행 권한이 없습니다: {python_path}")

    return python_path


def validate_audio_python() -> Path:
    """AI 파이썬 인터프리터 검증"""
    return get_audio_python()


def _resolve_device_from_runtime(python_executable: Path) -> str:
    script = (
        "import os\\n"
        "try:\\n"
        "    import torch\\n"
        "    if not torch.cuda.is_available():\\n"
        "        print('cpu')\\n"
        "    else:\\n"
        "        preferred = os.getenv('VERIFAKE_CUDA_DEVICE', '0').strip()\\n"
        "        if preferred.isdigit():\\n"
        "            print(f'cuda:{preferred}')\\n"
        "        else:\\n"
        "            print(preferred or 'cuda')\\n"
        "except Exception:\\n"
        "    print('cpu')\\n"
    )

    try:
        result = subprocess.run(
            [str(python_executable), "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return DEFAULT_AI_DEVICE

    selected = (result.stdout or "").strip()
    if selected:
        return selected
    return DEFAULT_AI_DEVICE


def get_audio_device(python_executable: Path | None = None) -> str:
    explicit_device = os.getenv("VERIFAKE_AI_DEVICE", "").strip()
    if explicit_device:
        return explicit_device
    if python_executable is not None:
        return _resolve_device_from_runtime(python_executable)
    return resolve_torch_device(default=DEFAULT_AI_DEVICE, cuda_index_env_var="VERIFAKE_CUDA_DEVICE")


def build_audio_stage1_command(
    *,
    python_executable: Path,
    input_path: Path,
    output_dir: Path,
    job_id: str,
    device: str,
) -> list[str]:
    """audio_stage1 실행 명령어 생성"""
    command = [
        str(python_executable),
        "-m",
        "services.ai.audio_pipeline.audio_stage1",
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--request-id",
        job_id,
        "--json-output",
        str(output_dir / RESULT_FILENAME),
    ]
    if device:
        command.extend(["--device", device])
    return command


def run_audio_job(job_id: str, input_path: Path) -> None:
    """오디오 작업 실행"""
    try:
        # 준비 단계
        python_executable = get_audio_python()
        device = get_audio_device(python_executable)
        output_dir = Path("storage/jobs") / job_id / "audio"
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / RESULT_FILENAME

        # 작업 시작
        start_audio_job(
            task_id=job_id,
            stage="audio_stage1",
            audio_path=str(input_path.resolve()),
            artifacts_dir=str(output_dir),
            result_path=str(result_path),
        )

        # 명령어 실행
        command = build_audio_stage1_command(
            python_executable=python_executable,
            input_path=input_path.resolve(),
            output_dir=output_dir,
            job_id=job_id,
            device=device,
        )

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=AUDIO_STAGE1_TIMEOUT_SEC,
            cwd=Path(__file__).resolve().parents[3],
        )

        stdout = _truncate_log(completed.stdout)
        stderr = _truncate_log(completed.stderr)

        # 결과 처리
        _handle_audio_stage1_result(
            job_id=job_id,
            completed=completed,
            stdout=stdout,
            stderr=stderr,
            result_path=result_path,
        )

    except subprocess.TimeoutExpired as exc:
        timeout_audio_job(
            task_id=job_id,
            stage="audio_stage1",
            timeout_sec=AUDIO_STAGE1_TIMEOUT_SEC,
            stdout=_truncate_log(exc.stdout),
            stderr=_truncate_log(exc.stderr),
        )
    except Exception as exc:
        # 예상치 못한 에러
        existing_job = get_audio_job(job_id) or {}
        fail_audio_job(
            task_id=job_id,
            stage=existing_job.get("stage", "audio_stage1"),
            error=str(exc),
        )


def _handle_audio_stage1_result(
    job_id: str,
    completed: subprocess.CompletedProcess[str],
    stdout: str,
    stderr: str,
    result_path: Path,
) -> None:
    """audio_stage1 실행 결과 처리"""
    stage = "audio_stage1"

    # 1. 프로세스 실패
    if completed.returncode != 0:
        fail_audio_job(
            task_id=job_id,
            stage=stage,
            error=stderr or f"{stage} subprocess failed with return code {completed.returncode}",
            stdout=stdout,
            stderr=stderr,
            returncode=completed.returncode,
        )
        return

    # 2. 결과 파일 미생성
    if not result_path.exists():
        fail_audio_job(
            task_id=job_id,
            stage=stage,
            error=f"{stage} 결과 파일이 생성되지 않았습니다: {result_path}",
            stdout=stdout,
            stderr=stderr,
            returncode=completed.returncode,
        )
        return

    # 3. 결과 파일 파싱 실패
    try:
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail_audio_job(
            task_id=job_id,
            stage=stage,
            error=f"{stage} 결과 파일을 읽을 수 없습니다: {result_path} ({exc})",
            stdout=stdout,
            stderr=stderr,
            returncode=completed.returncode,
        )
        return

    # 4. 성공
    succeed_audio_job(
        task_id=job_id,
        stage=stage,
        result=result_payload,
        stdout=stdout,
        stderr=stderr,
        returncode=completed.returncode,
    )
