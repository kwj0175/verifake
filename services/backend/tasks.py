from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, TypedDict

JobStatus = Literal["PENDING", "ANALYZING", "SUCCEEDED", "FAILED", "TIMED_OUT"]

# ── 인메모리 저장소 (공개 dict로 통일) ───────────────────────────────────────
# 버그 수정: _audio_jobs / _upload_tasks 이중 dict 제거 → 공개 dict로 통합
upload_tasks_db: dict[str, dict[str, Any]] = {}
audio_jobs_db: dict[str, dict[str, Any]] = {}
video_detect_jobs_db: dict[str, dict[str, Any]] = {}


class UploadTask(TypedDict):
    task_id: str
    status: JobStatus
    verdict: str | None
    timestamp: str


class AudioJob(TypedDict):
    task_id: str
    status: JobStatus
    stage: str
    file_path: str
    audio_path: str | None
    video_path: str | None
    artifacts_dir: str
    result_path: str | None
    result: Any
    error: str | None
    stdout: str
    stderr: str
    returncode: int | None
    created_at: str
    started_at: str | None
    finished_at: str | None


def _now() -> str:
    return datetime.now().isoformat()


# 하위 호환: create_video_detect_job 에서 _timestamp() 로 호출하는 코드와 맞춤
_timestamp = _now


# 버그 수정: 테스트에서 필요한 clear_stores() 함수 추가
def clear_stores() -> None:
    """테스트용: 모든 인메모리 저장소 초기화"""
    upload_tasks_db.clear()
    audio_jobs_db.clear()
    video_detect_jobs_db.clear()


def create_upload_task(task_id: str) -> UploadTask:
    task: UploadTask = {
        "task_id": task_id,
        "status": "PENDING",
        "verdict": None,
        "timestamp": _now(),
    }
    upload_tasks_db[task_id] = task
    return task


def get_upload_task(task_id: str) -> UploadTask | None:
    return upload_tasks_db.get(task_id)


def create_audio_job(task_id: str, file_path: str, artifacts_dir: str) -> AudioJob:
    job: AudioJob = {
        "task_id": task_id,
        "status": "PENDING",
        "stage": "queued",
        "file_path": file_path,
        "audio_path": None,
        "video_path": None,
        "artifacts_dir": artifacts_dir,
        "result_path": None,
        "result": None,
        "error": None,
        "stdout": "",
        "stderr": "",
        "returncode": None,
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
    }
    audio_jobs_db[task_id] = job
    return job


def get_audio_job(task_id: str) -> AudioJob | None:
    return audio_jobs_db.get(task_id)


def update_audio_job(task_id: str, **fields: Any) -> AudioJob:
    job = audio_jobs_db.get(task_id)
    if job is None:
        raise KeyError(f"audio job not found: {task_id}")
    job.update(fields)  # type: ignore[typeddict-item]
    return job


def create_video_detect_job(task_id: str, preprocessing_json: str, artifacts_dir: str) -> dict[str, Any]:
    job = {
        "task_id": task_id,
        "status": "PENDING",
        "stage": "queued",
        "preprocessing_json": preprocessing_json,
        "artifacts_dir": artifacts_dir,
        "detection_path": None,
        "result_path": None,
        "result": None,
        "error": None,
        "stdout": "",
        "stderr": "",
        "returncode": None,
        "created_at": _timestamp(),
        "started_at": None,
        "finished_at": None,
    }
    video_detect_jobs_db[task_id] = job
    return job


def get_video_detect_job(task_id: str) -> dict[str, Any] | None:
    return video_detect_jobs_db.get(task_id)


def update_video_detect_job(task_id: str, **fields: Any) -> dict[str, Any]:
    job = video_detect_jobs_db[task_id]
    job.update(fields)
    return job


def start_audio_job(task_id: str, stage: str, audio_path: str, artifacts_dir: str, result_path: str) -> AudioJob:
    job = audio_jobs_db.get(task_id)
    if job is None:
        raise KeyError(f"audio job not found: {task_id}")
    job.update({
        "status": "ANALYZING",
        "stage": stage,
        "audio_path": audio_path,
        "artifacts_dir": artifacts_dir,
        "result_path": result_path,
        "started_at": _now(),
    })
    return job


def fail_audio_job(task_id: str, stage: str, error: str, stdout: str = "", stderr: str = "", returncode: int | None = None) -> AudioJob:
    job = audio_jobs_db.get(task_id)
    if job is None:
        raise KeyError(f"audio job not found: {task_id}")
    job.update({
        "status": "FAILED",
        "stage": stage,
        "error": error,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "finished_at": _now(),
    })
    return job


def succeed_audio_job(task_id: str, stage: str, result: Any, stdout: str = "", stderr: str = "", returncode: int | None = None) -> AudioJob:
    job = audio_jobs_db.get(task_id)
    if job is None:
        raise KeyError(f"audio job not found: {task_id}")
    job.update({
        "status": "SUCCEEDED",
        "stage": stage,
        "result": result,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "finished_at": _now(),
    })
    return job


def timeout_audio_job(task_id: str, stage: str, timeout_sec: int, stdout: str = "", stderr: str = "") -> AudioJob:
    job = audio_jobs_db.get(task_id)
    if job is None:
        raise KeyError(f"audio job not found: {task_id}")
    job.update({
        "status": "TIMED_OUT",
        "stage": stage,
        "error": f"audio_stage1 subprocess timeout after {timeout_sec} seconds",
        "stdout": stdout,
        "stderr": stderr,
        "finished_at": _now(),
    })
    return job
