from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from services.backend.services.audio_analyzer import run_audio_job, validate_audio_python
from services.backend.tasks import AudioJob, create_audio_job, get_audio_job

router = APIRouter()


class AudioAnalyzeRequest(BaseModel):
    file_path: str = Field(...)


def _get_audio_job_or_404(task_id: str) -> AudioJob:
    job = get_audio_job(task_id)
    if job is None:
        raise HTTPException(status_code=404, detail="해당 task_id를 찾을 수 없습니다.")
    return job


@router.post("/jobs", summary="오디오 분석 작업 생성", tags=["Audio"], status_code=202)
async def create_audio_job_endpoint(
    background_tasks: BackgroundTasks,
    req: AudioAnalyzeRequest,
) -> dict:
    input_path = Path(req.file_path)
    if not input_path.exists():
        raise HTTPException(status_code=400, detail="입력 파일이 존재하지 않습니다.")

    try:
        validate_audio_python()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    task_id = input_path.stem
    artifacts_dir = Path("storage/jobs") / task_id / "audio"
    job = create_audio_job(task_id, str(input_path.resolve()), str(artifacts_dir))

    background_tasks.add_task(run_audio_job, task_id, input_path.resolve())

    return {
        "task_id": task_id,
        "status": job["status"],
        "audio_path": job["audio_path"],
        "result_path": job["result_path"],
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/jobs/{task_id}", summary="오디오 분석 상태 조회", tags=["Audio"])
async def get_audio_job_status(task_id: str) -> dict:
    job = _get_audio_job_or_404(task_id)
    return {
        "task_id": job["task_id"],
        "status": job["status"],
        "stage": job["stage"],
        "file_path": job["file_path"],
        "audio_path": job["audio_path"],
        "result_path": job["result_path"],
        "error": job["error"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
    }


@router.get("/jobs/{task_id}/result", summary="오디오 분석 결과 조회", tags=["Audio"])
async def get_audio_result(task_id: str) -> dict:
    job = _get_audio_job_or_404(task_id)

    if job["status"] != "SUCCEEDED":
        raise HTTPException(
            status_code=409,
            detail=f"결과가 아직 준비되지 않았습니다. 현재 상태: {job['status']}",
        )

    result_path = job.get("result_path")
    if result_path:
        resolved = Path(result_path)
        if resolved.exists():
            try:
                return json.loads(resolved.read_text(encoding="utf-8"))
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"결과 파일을 읽을 수 없습니다: {resolved}",
                ) from exc

    return job["result"]
