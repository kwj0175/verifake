# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportUninitializedInstanceVariable=false
from __future__ import annotations


import json
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from services.ai.common.job_paths import build_job_paths
from services.ai.pipelines.video_stage1.config import get_stage1_storage_root
from services.ai.pipelines.video_stage1.detect import run_video_stage1_detection
from services.ai.pipelines.video_stage1.exceptions import Stage1UnavailableError
from services.backend.services.video_analyzer import run_video_detect_job, validate_video_ai_python
from services.backend.tasks import create_video_detect_job, get_video_detect_job

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALID_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

_GENERIC_ERROR = "내부 서버 오류가 발생했습니다."
_PREPROCESS_RUNTIME_ERROR = "Stage1 A AI 런타임이 준비되지 않았습니다."
_DETECT_RUNTIME_ERROR = "Stage1 B AI 런타임이 준비되지 않았습니다."


# ---------------------------------------------------------------------------
# 요청 스키마
# ---------------------------------------------------------------------------

class VideoStage1PreprocessRequest(BaseModel):
    file_path: str
    job_id: Optional[str] = None


class VideoStage1DetectRequest(BaseModel):
    preprocessing_json: str


class VideoStage1PreprocessResult(TypedDict):
    job_id: str
    status: str


def _get_project_root() -> Path:
    return PROJECT_ROOT


def _get_stage1_storage_root() -> Path:
    return Path(get_stage1_storage_root())


def run_video_stage1_preprocess_job(
    input_file: Path,
    job_id: Optional[str] = None,
) -> VideoStage1PreprocessResult:
    from services.ai.pipelines.video_stage1.preprocess import run_video_stage1_preprocess

    raw_result: dict[str, object] = run_video_stage1_preprocess(input_path=str(input_file), job_id=job_id)
    raw_job_id = raw_result.get("job_id")
    raw_status = raw_result.get("status")
    if not isinstance(raw_job_id, str) or not isinstance(raw_status, str):
        raise RuntimeError("Stage1 preprocessing returned an invalid result.")

    return {"job_id": raw_job_id, "status": raw_status}


def _get_resolved_stage1_storage_root() -> Path:
    storage_root = _get_stage1_storage_root()
    if not storage_root.is_absolute():
        storage_root = _get_project_root() / storage_root
    return storage_root.resolve()


def _is_within_directory(path: Path, directory: Path) -> bool:
    try:
        _ = path.relative_to(directory)
        return True
    except ValueError:
        return False


def _resolve_existing_path(raw_path: str) -> Path:
    resolved = Path(raw_path).expanduser().resolve()
    if not resolved.exists():
        raise HTTPException(status_code=400, detail="파일이 존재하지 않습니다.")
    return resolved


def _validate_project_file_path(raw_path: str) -> Path:
    resolved = _resolve_existing_path(raw_path)
    if not _is_within_directory(resolved, PROJECT_ROOT):
        raise HTTPException(status_code=400, detail="프로젝트 내부 파일만 허용됩니다.")
    return resolved


def _validate_preprocessing_json_path(raw_path: str) -> Path:
    resolved = _resolve_existing_path(raw_path)
    storage_root = _get_stage1_storage_root()
    if resolved.name != "preprocessing.json" or not _is_within_directory(resolved, storage_root):
        raise HTTPException(
            status_code=400,
            detail="Stage1 storage_root 내부 preprocessing.json만 허용됩니다.",
        )
    return resolved


def _validate_job_id(job_id: str | None) -> None:
    if job_id is not None and not VALID_JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=400, detail="job_id 형식이 올바르지 않습니다.")


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

@router.post("/video-stage1/preprocess", summary="Stage1 A 전처리 실행", tags=["Video"])
def preprocess_video_stage1(req: VideoStage1PreprocessRequest) -> dict[str, str]:
    try:
        _validate_job_id(req.job_id)
        input_path = _validate_project_file_path(req.file_path)

        result = run_video_stage1_preprocess_job(input_path, job_id=req.job_id)
        storage_root = _get_stage1_storage_root()
        job_paths = build_job_paths(result["job_id"], storage_root=storage_root)

        return {
            "job_id": result["job_id"],
            "status": result["status"],
            "preprocessing_json": job_paths["preprocessing_json_path"].as_posix(),
        }

    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="ffmpeg 실행 실패")
    except Stage1UnavailableError:
        raise HTTPException(status_code=500, detail=_PREPROCESS_RUNTIME_ERROR)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail=_GENERIC_ERROR)


@router.post("/video-stage1/detect", summary="Stage1 B 탐지 실행", tags=["Video"])
def detect_video_stage1(req: VideoStage1DetectRequest) -> dict[str, object]:
    preprocessing_json_path = _validate_preprocessing_json_path(req.preprocessing_json)

    try:
        detection = run_video_stage1_detection(str(preprocessing_json_path))
    except Stage1UnavailableError:
        raise HTTPException(status_code=500, detail=_DETECT_RUNTIME_ERROR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_GENERIC_ERROR) from exc

    storage_root = _get_stage1_storage_root()
    job_paths = build_job_paths(detection["job_id"], storage_root=storage_root)

    return {
        "job_id": detection["job_id"],
        "status": detection.get("status", "success"),
        "detection_json": job_paths["detection_json_path"].as_posix(),
        "result_json": job_paths["result_json_path"].as_posix(),
    }


@router.post("/video-stage1/detect/jobs", summary="Stage1 B 탐지 작업 생성", tags=["Video"], status_code=202)
def create_video_detect_job_endpoint(background_tasks: BackgroundTasks, req: VideoStage1DetectRequest):
    preprocessing_json_path = Path(req.preprocessing_json)
    if not preprocessing_json_path.exists():
        raise HTTPException(
            status_code=400,
            detail="preprocessing.json 파일이 존재하지 않습니다.",
        )

    preprocessing_json_path = _validate_preprocessing_json_path(req.preprocessing_json)

    try:
        validate_video_ai_python()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    task_id = preprocessing_json_path.parents[1].name
    artifacts_dir = Path("storage/jobs") / task_id / "output"
    job = create_video_detect_job(task_id, str(preprocessing_json_path.resolve()), str(artifacts_dir))

    background_tasks.add_task(run_video_detect_job, task_id, preprocessing_json_path.resolve())

    return {
        "task_id": task_id,
        "status": job["status"],
        "preprocessing_json": job["preprocessing_json"],
        "detection_path": job["detection_path"],
        "result_path": job["result_path"],
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/video-stage1/detect/jobs/{task_id}", summary="Stage1 B 탐지 작업 상태 조회", tags=["Video"])
def get_video_detect_job_status(task_id: str):
    job = get_video_detect_job(task_id)
    if job is None:
        raise HTTPException(status_code=404, detail="해당 task_id를 찾을 수 없습니다.")

    return {
        "task_id": job["task_id"],
        "status": job["status"],
        "stage": job["stage"],
        "preprocessing_json": job["preprocessing_json"],
        "detection_path": job["detection_path"],
        "result_path": job["result_path"],
        "error": job["error"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
    }


@router.get("/video-stage1/detect/jobs/{task_id}/result", summary="Stage1 B 탐지 작업 결과 조회", tags=["Video"])
def get_video_detect_result(task_id: str):
    job = get_video_detect_job(task_id)
    if job is None:
        raise HTTPException(status_code=404, detail="해당 task_id를 찾을 수 없습니다.")
    if job["status"] != "SUCCEEDED":
        raise HTTPException(status_code=409, detail=f"결과가 아직 준비되지 않았습니다. 현재 상태: {job['status']}")
    result_path = job.get("result_path")
    if result_path:
        resolved_result_path = Path(result_path)
        if resolved_result_path.exists():
            try:
                return json.loads(resolved_result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"결과 파일을 읽을 수 없습니다: {resolved_result_path}") from exc
    return job["result"]
