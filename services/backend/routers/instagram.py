# pyright: reportMissingImports=false

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from uuid import uuid4
from datetime import datetime

from services.backend.services.download import run_download
from services.backend.services.processor import save_and_split

router = APIRouter()

tasks_db = {}


@router.post("/instagram", summary="인스타그램 영상 수집", tags=["Upload"])
async def receive_instagram(
    background_tasks: BackgroundTasks,
    title: str = Form(..., description="영상 제목"),
    link: str = Form(..., description="인스타그램 영상 링크"),
):
    if "instagram.com" not in link:
        raise HTTPException(status_code=400, detail="유효한 인스타그램 링크가 아닙니다.")

    task_id = str(uuid4())
    tasks_db[task_id] = {"status": "PENDING", "verdict": None, "title": title}
    background_tasks.add_task(run_download, task_id, link, tasks_db)

    return {
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "message": "수집 요청이 완료되었습니다.",
    }


@router.post("/video", summary="영상 파일 수집", tags=["Upload"])
async def receive_video(
    title: str = Form(..., description="영상 제목"),
    videoFile: UploadFile = File(..., description="업로드할 영상 파일"),
):
    task_id = str(uuid4())
    content = await videoFile.read()
    download_dir, video_path, audio_path = save_and_split(task_id, videoFile.filename, content)
    tasks_db[task_id] = {
        "status": "DONE",
        "verdict": None,
        "title": title,
        "download_dir": download_dir,
        "video_path": video_path,
        "audio_path": audio_path,
    }

    return {
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "message": "수집 요청이 완료되었습니다.",
    }


@router.get("/status/{task_id}", summary="분석 상태 조회", tags=["Status"])
async def get_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="해당 task_id를 찾을 수 없습니다.")

    task = tasks_db[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "verdict": task["verdict"],
        "video_path": task.get("video_path"),
        "audio_path": task.get("audio_path"),
        "error": task.get("error"),
        "timestamp": datetime.now().isoformat(),
    }
