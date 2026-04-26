from fastapi import APIRouter, Form, HTTPException
from uuid import uuid4
from ..internal.task_manager import TaskManager, tasks_db

router = APIRouter()

@router.post("/share", tags=["Upload"])
async def upload_video(title: str = Form(...), link: str = Form(None)):
    task_id = str(uuid4())
    TaskManager.create_task(task_id) # 관제 센터에 등록
    
    # 인스타그램 링크 유효성 검사
    if link and "instagram.com" not in link:
        tasks_db[task_id].status = "FAILED"
        tasks_db[task_id].unsupported_reason = "Invalid Link"
        raise HTTPException(status_code=400, detail="인스타 링크가 아닙니다.")
        
    return {"task_id": task_id, "message": "수집 시작"}

@router.get("/status/{task_id}", tags=["Status"])
async def get_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks_db[task_id]
