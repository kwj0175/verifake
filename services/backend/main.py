import static_ffmpeg
static_ffmpeg.add_paths()
from fastapi import FastAPI
from services.backend.routers import media, download
from services.backend.internal.task_manager import TaskManager

app = FastAPI(title="VeriFake API", version="1.1.0")

@app.on_event("startup")
async def startup_event():
    # 서버 기동 시 예외 및 만료 관리 엔진 실행
    TaskManager.initialize_tasks()
    TaskManager.cleanup_expired_tasks()

app.include_router(media.router, prefix="/media")
app.include_router(download.router, prefix="/api/v1")
