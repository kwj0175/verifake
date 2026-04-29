# pyright: reportMissingImports=false

import static_ffmpeg
static_ffmpeg.add_paths()

from fastapi import FastAPI
from services.backend.routers import video, instagram

import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

app = FastAPI(
    title="VeriFake API",
    description="영상 업로드 및 분석 상태 조회 API 문서",
    version="1.0.0"
)

app.include_router(video.router, prefix="/api/v1")
app.include_router(video.router, prefix="/media")
app.include_router(instagram.router, prefix="/api/v1")
