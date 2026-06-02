# pyright: reportMissingImports=false, reportUnknownMemberType=false

import logging
import os
from enum import Enum
from types import ModuleType

import PIL.Image
import static_ffmpeg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

try:
    static_ffmpeg.add_paths()
except Exception as exc:
    logger.warning(
        "static_ffmpeg add_paths() failed, skipping runtime auto-download: %s",
        exc,
    )

# ANTIALIAS는 Pillow 10에서 제거됨
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS


# ============ 환경변수 로드 ============

def _get_cors_origins() -> list[str]:
    """CORS 허용 오리진 로드"""
    cors_origins = os.getenv("CORS_ORIGINS", "")
    if cors_origins:
        return [origin.strip() for origin in cors_origins.split(",")]

    return [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "capacitor://localhost",
    ]


def _get_app_environment() -> str:
    """실행 환경 로드 (development, staging, production)"""
    return os.getenv("APP_ENV", "development")


# ============ FastAPI 앱 초기화 ============

app = FastAPI(
    title="VeriFake API",
    description="영상 업로드 및 분석 상태 조회 API 문서",
    version="1.0.0",
)


# ============ 데이터베이스 초기화 ============

database_connected = False
try:
    from services.backend import models
    from services.backend.database import engine

    models.Base.metadata.create_all(bind=engine)
    database_connected = True
except Exception as exc:
    logger.warning("Database initialization skipped: %s", exc)


# ============ CORS 미들웨어 ============

cors_origins = _get_cors_origins()
app_env = _get_app_environment()
allow_all_origins = app_env == "development"

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if not allow_all_origins else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "Accept",
        "Origin",
        "User-Agent",
    ],
    max_age=3600,
)

if app_env != "production":
    logger.info("CORS allowed origins (%s): %s", app_env, cors_origins)


def _import_router_module(module_name: str) -> ModuleType | None:
    try:
        return __import__(f"services.backend.routers.{module_name}", fromlist=[module_name])
    except Exception as exc:
        logger.warning("%s router disabled during import: %s", module_name, exc)
        return None


def _include_router(module_name: str, *, prefix: str, tags: list[str | Enum] | None = None) -> ModuleType | None:
    module = _import_router_module(module_name)
    if module is None:
        return None

    try:
        app.include_router(module.router, prefix=prefix, tags=tags)
    except Exception as exc:
        logger.warning("%s router disabled while registering: %s", module_name, exc)
        return None

    return module


# ============ 라우터 등록 ============

_include_router("video", prefix="/api/v1")
_include_router("instagram", prefix="/api/v1")
_include_router("audio", prefix="/api/v1/audio", tags=["Audio"])
_include_router("user", prefix="/api/v1", tags=["User"])
media = _include_router("media", prefix="/api/v1/media", tags=["Media"])
_include_router("history", prefix="/api/v1")

if media is not None:
    app.add_api_route(
        "/media/video-stage1/explain",
        media.explain_video_stage1,
        methods=["POST"],
        summary="영상/음성 result.json 기반 LLM 설명 생성",
        tags=["Media"],
    )


# ============ 헬스 체크 ============

@app.get("/", tags=["Health Check"])
def read_root() -> dict[str, str]:
    """서버 상태 확인"""
    return {
        "status": "running",
        "database": "connected" if database_connected else "disabled",
        "environment": app_env,
    }


@app.get("/health", tags=["Health Check"])
def health_check() -> dict[str, str]:
    """헬스 체크 엔드포인트"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))

    logger.info("VeriFake API starting: %s:%s (%s)", host, port, app_env)

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=app_env != "production",
    )
