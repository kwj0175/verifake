from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, Float
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from services.backend.database import Base


class VideoMetadata(Base):
    __tablename__ = "video_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    task_id: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        index=True,
        default=lambda: str(uuid.uuid4()),
    )

    # 경로 정보
    origin_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    download_dir: Mapped[str | None] = mapped_column(Text, nullable=True)   # 원본 다운로드 경로
    storage_path: Mapped[str | None] = mapped_column(Text, nullable=True)   # 분리된 영상 경로
    audio_path: Mapped[str | None] = mapped_column(Text, nullable=True)     # 분리된 음성 경로

    # 유저 정보
    user_id: Mapped[str | None] = mapped_column(String(50), index=True, nullable=True)
    fcm_token: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # 분석 결과
    phash_value: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="PENDING")
    # 신규 분석 결과 필드 추가 (nullable=True)
    verdict: Mapped[str | None] = mapped_column(String(50), nullable=True)
    deepfake_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<VideoMetadata task_id={self.task_id!r} status={self.status!r}>"
