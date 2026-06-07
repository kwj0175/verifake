from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from services.backend import models
from services.backend.database import get_db

router = APIRouter()


class HistoryItemResponse(BaseModel):
    # 버그 수정: Pydantic v2 경고 → class Config → model_config = ConfigDict(...)
    model_config = ConfigDict(from_attributes=True)

    id: int
    task_id: str
    status: str
    verdict: Optional[str] = None
    deepfake_score: Optional[float] = None
    title: Optional[str] = "분석 영상"
    thumbnail_path: Optional[str] = None
    created_at: Optional[datetime] = None


class HistoryListResponse(BaseModel):
    total: int
    items: List[HistoryItemResponse]
    limit: int
    offset: int


@router.get(
    "/history",
    summary="분석 기록 목록 조회",
    tags=["History"],
    response_model=HistoryListResponse,
)
async def get_history_list(
    user_id: Optional[str] = Query(None, description="특정 유저의 기록만 필터링"),
    limit: int = Query(10, ge=1, le=100, description="페이지당 항목 수"),
    offset: int = Query(0, ge=0, description="건너뛸 항목 수"),
    db: Session = Depends(get_db),
):
    """
    전체 또는 특정 유저의 분석 기록 목록을 반환합니다.
    페이지네이션(limit, offset)을 지원하며 최신순으로 정렬됩니다.
    """
    query = db.query(models.VideoMetadata)

    if user_id:
        query = query.filter(models.VideoMetadata.user_id == user_id)

    total = query.count()

    tasks = (
        query.order_by(models.VideoMetadata.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "total": total,
        "items": tasks,
        "limit": limit,
        "offset": offset,
    }
