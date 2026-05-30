from __future__ import annotations

from sqlalchemy.orm import Session

from services.backend import models
from services.backend.services.fcm_service import send_push_notification


def get_latest_fcm_token_by_user(db: Session, user_id: str) -> str | None:
    """특정 유저의 가장 최신 FCM 토큰을 반환합니다."""
    row = (
        db.query(models.VideoMetadata.fcm_token)
        .filter(
            models.VideoMetadata.user_id == user_id,
            models.VideoMetadata.fcm_token.isnot(None),
        )
        .order_by(models.VideoMetadata.created_at.desc())
        .first()
    )
    return row[0] if row else None


def get_task_by_id(db: Session, task_id: str) -> models.VideoMetadata | None:
    return (
        db.query(models.VideoMetadata)
        .filter(models.VideoMetadata.task_id == task_id)
        .first()
    )


def update_user_fcm_token(db: Session, user_id: str, fcm_token: str) -> bool:
    """특정 유저의 가장 최근 태스크 기록에 FCM 토큰을 갱신하거나 등록합니다."""
    last_task = (
        db.query(models.VideoMetadata)
        .filter(models.VideoMetadata.user_id == user_id)
        .order_by(models.VideoMetadata.created_at.desc())
        .first()
    )
    if last_task:
        last_task.fcm_token = fcm_token
        db.commit()
        return True
    return False


def update_task_analysis(
    db: Session,
    task_id: str,
    verdict: str,
    deepfake_score: float,
    # 버그 수정: 기본값 "DONE" → "COMPLETED" (파이프라인 전체에서 "COMPLETED" 사용)
    status: str = "COMPLETED",
) -> models.VideoMetadata | None:
    """분석 완료 시 판정 결과 및 스코어를 저장하고, 해당 유저에게 푸시 알림을 발송합니다."""
    task = get_task_by_id(db, task_id)
    if task:
        task.verdict = verdict
        task.deepfake_score = deepfake_score
        task.status = status
        db.commit()
        db.refresh(task)

        if task.fcm_token:
            title = "영상 분석 완료 🔍"
            body = f"요청하신 영상의 분석이 완료되었습니다. 판정: {verdict}"
            data = {
                "task_id": task.task_id,
                "verdict": verdict,
                "deepfake_score": str(deepfake_score),
            }
            send_push_notification(
                fcm_token=task.fcm_token,
                title=title,
                body=body,
                data=data,
            )

    return task
