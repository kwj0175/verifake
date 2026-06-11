from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Body  # [+ 추가] Body
from sqlalchemy.orm import Session

from services.backend import crud
from services.backend.database import get_db

router = APIRouter()


@router.get("/user/{user_id}/fcm-token", summary="유저의 최신 FCM 토큰 조회", tags=["Status"])
async def read_user_fcm_token(user_id: str, db: Session = Depends(get_db)) -> dict:
    token = crud.get_latest_fcm_token_by_user(db, user_id)
    if not token:
        raise HTTPException(status_code=404, detail="해당 유저의 FCM 토큰을 찾을 수 없습니다.")
    return {"user_id": user_id, "fcm_token": token}

#🟥 POST /api/v1/user/{user_id}/fcm-token 엔드포인트 신규 추가
@router.post(                                                                    # [+ 추가]
    "/user/{user_id}/fcm-token",                                                 # [+ 추가]
    summary="유저 FCM 토큰 등록/갱신",                                              # [+ 추가]
    tags=["Status"]  # 기존 태그와 통일성 유지                                        # [+ 추가]
)                                                                                # [+ 추가]
async def register_fcm_token(                                                   # [+ 추가]
    user_id: str,                                                                # [+ 추가]
    fcm_token: str = Body(..., embed=True, description="FCM 기기 토큰"),          # [+ 추가]
    db: Session = Depends(get_db)                                                # [+ 추가]
) -> dict:                                                                       # [+ 추가]
    """                                                                          # [+ 추가]
    클라이언트 앱(모바일 등)에서 발급받은 FCM 토큰을 유저 ID와 매핑하여 서버에 등록 및 갱신합니다. # [+ 추가]
    """                                                                          # [+ 추가]
    success = crud.update_user_fcm_token(db, user_id=user_id, fcm_token=fcm_token)# [+ 추가]
    if not success:                                                              # [+ 추가]
        raise HTTPException(                                                     # [+ 추가]
            status_code=404,                                                     # [+ 추가]
            detail="해당 유저의 작업(태스크) 기록을 찾을 수 없어 토큰을 등록하지 못했습니다."# [+ 추가]
        )                                                                        # [+ 추가]
    return {"message": "FCM 토큰이 성공적으로 등록되었습니다."}                         # [+ 추가]