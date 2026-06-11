from __future__ import annotations

import os
import sys
from typing import Any

import firebase_admin
from firebase_admin import credentials, messaging


# config 모듈 임포트
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import config
except ImportError:
    print("config.py 파일을 찾을 수 없습니다. 위치를 확인해주세요.")
    config = None  # type: ignore


class FCMInitializationError(Exception):
    """FCM 초기화 실패 시 발생하는 예외"""
    pass


class FCMNotificationError(Exception):
    """FCM 알림 발송 실패 시 발생하는 예외"""
    pass


def _validate_config() -> None:
    """config 모듈 및 Firebase 키 경로 검증
    
    Raises:
        FCMInitializationError: config가 없거나 키 경로가 유효하지 않을 때
    """
    if config is None:
        raise FCMInitializationError("config.py 파일을 찾을 수 없습니다.")
    
    if not hasattr(config, "FIREBASE_KEY_PATH"):
        raise FCMInitializationError("config.FIREBASE_KEY_PATH가 정의되지 않았습니다.")
    
    key_path = config.FIREBASE_KEY_PATH
    if not os.path.exists(key_path):
        raise FCMInitializationError(f"Firebase 키 파일이 존재하지 않습니다: {key_path}")


def _load_firebase_credentials() -> credentials.Certificate:
    """Firebase 자격증명 로드
    
    Returns:
        Firebase Certificate 객체
        
    Raises:
        FCMInitializationError: 자격증명 로드 실패 시
    """
    try:
        cred = credentials.Certificate(config.FIREBASE_KEY_PATH)
        return cred
    except Exception as exc:
        raise FCMInitializationError(f"Firebase 자격증명 로드 실패: {exc}")


def initialize_fcm() -> None:
    """FCM Admin SDK 초기화
    
    서버 시작 시 한 번만 호출되어야 함.
    
    Raises:
        FCMInitializationError: 초기화 실패 시
    """
    try:
        # config 및 Firebase 키 검증
        _validate_config()
        
        # 이미 초기화되었으면 건너뛰기
        if firebase_admin._apps:
            print("Firebase Admin SDK는 이미 초기화되었습니다.")
            return
        
        # Firebase 자격증명 로드 및 앱 초기화
        cred = _load_firebase_credentials()
        firebase_admin.initialize_app(cred)
        print("✅ Firebase Admin SDK 초기화 완료")
        
    except FCMInitializationError as exc:
        print(f"❌ Firebase 초기화 실패: {exc}")
        raise


def _validate_notification_params(
    fcm_token: str,
    title: str,
    body: str,
) -> None:
    """푸시 알림 매개변수 검증
    
    Args:
        fcm_token: FCM 토큰
        title: 알림 제목
        body: 알림 본문
        
    Raises:
        ValueError: 매개변수가 유효하지 않을 때
    """
    if not fcm_token or not isinstance(fcm_token, str):
        raise ValueError("fcm_token은 공백이 아닌 문자열이어야 합니다.")
    if not title or not isinstance(title, str):
        raise ValueError("title은 공백이 아닌 문자열이어야 합니다.")
    if not body or not isinstance(body, str):
        raise ValueError("body는 공백이 아닌 문자열이어야 합니다.")


def _build_fcm_message(
    fcm_token: str,
    title: str,
    body: str,
    data: dict[str, str] | None = None,
) -> messaging.Message:
    """FCM 메시지 객체 생성
    
    Args:
        fcm_token: FCM 토큰
        title: 알림 제목
        body: 알림 본문
        data: 추가 데이터 (선택사항)
        
    Returns:
        messaging.Message 객체
    """
    return messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        data=data or {},
        token=fcm_token,
    )


def send_push_notification(
    fcm_token: str,
    title: str,
    body: str,
    data: dict[str, str] | None = None,
) -> bool:
    """사용자에게 푸시 알림 발송
    
    Args:
        fcm_token: 기기 고유 FCM 토큰
        title: 알림 제목
        body: 알림 본문
        data: 상세 페이지 이동 등을 위한 추가 데이터 (선택사항)
        
    Returns:
        성공 여부 (True/False)
    """
    try:
        # 매개변수 검증
        _validate_notification_params(fcm_token, title, body)
        
        # FCM 메시지 구성
        message = _build_fcm_message(fcm_token, title, body, data)
        
        # Firebase 서버로 전송
        response = messaging.send(message)
        print(f"✅ 알림 전송 성공 (message_id: {response})")
        return True
        
    except ValueError as exc:
        print(f"❌ 알림 매개변수 오류: {exc}")
        return False
    except Exception as exc:
        print(f"❌ 알림 전송 실패: {exc}")
        return False


# ============ 모듈 로드 시 자동 초기화 ============

try:
    initialize_fcm()
except FCMInitializationError:
    # 초기화 실패해도 모듈 로드는 계속 진행
    # (개발 환경이나 테스트 환경에서 Firebase가 없을 수 있음)
    print("⚠️  Firebase 없이 실행 중입니다. 푸시 알림이 작동하지 않습니다.")