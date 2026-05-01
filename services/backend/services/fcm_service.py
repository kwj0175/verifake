import firebase_admin
from firebase_admin import credentials, messaging
import os
import sys

# config인식
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import config
except ImportError:
    print("config.py 파일을 찾을 수 없습니다. 위치를 확인해주세요.")

def initialize_fcm():
    """서버 시작 시 FCM Admin SDK초기화"""
    try:
        if not firebase_admin._apps:
            # JSON 키 경로를 가져오기.
            cred = credentials.Certificate(config.FIREBASE_KEY_PATH)
            firebase_admin.initialize_app(cred)
            print("Firebase Admin SDK 초기화 완료")
    except Exception as e:
        print(f" Firebase 초기화 실패: {e}")

def send_push_notification(fcm_token, title, body, data=None):
    """
    사용자에게 푸시 알림을 발송합니다.
    :fcm_token: 기기 고유 토큰
    :title: 알림 제목
    :body: 알림 내용
    :data: 상세 페이지 이동 등을 위한 추가 데이터 (딕셔너리)
    """
    # 메시지 구성
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        data=data,
        token=fcm_token,
    )

    try:
        # Firebase 서버로 전송
        response = messaging.send(message)
        print(f"알림 전송 성공: {response}")
        return True
    except Exception as e:
        print(f" 알림 전송 실패: {e}")
        return False

# 모듈 로드 시 자동 초기화 실행
initialize_fcm()
