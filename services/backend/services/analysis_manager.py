# services/backend/services/analysis_manager.py

from fcm_service import send_push_notification
# AI 파트의 파일이 나오면 여기서 추가로 임포트.
# 예시 from ai_module import detect_fake

def run_total_analysis(user_id, fcm_token):
    print(f"{user_id}님의 영상 분석중 ")

    # [STEP 1] AI 분석 결과 데이터 구성 (이미지 내용 기준)
    # 실제 연동 시에는 AI 파트 함수가 아래와 같은 딕셔너리를 반환하게 하면 됨.
    ai_results = {
        "summary": {
            "deepfake_chance": 87,
            "confidence": 64,
            "consistency": 71
        },
        "video_analysis": {
            "manipulation_chance": 91,
            "suspicious_segments": ["3:12~3:18", "5:02~5:09"],
            "detection_rate": 94
        },
        "audio_analysis": {
            "manipulation_chance": 74,
            "suspicious_segments": ["2:45~2:51", "4:30~4:35"],
            "detection_rate": 88
        }
    }

    # [STEP 2] 알림 메시지 구성
    # 간단하게 보여줄 때는 핵심 수치 위주로 작성.
    title = "정밀 분석"
    body = f"딥페이크 가능성 {ai_results['summary']['deepfake_chance']}% 감지. 상세보기에서 의심 구간을 확인하세요."
    
    # [STEP 3] 상세 데이터 연동 (extra_data)
    # 푸시 알림을 눌렀을 때 앱에서 이미지와 같은 '상세보기' 화면을 띄울 수 있도록 데이터를 다 넣음.
    extra_data = {
        "deepfake_chance": str(ai_results['summary']['deepfake_chance']),
        "video_suspicious": ", ".join(ai_results['video_analysis']['suspicious_segments']),
        "audio_suspicious": ", ".join(ai_results['audio_analysis']['suspicious_segments']),
        "analysis_type": "detailed"
    }

    # [STEP 4] FCM 발송
    send_push_notification(fcm_token, title, body, data=extra_data)


#테스트용 더미 데이터
if __name__ == "__main__":
    TEST_USER = "이름"
    TEST_TOKEN = "토큰값"

    print("===테스트 시작 ===")
    
    try:
        # 2. 전체 분석 로직 실행
        run_total_analysis(TEST_USER, TEST_TOKEN)
        print("=== ✅ 테스트 프로세스 종료 ===")
        
    except Exception as e:
        print(f"=== ❌ 테스트 중 에러 발생: {e} ===")
