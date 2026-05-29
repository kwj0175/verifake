from __future__ import annotations

from typing import Any, TypedDict

from services.backend.services.fcm_service import send_push_notification


class AnalysisSummary(TypedDict):
    """분석 요약 정보"""
    deepfake_chance: int
    confidence: int
    consistency: int


class VideoAnalysis(TypedDict):
    """영상 분석 결과"""
    manipulation_chance: int
    suspicious_segments: list[str]
    detection_rate: int


class AudioAnalysis(TypedDict):
    """음성 분석 결과"""
    manipulation_chance: int
    suspicious_segments: list[str]
    detection_rate: int


class AIAnalysisResult(TypedDict):
    """AI 분석 전체 결과"""
    summary: AnalysisSummary
    video_analysis: VideoAnalysis
    audio_analysis: AudioAnalysis


# ---------------------------------------------------------------------------
# [추가] 비디오 + 오디오 통합 verdict 계산
# ---------------------------------------------------------------------------

# 가중치: 비디오 60%, 오디오 40%
_VIDEO_WEIGHT = 0.6
_AUDIO_WEIGHT = 0.4

# FAKE 판정 임계값 (%)
_FAKE_THRESHOLD = 50.0


def compute_final_verdict(
    video_score: float | None,
    audio_score: float | None,
) -> tuple[str, float]:
    """비디오·오디오 점수를 가중 평균하여 최종 verdict와 score 반환.

    정책:
    - 둘 다 있으면 가중 평균 (video 60%, audio 40%)
    - 하나만 있으면 해당 점수만 사용
    - 둘 다 없으면 REAL / 0.0 반환

    Args:
        video_score: 0.0 ~ 100.0 범위의 비디오 딥페이크 점수 (없으면 None)
        audio_score: 0.0 ~ 100.0 범위의 오디오 딥페이크 점수 (없으면 None)

    Returns:
        (verdict, final_score) — verdict는 "FAKE" 또는 "REAL"
    """
    if video_score is not None and audio_score is not None:
        final_score = round(
            video_score * _VIDEO_WEIGHT + audio_score * _AUDIO_WEIGHT, 1
        )
    elif video_score is not None:
        final_score = round(video_score, 1)
    elif audio_score is not None:
        final_score = round(audio_score, 1)
    else:
        return "REAL", 0.0

    verdict = "FAKE" if final_score >= _FAKE_THRESHOLD else "REAL"
    return verdict, final_score


# ---------------------------------------------------------------------------
# 기존 알림 로직
# ---------------------------------------------------------------------------

def _build_notification_title(ai_results: AIAnalysisResult) -> str:
    """알림 제목 생성"""
    deepfake_chance = ai_results["summary"]["deepfake_chance"]
    if deepfake_chance >= 80:
        return "🚨 높은 위험도 감지"
    elif deepfake_chance >= 50:
        return "⚠️ 의심 신호 감지"
    else:
        return "✅ 정밀 분석"


def _build_notification_body(ai_results: AIAnalysisResult) -> str:
    """알림 본문 생성"""
    deepfake_chance = ai_results["summary"]["deepfake_chance"]
    return f"딥페이크 가능성 {deepfake_chance}% 감지. 상세보기에서 의심 구간을 확인하세요."


def _build_extra_data(ai_results: AIAnalysisResult) -> dict[str, str]:
    """푸시 알림 상세 데이터 구성"""
    video_suspicious = ", ".join(ai_results["video_analysis"]["suspicious_segments"])
    audio_suspicious = ", ".join(ai_results["audio_analysis"]["suspicious_segments"])

    return {
        "deepfake_chance": str(ai_results["summary"]["deepfake_chance"]),
        "confidence": str(ai_results["summary"]["confidence"]),
        "consistency": str(ai_results["summary"]["consistency"]),
        "video_suspicious": video_suspicious,
        "video_detection_rate": str(ai_results["video_analysis"]["detection_rate"]),
        "audio_suspicious": audio_suspicious,
        "audio_detection_rate": str(ai_results["audio_analysis"]["detection_rate"]),
        "analysis_type": "detailed",
    }


def _send_analysis_notification(
    fcm_token: str,
    ai_results: AIAnalysisResult,
) -> None:
    """분석 결과 알림 발송"""
    if not fcm_token:
        raise ValueError("FCM 토큰이 유효하지 않습니다.")

    title = _build_notification_title(ai_results)
    body = _build_notification_body(ai_results)
    extra_data = _build_extra_data(ai_results)

    send_push_notification(fcm_token, title, body, data=extra_data)


def run_total_analysis(
    user_id: str,
    fcm_token: str,
    ai_results: AIAnalysisResult | None = None,
) -> None:
    """전체 분석 실행 및 알림 발송"""
    if not user_id:
        raise ValueError("user_id가 필요합니다.")
    if not fcm_token:
        raise ValueError("fcm_token이 필요합니다.")

    print(f"{user_id}님의 영상 분석중...")

    if ai_results is None:
        ai_results = _get_dummy_ai_results()

    _send_analysis_notification(fcm_token, ai_results)

    print(f"{user_id}님의 분석 알림이 발송되었습니다.")


def _get_dummy_ai_results() -> AIAnalysisResult:
    """테스트용 더미 AI 분석 결과 생성"""
    return {
        "summary": {
            "deepfake_chance": 87,
            "confidence": 64,
            "consistency": 71,
        },
        "video_analysis": {
            "manipulation_chance": 91,
            "suspicious_segments": ["3:12~3:18", "5:02~5:09"],
            "detection_rate": 94,
        },
        "audio_analysis": {
            "manipulation_chance": 74,
            "suspicious_segments": ["2:45~2:51", "4:30~4:35"],
            "detection_rate": 88,
        },
    }


# ============ 테스트 코드 ============

if __name__ == "__main__":
    # compute_final_verdict 동작 확인
    print("=== compute_final_verdict 테스트 ===")
    cases = [
        (80.0, 60.0),   # 둘 다 있음 → 80*0.6 + 60*0.4 = 72.0
        (30.0, 20.0),   # 둘 다 낮음 → 30*0.6 + 20*0.4 = 26.0
        (90.0, None),   # 비디오만 → 90.0
        (None, 70.0),   # 오디오만 → 70.0
        (None, None),   # 둘 다 없음 → REAL / 0.0
    ]
    for v, a in cases:
        verdict, score = compute_final_verdict(v, a)
        print(f"  video={v}, audio={a} → {verdict} / {score}%")

    TEST_USER = "테스트 사용자"
    TEST_TOKEN = "test_fcm_token_value"

    print("\n=== 알림 발송 테스트 ===")
    try:
        run_total_analysis(TEST_USER, TEST_TOKEN)
        print("=== ✅ 테스트 프로세스 종료 ===")
    except Exception as exc:
        print(f"=== ❌ 테스트 중 에러 발생: {exc} ===")