"""Pydantic schemas for the audio detection pipeline."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class EvidenceLevel(str, Enum):
    """음성파일이 얼마나 분석가능한 수준인지 (사람음성같은게 충분히 포함되어있는지)"""

    SUFFICIENT = "sufficient"
    LOW_EVIDENCE = "low_evidence"
    UNSUPPORTED_CONTENT = "unsupported_content"


class QualityFlag(str, Enum):
    """전처리 중에서 나오는 정보"""

    LONG_LEADING_SILENCE = "long_leading_silence"
    LONG_TRAILING_SILENCE = "long_trailing_silence"
    HIGH_SILENCE_RATIO = "high_silence_ratio"
    LOW_BITRATE_SOURCE = "low_bitrate_source"
    LOW_SPEECH_RATIO = "low_speech_ratio"
    NO_HUMAN_SPEECH = "no_human_speech"
    TOO_SHORT = "too_short"
    CLIPPING_DETECTED = "clipping_detected"
    HEAVY_BACKGROUND_NOISE = "heavy_background_noise"


class AudioAnalysisRequest(BaseModel):
    """입력 리퀘스트"""

    request_id: str = Field(..., min_length=1, description="request ID.")
    file_path: str = Field(..., min_length=1, description="오디오 파일 경로")


class OriginalAudioMetadata(BaseModel):

    codec: Optional[str] = None
    bitrate: Optional[int] = Field(
        default=None,
        ge=0,
        description="원본 비트레이트 / 단위 : bits per second",
    )
    duration_sec: float = Field(..., ge=0.0 ,description="전체오디오길이")
    sample_rate_hz: int = Field(..., gt=0, description= "샘플링 레이트")
    channel_count: int = Field(..., gt=0, description="채널 수")


class SuspiciousAudioSegment(BaseModel):
    """연속된 의심구간 병합"""

    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., gt=0.0)
    fake_score_raw: float
    real_score_raw: float
    fake_prob_like: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., ge=1)

    @model_validator(mode="after")
    def validate_time_range(self) -> "SuspiciousAudioSegment":
        if self.end_sec <= self.start_sec:
            raise ValueError("구간 끝 시간이 항상 시작보다 커야함")
        return self


class AudioAnalysisResult(BaseModel):
    """1단계 출력"""

    request_id: str = Field(..., min_length=1)
    file_path: str = Field(..., min_length=1)
    original_metadata: OriginalAudioMetadata

    audio_fake_score_raw: Optional[float] = None
    audio_real_score_raw: Optional[float] = None
    audio_fake_prob_like: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    audio_uncertainty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="높을수록 결과가 덜 안정적이거나 충분한 근거가 부족하다는 뜻",
    )
    scored_window_count: int = Field(..., ge=0)
    failed_window_count: int = Field(..., ge=0)
    skipped_window_count: int = Field(..., ge=0)
    audio_model_error: Optional[str] = None

    human_speech_detected: bool
    evidence_level: EvidenceLevel = Field(
        ...,
        description="전체 결과를 해석할 때, 근거가 충분한가",
    )

    speech_duration_sec: float = Field(..., ge=0.0, description="사람 음성으로 판단된 길이")
    speech_ratio: float = Field(..., ge=0.0, le=1.0, description="말하는 구간 비율")
    silence_ratio: float = Field(..., ge=0.0, le=1.0, description="무음 비율")
    pause_count: int = Field(
        ...,
        ge=0,
        description="기준보다 긴 비발화 구간의 개수",
    )
    leading_silence_sec: float = Field(..., ge=0.0, description="앞부분 무음 길이")
    trailing_silence_sec: float = Field(..., ge=0.0, description="뒷부분 무음길이")

    quality_flags: List[QualityFlag] = Field(default_factory=list)
    top_suspicious_audio_segments: List[SuspiciousAudioSegment] = Field(default_factory=list)

    """세그먼트 랭크 중복 검사"""
    @field_validator("top_suspicious_audio_segments")
    @classmethod
    def validate_segment_ranks(
        cls, segments: List[SuspiciousAudioSegment]
    ) -> List[SuspiciousAudioSegment]:
        ranks = [segment.rank for segment in segments]
        if len(ranks) != len(set(ranks)):
            raise ValueError("segment ranks must be unique")
        return segments

    """전체 결과 검사"""
    @model_validator(mode="after")
    def validate_consistency(self) -> "AudioAnalysisResult":
        ratio_sum = self.speech_ratio + self.silence_ratio
        if abs(ratio_sum - 1.0) > 0.05:
            raise ValueError("speech + silence 비율은 대략 1.0, +_0.05 허용")

        if not self.human_speech_detected and self.speech_duration_sec > 0:
            raise ValueError(
                "사람 음성 없는 경우 speech 길이는 0이어야함"
            )

        if (
            self.evidence_level == EvidenceLevel.UNSUPPORTED_CONTENT
            and self.human_speech_detected
        ):
            raise ValueError(
                "분석 불가 음성은 사람 음성 감지 됐다고 하면 안됨"
            )

        return self
