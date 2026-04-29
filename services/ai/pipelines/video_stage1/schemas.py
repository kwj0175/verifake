"""Schemas and config helpers for the video Stage 1 B pipeline."""
# pyright: reportMissingImports=false

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class PipelineStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class InferenceStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


def load_stage1_config() -> dict[str, Any]:
    config_path = Path(__file__).with_name("config.stage1.json")
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


class DetectorMetadata(BaseModel):
    framework: str
    model_name: str
    detector_type: str
    score_type: str
    score_range: tuple[float, float]


class InferenceSummary(BaseModel):
    analyzed_face_crop_count: int = Field(..., ge=0)
    analyzed_frame_count: int = Field(..., ge=0)
    skipped_frame_count: int = Field(..., ge=0)
    inference_status: InferenceStatus


class FaceScore(BaseModel):
    face_id: str
    frame_index: int = Field(..., ge=0)
    timestamp_sec: float = Field(..., ge=0.0)
    crop_path: str
    raw_fake_score: float = Field(..., ge=0.0, le=1.0)
    inference_success: bool


class FrameScore(BaseModel):
    frame_index: int = Field(..., ge=0)
    timestamp_sec: float = Field(..., ge=0.0)
    face_count: int = Field(..., ge=0)
    max_fake_score: float = Field(..., ge=0.0, le=1.0)
    avg_fake_score: float = Field(..., ge=0.0, le=1.0)
    score_source: str


class SegmentScore(BaseModel):
    segment_id: str
    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., ge=0.0)
    duration_sec: float = Field(..., ge=0.0)
    frame_count: int = Field(..., ge=0)
    max_fake_score: float = Field(..., ge=0.0, le=1.0)
    avg_fake_score: float = Field(..., ge=0.0, le=1.0)
    representative_frame_index: int | None = Field(default=None, ge=0)
    representative_frame_path: str | None = None


class TopSegment(BaseModel):
    rank: int = Field(..., ge=1)
    segment_id: str
    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., ge=0.0)
    segment_score: float = Field(..., ge=0.0, le=1.0)
    reason: str | None = None
    representative_frame_path: str | None = None


class VideoScore(BaseModel):
    max_fake_score: float = Field(..., ge=0.0, le=1.0)
    topk_mean_fake_score: float = Field(..., ge=0.0, le=1.0)
    avg_fake_score: float = Field(..., ge=0.0, le=1.0)
    final_fake_score: float = Field(..., ge=0.0, le=1.0)
    aggregation_method: str


class DetectionOutput(BaseModel):
    schema_version: str
    job_id: str
    pipeline_stage: int = Field(..., ge=1)
    status: PipelineStatus
    created_at: str
    detector: DetectorMetadata
    inference_summary: InferenceSummary
    face_scores: list[FaceScore]
    frame_scores: list[FrameScore]
    segment_scores: list[SegmentScore]
    top_segments: list[TopSegment]
    video_score: VideoScore
    errors: list[dict[str, Any] | str]


class ResultInput(BaseModel):
    normalized_video_path: str


class ResultVideoMetadata(BaseModel):
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    fps: float = Field(..., gt=0.0)
    duration_sec: float = Field(..., ge=0.0)
    sample_fps: float = Field(..., gt=0.0)


class QualityMetrics(BaseModel):
    face_detect_ratio: float = Field(..., ge=0.0, le=1.0)
    face_visibility_ratio: float = Field(..., ge=0.0, le=1.0)
    avg_face_size_ratio: float = Field(..., ge=0.0, le=1.0)
    blur_score: float = Field(..., ge=0.0, le=1.0)
    motion_blur_ratio: float = Field(..., ge=0.0, le=1.0)
    dark_frame_ratio: float = Field(..., ge=0.0, le=1.0)
    compression_artifact_score: float = Field(..., ge=0.0, le=1.0)


class FaceSummary(BaseModel):
    human_face_detected: bool
    multi_face_flag: bool
    face_track_stability: float = Field(..., ge=0.0, le=1.0)


class ResultVideoScore(BaseModel):
    final_fake_score: float = Field(..., ge=0.0, le=1.0)
    max_fake_score: float = Field(..., ge=0.0, le=1.0)
    aggregation_method: str


class ResultDetection(BaseModel):
    detector: str
    video_score: ResultVideoScore
    top_segments: list[TopSegment]


class ResultOutput(BaseModel):
    schema_version: str
    job_id: str
    pipeline_stage: int = Field(..., ge=1)
    status: PipelineStatus
    input: ResultInput
    video_metadata: ResultVideoMetadata
    quality_metrics: QualityMetrics
    face_summary: FaceSummary
    detection: ResultDetection
    stage1_note: str
