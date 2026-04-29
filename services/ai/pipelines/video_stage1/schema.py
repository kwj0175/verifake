from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


PipelineStatus = Literal["success", "partial_success", "failed"]


class Stage1InputInfo(BaseModel):
    original_video_path: str = Field(..., min_length=1)
    normalized_video_path: str = Field(..., min_length=1)
    original_extension: str = Field(..., min_length=1)
    internal_format: str = Field(..., min_length=1)


class VideoMetadata(BaseModel):
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    fps: float = Field(..., gt=0.0)
    duration_sec: float = Field(..., ge=0.0)
    total_frame_count: int = Field(..., ge=0)
    sample_fps: float = Field(..., gt=0.0)
    sampled_frame_count: int = Field(..., ge=0)


class QualityMetrics(BaseModel):
    face_detect_ratio: float = Field(..., ge=0.0, le=1.0)
    face_visibility_ratio: float = Field(..., ge=0.0, le=1.0)
    avg_face_size_ratio: float = Field(..., ge=0.0, le=1.0)
    min_face_size_ratio: float = Field(..., ge=0.0, le=1.0)
    max_face_size_ratio: float = Field(..., ge=0.0, le=1.0)
    blur_score: float = Field(..., ge=0.0, le=1.0)
    motion_blur_ratio: float = Field(..., ge=0.0, le=1.0)
    dark_frame_ratio: float = Field(..., ge=0.0, le=1.0)
    compression_artifact_score: float = Field(..., ge=0.0, le=1.0)


class FaceSummary(BaseModel):
    human_face_detected: bool
    face_detect_failed_frame_count: int = Field(..., ge=0)
    max_face_count_per_frame: int = Field(..., ge=0)
    avg_face_count_per_frame: float = Field(..., ge=0.0)
    multi_face_flag: bool
    face_track_stability: float = Field(..., ge=0.0, le=1.0)


class FaceEntry(BaseModel):
    face_id: str = Field(..., min_length=1)
    face_index: int = Field(..., ge=0)
    detected: bool
    bbox: list[int] = Field(..., min_length=4, max_length=4)
    bbox_area_ratio: float = Field(..., ge=0.0, le=1.0)
    detection_confidence: float = Field(..., ge=0.0, le=1.0)
    crop_path: str = Field(..., min_length=1)


class FrameEntry(BaseModel):
    frame_index: int = Field(..., ge=0)
    timestamp_sec: float = Field(..., ge=0.0)
    frame_path: str = Field(..., min_length=1)
    face_count: int = Field(..., ge=0)
    faces: list[FaceEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_face_count(self) -> "FrameEntry":
        if self.face_count != len(self.faces):
            raise ValueError("face_count must match the number of faces entries")
        return self


class PreprocessingDocument(BaseModel):
    schema_version: str = Field(..., min_length=1)
    job_id: str = Field(..., min_length=1)
    pipeline_stage: int = Field(..., ge=1)
    status: PipelineStatus
    created_at: str = Field(..., min_length=1)
    input: Stage1InputInfo
    video_metadata: VideoMetadata
    quality_metrics: QualityMetrics
    face_summary: FaceSummary
    frames: list[FrameEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
