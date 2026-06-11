from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest


pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient


media = importlib.import_module("services.backend.routers.media")
exceptions = importlib.import_module("services.ai.pipelines.video_stage1.exceptions")


app_under_test = FastAPI()
app_under_test.include_router(media.router, prefix="/media")
client = TestClient(app_under_test)


def test_video_stage1_explain_returns_400_for_missing_file() -> None:
    response = client.post(
        "/media/video-stage1/explain",
        json={
            "video_result_json": "missing-video-result.json",
            "audio_result_json": "missing-audio-result.json",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "video_result.json 파일이 존재하지 않습니다."


def test_video_stage1_explain_returns_400_for_missing_audio_file(
    tmp_path: Path,
) -> None:
    video_result_path = tmp_path / "video_result.json"
    video_result_path.write_text("{}", encoding="utf-8")

    response = client.post(
        "/media/video-stage1/explain",
        json={
            "video_result_json": str(video_result_path),
            "audio_result_json": str(tmp_path / "missing-audio-result.json"),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "audio_result.json 파일이 존재하지 않습니다."


def test_video_stage1_explain_returns_explanation_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_result_path = tmp_path / "video_result.json"
    audio_result_path = tmp_path / "audio_result.json"
    video_result_path.write_text("{}", encoding="utf-8")
    audio_result_path.write_text("{}", encoding="utf-8")

    def fake_run_video_stage1_result_explainer_job(
        video_json_path: Path,
        audio_json_path: Path,
    ) -> dict[str, Any]:
        assert video_json_path == video_result_path
        assert audio_json_path == audio_result_path
        return {
            "job_id": "job_test_201",
            "status": "success",
            "quality_metrics": {
                "face_detect_ratio": 0.93,
                "face_visibility_ratio": 0.88,
                "blur_score": 0.12,
                "dark_frame_ratio": 0.03,
            },
            "detection": {
                "video_score": {
                    "final_fake_score": 0.724,
                    "max_fake_score": 0.91,
                    "aggregation_method": "topk_mean",
                },
                "top_segments": [
                    {
                        "start_sec": 0,
                        "end_sec": 5,
                        "segment_score": 0.91,
                        "reason": "얼굴 경계가 부자연스럽게 보임",
                        "representative_frame_path": "storage/jobs/job_test_201/frames/frame_001.jpg",
                    }
                ],
            },
            "llm_explanations": {
                "summary_text": "요약",
                "detail_text": "상세",
            },
        }

    monkeypatch.setattr(
        media,
        "run_video_stage1_result_explainer_job",
        fake_run_video_stage1_result_explainer_job,
    )

    response = client.post(
        "/media/video-stage1/explain",
        json={
            "video_result_json": str(video_result_path),
            "audio_result_json": str(audio_result_path),
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job_test_201",
        "source_status": "success",
        "explain_status": "success",
        "video_result_json": video_result_path.as_posix(),
        "audio_result_json": audio_result_path.as_posix(),
        "llm_explanations": {
            "summary_text": "요약",
            "detail_text": "상세",
        },
        "result": {
            "status": "FAKE",
            "score": 72.4,
            "maxScore": 91.0,
            "aggregationMethod": "topk_mean",
            "llmSummary": "요약",
            "llmDetail": "상세",
            "llmReason": "상세",
            "thumbnailUrl": None,
            "thumbnailPath": "storage/jobs/job_test_201/frames/frame_001.jpg",
            "videoDetection": "감지됨",
            "audioDetection": "미분석",
            "segments": [
                {
                    "startSec": 0,
                    "endSec": 5,
                    "score": 91.0,
                    "reason": "얼굴 경계가 부자연스럽게 보임",
                    "framePath": "storage/jobs/job_test_201/frames/frame_001.jpg",
                }
            ],
            "quality": {
                "faceDetectRatio": 0.93,
                "faceVisibilityRatio": 0.88,
                "blurScore": 0.12,
                "darkFrameRatio": 0.03,
            },
        },
    }


def test_video_stage1_explain_returns_400_for_invalid_result_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_result_path = tmp_path / "video_result.json"
    audio_result_path = tmp_path / "audio_result.json"
    video_result_path.write_text("{}", encoding="utf-8")
    audio_result_path.write_text("{}", encoding="utf-8")

    def fake_run_video_stage1_result_explainer_job(
        video_json_path: Path,
        audio_json_path: Path,
    ) -> dict[str, Any]:
        raise ValueError("result.json must include detection.video_score.")

    monkeypatch.setattr(
        media,
        "run_video_stage1_result_explainer_job",
        fake_run_video_stage1_result_explainer_job,
    )

    response = client.post(
        "/media/video-stage1/explain",
        json={
            "video_result_json": str(video_result_path),
            "audio_result_json": str(audio_result_path),
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "result.json must include detection.video_score.",
    }


def test_video_stage1_explain_returns_400_for_invalid_score_value(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_result_path = tmp_path / "video_result.json"
    audio_result_path = tmp_path / "audio_result.json"
    video_result_path.write_text("{}", encoding="utf-8")
    audio_result_path.write_text("{}", encoding="utf-8")

    def fake_run_video_stage1_result_explainer_job(
        video_json_path: Path,
        audio_json_path: Path,
    ) -> dict[str, Any]:
        return {
            "job_id": "job_test_202",
            "status": "success",
            "detection": {
                "video_score": {
                    "final_fake_score": "bad",
                    "max_fake_score": 0.91,
                    "aggregation_method": "topk_mean",
                },
                "top_segments": [],
            },
            "llm_explanations": {
                "summary_text": "요약",
                "detail_text": "상세",
            },
        }

    monkeypatch.setattr(
        media,
        "run_video_stage1_result_explainer_job",
        fake_run_video_stage1_result_explainer_job,
    )

    response = client.post(
        "/media/video-stage1/explain",
        json={
            "video_result_json": str(video_result_path),
            "audio_result_json": str(audio_result_path),
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "result.json contains a non-numeric score value.",
    }


def test_video_stage1_explain_returns_500_for_missing_ai_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_result_path = tmp_path / "video_result.json"
    audio_result_path = tmp_path / "audio_result.json"
    video_result_path.write_text("{}", encoding="utf-8")
    audio_result_path.write_text("{}", encoding="utf-8")

    def fake_run_video_stage1_result_explainer_job(
        video_json_path: Path,
        audio_json_path: Path,
    ) -> dict[str, Any]:
        raise exceptions.Stage1UnavailableError("missing gemini runtime")

    monkeypatch.setattr(
        media,
        "run_video_stage1_result_explainer_job",
        fake_run_video_stage1_result_explainer_job,
    )

    response = client.post(
        "/media/video-stage1/explain",
        json={
            "video_result_json": str(video_result_path),
            "audio_result_json": str(audio_result_path),
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "missing gemini runtime",
    }
