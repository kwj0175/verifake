# pyright: reportMissingImports=false

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from services.backend.routers import video


def test_video_stage1_detect_rejects_missing_preprocessing_json() -> None:
    with pytest.raises(HTTPException) as exc_info:
        video.detect_video_stage1(
            video.VideoStage1DetectRequest(
                preprocessing_json="/tmp/does-not-exist/preprocessing.json"
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "preprocessing.json 파일이 존재하지 않습니다."


def test_video_stage1_detect_rejects_preprocessing_json_outside_stage1_storage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "storage" / "jobs"
    external_preprocessing_json = tmp_path / "outside" / "preprocessing.json"
    external_preprocessing_json.parent.mkdir(parents=True, exist_ok=True)
    external_preprocessing_json.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(video, "_get_stage1_storage_root", lambda: storage_root)

    with pytest.raises(HTTPException) as exc_info:
        video.detect_video_stage1(
            video.VideoStage1DetectRequest(
                preprocessing_json=str(external_preprocessing_json)
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Stage1 storage_root 내부 preprocessing.json만 허용됩니다."


def test_video_stage1_detect_returns_result_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "storage" / "jobs"
    preprocessing_json = storage_root / "job_001" / "metadata" / "preprocessing.json"
    preprocessing_json.parent.mkdir(parents=True, exist_ok=True)
    preprocessing_json.write_text("{}", encoding="utf-8")

    def fake_run_video_stage1_detection(preprocessing_json_path: str) -> dict[str, str]:
        assert preprocessing_json_path == str(preprocessing_json)
        return {"job_id": "job_001"}

    monkeypatch.setattr(
        video,
        "run_video_stage1_detection",
        fake_run_video_stage1_detection,
    )
    monkeypatch.setattr(video, "_get_stage1_storage_root", lambda: storage_root)

    response = video.detect_video_stage1(
        video.VideoStage1DetectRequest(
            preprocessing_json=str(preprocessing_json)
        )
    )

    assert response == {
        "job_id": "job_001",
        "status": "success",
        "detection_json": str(storage_root / "job_001" / "output" / "detection.json"),
        "result_json": str(storage_root / "job_001" / "output" / "result.json"),
    }


def test_video_stage1_detect_returns_sanitized_500_for_unexpected_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "storage" / "jobs"
    preprocessing_json = storage_root / "job_001" / "metadata" / "preprocessing.json"
    preprocessing_json.parent.mkdir(parents=True, exist_ok=True)
    preprocessing_json.write_text("{}", encoding="utf-8")

    def fake_run_video_stage1_detection(preprocessing_json_path: str) -> dict[str, str]:
        raise RuntimeError("sensitive detail")

    monkeypatch.setattr(video, "_get_stage1_storage_root", lambda: storage_root)
    monkeypatch.setattr(
        video,
        "run_video_stage1_detection",
        fake_run_video_stage1_detection,
    )

    with pytest.raises(HTTPException) as exc_info:
        video.detect_video_stage1(
            video.VideoStage1DetectRequest(
                preprocessing_json=str(preprocessing_json)
            )
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "내부 서버 오류가 발생했습니다."
