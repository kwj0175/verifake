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


def test_video_stage1_preprocess_returns_400_for_missing_file() -> None:
    response = client.post(
        "/media/video-stage1/preprocess",
        json={"file_path": "missing-file.mp4"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "파일이 존재하지 않습니다."


def test_video_stage1_preprocess_accepts_existing_file_without_project_root_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    external_file = tmp_path / "outside.mp4"
    external_file.write_bytes(b"data")

    def fake_run_video_stage1_preprocess_job(
        input_file: Path,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        assert input_file == external_file
        assert job_id is None
        return {"job_id": "job_test_001", "status": "success"}

    monkeypatch.setattr(media, "run_video_stage1_preprocess_job", fake_run_video_stage1_preprocess_job)

    response = client.post(
        "/media/video-stage1/preprocess",
        json={"file_path": str(external_file)},
    )

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job_test_001",
        "status": "success",
        "preprocessing_json": "storage/jobs/job_test_001/metadata/preprocessing.json",
    }


def test_video_stage1_preprocess_forwards_job_id_to_processor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    input_path = project_root / "sample.mov"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"data")

    def fake_run_video_stage1_preprocess_job(
        input_file: Path,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        assert input_file == input_path
        assert job_id == "../escape"
        return {"job_id": job_id or "job_test_002", "status": "success"}

    monkeypatch.setattr(media, "run_video_stage1_preprocess_job", fake_run_video_stage1_preprocess_job)

    response = client.post(
        "/media/video-stage1/preprocess",
        json={"file_path": str(input_path), "job_id": "../escape"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "../escape",
        "status": "success",
        "preprocessing_json": "storage/jobs/../escape/metadata/preprocessing.json",
    }


def test_video_stage1_preprocess_returns_summary_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    input_path = project_root / "fixtures" / "sample.mov"
    custom_storage_root = tmp_path / "custom-storage" / "jobs"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"data")

    def fake_run_video_stage1_preprocess_job(
        input_file: Path,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        assert input_file == input_path
        return {
            "job_id": job_id or "job_test_003",
            "status": "success",
        }

    monkeypatch.setattr(media, "run_video_stage1_preprocess_job", fake_run_video_stage1_preprocess_job)
    monkeypatch.setattr(
        media,
        "build_job_paths",
        lambda job_id: {
            "preprocessing_json_path": custom_storage_root / job_id / "metadata" / "preprocessing.json",
        },
    )

    response = client.post(
        "/media/video-stage1/preprocess",
        json={"file_path": str(input_path), "job_id": "job_test_003"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job_test_003",
        "status": "success",
        "preprocessing_json": str(custom_storage_root / "job_test_003" / "metadata" / "preprocessing.json"),
    }


def test_video_stage1_preprocess_returns_500_for_missing_ai_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    input_path = project_root / "sample.mov"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"data")

    def fake_run_video_stage1_preprocess_job(
        input_file: Path,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        raise exceptions.Stage1UnavailableError("missing ai runtime")

    monkeypatch.setattr(media, "run_video_stage1_preprocess_job", fake_run_video_stage1_preprocess_job)

    response = client.post(
        "/media/video-stage1/preprocess",
        json={"file_path": str(input_path), "job_id": "job_test_004"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "missing ai runtime",
    }


def test_video_stage1_preprocess_returns_raw_500_detail_for_unexpected_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    input_path = project_root / "sample.mov"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"data")

    def fake_run_video_stage1_preprocess_job(
        input_file: Path,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError("sensitive detail")

    monkeypatch.setattr(media, "run_video_stage1_preprocess_job", fake_run_video_stage1_preprocess_job)

    response = client.post(
        "/media/video-stage1/preprocess",
        json={"file_path": str(input_path), "job_id": "job_test_005"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "sensitive detail",
    }
