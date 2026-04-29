# pyright: reportMissingImports=false

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest


pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient


video = importlib.import_module("services.backend.routers.video")
exceptions = importlib.import_module("services.ai.pipelines.video_stage1.exceptions")


app_under_test = FastAPI()
app_under_test.include_router(video.router, prefix="/api/v1")
client = TestClient(app_under_test)


def test_video_stage1_preprocess_returns_400_for_missing_file() -> None:
    response = client.post(
        "/api/v1/video-stage1/preprocess",
        json={"file_path": "missing-file.mp4"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "파일이 존재하지 않습니다."


def test_video_stage1_preprocess_rejects_path_outside_project_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    external_file = tmp_path / "outside.mp4"
    external_file.write_bytes(b"data")

    monkeypatch.setattr(video, "_get_project_root", lambda: project_root)

    response = client.post(
        "/api/v1/video-stage1/preprocess",
        json={"file_path": str(external_file)},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "프로젝트 내부 파일만 허용됩니다."


def test_video_stage1_preprocess_rejects_invalid_job_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    input_path = project_root / "sample.mov"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"data")

    monkeypatch.setattr(video, "_get_project_root", lambda: project_root)

    response = client.post(
        "/api/v1/video-stage1/preprocess",
        json={"file_path": str(input_path), "job_id": "../escape"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "job_id 형식이 올바르지 않습니다."


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

    monkeypatch.setattr(video, "_get_project_root", lambda: project_root)
    monkeypatch.setattr(video, "_get_stage1_storage_root", lambda: custom_storage_root)
    monkeypatch.setattr(video, "run_video_stage1_preprocess_job", fake_run_video_stage1_preprocess_job)

    response = client.post(
        "/api/v1/video-stage1/preprocess",
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

    monkeypatch.setattr(video, "_get_project_root", lambda: project_root)
    monkeypatch.setattr(video, "run_video_stage1_preprocess_job", fake_run_video_stage1_preprocess_job)

    response = client.post(
        "/api/v1/video-stage1/preprocess",
        json={"file_path": str(input_path), "job_id": "job_test_004"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Stage1 A AI 런타임이 준비되지 않았습니다.",
    }


def test_video_stage1_preprocess_returns_sanitized_500_for_unexpected_error(
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

    monkeypatch.setattr(video, "_get_project_root", lambda: project_root)
    monkeypatch.setattr(video, "run_video_stage1_preprocess_job", fake_run_video_stage1_preprocess_job)

    response = client.post(
        "/api/v1/video-stage1/preprocess",
        json={"file_path": str(input_path), "job_id": "job_test_005"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "내부 서버 오류가 발생했습니다.",
    }
