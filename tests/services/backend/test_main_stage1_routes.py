# pyright: reportMissingImports=false

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from services.backend.main import app
from services.backend.routers import video


client = TestClient(app)


def test_main_only_exposes_stage1_runtime_routes() -> None:
    route_paths = {
        path
        for route in app.routes
        if (path := getattr(route, "path", None)) is not None
    }

    assert "/api/v1/video-stage1/preprocess" in route_paths
    assert "/api/v1/video-stage1/detect" in route_paths
    assert "/media/video-stage1/preprocess" in route_paths
    assert "/media/video-stage1/detect" in route_paths

    assert "/api/v1/instagram" not in route_paths
    assert "/api/v1/video" not in route_paths
    assert "/api/v1/status/{task_id}" not in route_paths
    assert "/media/instagram" not in route_paths
    assert "/media/video" not in route_paths
    assert "/media/status/{task_id}" not in route_paths


def test_legacy_video_ingest_routes_are_not_exposed() -> None:
    assert client.post("/api/v1/instagram").status_code == 404
    assert client.post("/api/v1/video").status_code == 404
    assert client.get("/api/v1/status/some-id").status_code == 404
    assert client.post("/media/instagram").status_code == 404
    assert client.post("/media/video").status_code == 404
    assert client.get("/media/status/some-id").status_code == 404


def test_main_exposes_media_stage1_preprocess_route(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    input_path = project_root / "sample.mov"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"data")

    def fake_run_video_stage1_preprocess_job(file_path: Path, job_id: str | None = None) -> dict[str, str]:
        return {"job_id": job_id or "job_media_001", "status": "success"}

    monkeypatch.setattr(
        video,
        "run_video_stage1_preprocess_job",
        fake_run_video_stage1_preprocess_job,
    )
    monkeypatch.setattr(video, "_get_project_root", lambda: project_root)

    response = client.post(
        "/media/video-stage1/preprocess",
        json={"file_path": str(input_path), "job_id": "job_media_001"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job_media_001",
        "status": "success",
        "preprocessing_json": "storage/jobs/job_media_001/metadata/preprocessing.json",
    }


def test_main_exposes_media_stage1_detect_route(
    monkeypatch,
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "storage" / "jobs"
    preprocessing_json = storage_root / "job_media_002" / "metadata" / "preprocessing.json"
    preprocessing_json.parent.mkdir(parents=True, exist_ok=True)
    preprocessing_json.write_text("{}", encoding="utf-8")

    def fake_run_video_stage1_detection(preprocessing_json_path: str) -> dict[str, str]:
        return {"job_id": "job_media_002", "status": "success"}

    monkeypatch.setattr(
        video,
        "run_video_stage1_detection",
        fake_run_video_stage1_detection,
    )
    monkeypatch.setattr(video, "_get_stage1_storage_root", lambda: storage_root)

    response = client.post(
        "/media/video-stage1/detect",
        json={"preprocessing_json": str(preprocessing_json)},
    )

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job_media_002",
        "status": "success",
        "detection_json": str(preprocessing_json.parent.parent / "output" / "detection.json"),
        "result_json": str(preprocessing_json.parent.parent / "output" / "result.json"),
    }
