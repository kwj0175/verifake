# pyright: reportMissingImports=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false

from __future__ import annotations

import importlib
from pathlib import Path

from fastapi.testclient import TestClient

from services.backend.main import app


client = TestClient(app)
media = importlib.import_module("services.backend.routers.media")


def test_main_exposes_media_collection_and_llm_explain_routes_not_ai_runtime_routes() -> None:
    route_paths = {
        path
        for route in app.routes
        if (path := getattr(route, "path", None)) is not None
    }

    assert "/api/v1/instagram" in route_paths
    assert "/api/v1/video" in route_paths
    assert "/api/v1/status/{task_id}" in route_paths
    assert "/media/video-stage1/explain" in route_paths

    assert "/api/v1/video-stage1/preprocess" not in route_paths
    assert "/api/v1/video-stage1/detect" not in route_paths
    assert "/api/v1/video-stage1/detect/jobs" not in route_paths
    assert "/api/v1/audio/jobs" not in route_paths
    assert "/api/v1/audio/jobs/{task_id}" not in route_paths
    assert "/api/v1/audio/jobs/{task_id}/result" not in route_paths
    assert "/media/video-stage1/preprocess" not in route_paths
    assert "/media/video-stage1/detect" not in route_paths
    assert "/media/instagram" not in route_paths
    assert "/media/video" not in route_paths
    assert "/media/status/{task_id}" not in route_paths


def test_ai_runtime_routes_are_not_exposed() -> None:
    assert client.post("/api/v1/video-stage1/preprocess").status_code == 404
    assert client.post("/api/v1/video-stage1/detect").status_code == 404
    assert client.post("/api/v1/video-stage1/detect/jobs").status_code == 404
    assert client.post("/api/v1/audio/jobs").status_code == 404
    assert client.post("/media/instagram").status_code == 404
    assert client.post("/media/video").status_code == 404
    assert client.get("/media/status/some-id").status_code == 404


def test_main_llm_explain_route_uses_generated_audio_and_video_result_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_result_path = tmp_path / "video" / "result.json"
    audio_result_path = tmp_path / "audio" / "audio_stage1_result.json"
    video_result_path.parent.mkdir(parents=True)
    audio_result_path.parent.mkdir(parents=True)
    _ = video_result_path.write_text("{}", encoding="utf-8")
    _ = audio_result_path.write_text("{}", encoding="utf-8")

    seen_paths: list[tuple[Path, Path]] = []

    def fake_run_video_stage1_result_explainer_job(
        video_json_path: Path,
        audio_json_path: Path,
    ) -> dict[str, object]:
        seen_paths.append((video_json_path, audio_json_path))
        return {
            "job_id": "job_test_301",
            "status": "success",
            "detection": {
                "video_score": {
                    "final_fake_score": 0.42,
                    "max_fake_score": 0.73,
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

    assert response.status_code == 200
    assert seen_paths == [(video_result_path, audio_result_path)]
    assert response.json()["llm_explanations"] == {
        "summary_text": "요약",
        "detail_text": "상세",
    }
