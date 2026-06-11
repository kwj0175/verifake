from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import BackgroundTasks
from fastapi import HTTPException
from fastapi.testclient import TestClient


class AudioRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from services.backend.tasks import clear_stores
        clear_stores()

    async def test_submit_rejects_missing_source_file(self) -> None:
        from services.backend.routers.audio import AudioAnalyzeRequest, create_audio_job_endpoint

        with self.assertRaises(HTTPException) as context:
            await create_audio_job_endpoint(
                background_tasks=BackgroundTasks(),
                req=AudioAnalyzeRequest(file_path="C:/missing.wav"),
            )

        self.assertEqual(context.exception.status_code, 400)

    async def test_submit_rejects_missing_runtime_env(self) -> None:
        from services.backend.routers.audio import AudioAnalyzeRequest, create_audio_job_endpoint

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.wav"
            input_path.write_text("wav", encoding="utf-8")

            with patch.dict("os.environ", {}, clear=False):
                with self.assertRaises(HTTPException) as context:
                    await create_audio_job_endpoint(
                        background_tasks=BackgroundTasks(),
                        req=AudioAnalyzeRequest(file_path=str(input_path)),
                    )

        self.assertIn(context.exception.status_code, {500, 503})

    async def test_result_endpoint_returns_409_before_success(self) -> None:
        from services.backend.routers.audio import get_audio_result
        from services.backend.tasks import create_audio_job

        create_audio_job("job-1", "C:/input.wav", "storage/jobs/job-1/audio")

        with self.assertRaises(HTTPException) as context:
            await get_audio_result("job-1")

        self.assertEqual(context.exception.status_code, 409)

    async def test_result_endpoint_reads_result_file_when_memory_result_missing(self) -> None:
        from services.backend.routers.audio import get_audio_result
        from services.backend.tasks import create_audio_job, update_audio_job

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            result_path = temp_path / "audio_stage1_result.json"
            payload = {"request_id": "job-file", "evidence_level": "sufficient"}
            result_path.write_text(json.dumps(payload), encoding="utf-8")

            create_audio_job("job-file", "C:/input.wav", str(temp_path))
            update_audio_job("job-file", status="SUCCEEDED", result=None, result_path=str(result_path))

            result = await get_audio_result("job-file")

        self.assertEqual(result, payload)

    async def test_result_endpoint_returns_404_for_unknown_job(self) -> None:
        from services.backend.routers.audio import get_audio_result

        with self.assertRaises(HTTPException) as context:
            await get_audio_result("missing")

        self.assertEqual(context.exception.status_code, 404)

    async def test_submit_uses_input_filename_stem_as_task_id(self) -> None:
        from services.backend.routers.audio import AudioAnalyzeRequest, create_audio_job_endpoint
        from services.backend.tasks import audio_jobs_db

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "6a26db6c-0951-44c7-8208-e2f3ebf4cc8e_audio.wav"
            input_path.write_text("wav", encoding="utf-8")

            with patch(
                "services.backend.routers.audio.validate_audio_python",
                return_value=input_path,
            ):
                response = await create_audio_job_endpoint(
                    background_tasks=BackgroundTasks(),
                    req=AudioAnalyzeRequest(file_path=str(input_path)),
                )

        expected_task_id = "6a26db6c-0951-44c7-8208-e2f3ebf4cc8e_audio"
        self.assertEqual(response["task_id"], expected_task_id)
        self.assertIn(expected_task_id, audio_jobs_db)
        self.assertEqual(audio_jobs_db[expected_task_id]["artifacts_dir"], f"storage/jobs/{expected_task_id}/audio")

    def test_post_jobs_route_is_not_registered_on_main_app(self) -> None:
        fake_static_ffmpeg = types.SimpleNamespace(add_paths=lambda: None)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.wav"
            input_path.write_text("wav", encoding="utf-8")

            with patch.dict(sys.modules, {"static_ffmpeg": fake_static_ffmpeg}), patch.dict(
                "os.environ", {"VERIFAKE_AI_PYTHON": str(input_path)}, clear=False
            ), patch(
                "services.backend.routers.audio.validate_audio_python",
                return_value=input_path,
            ):
                backend_main = importlib.import_module("services.backend.main")
                client = TestClient(backend_main.app)
                response = client.post("/api/v1/audio/jobs", json={"file_path": str(input_path)})

        self.assertEqual(response.status_code, 404)

    async def test_main_does_not_register_audio_ai_routes(self) -> None:
        fake_static_ffmpeg = types.SimpleNamespace(add_paths=lambda: None)

        with patch.dict(sys.modules, {"static_ffmpeg": fake_static_ffmpeg}):
            backend_main = importlib.import_module("services.backend.main")

        paths = sorted(route.path for route in backend_main.app.routes)
        self.assertIn("/api/v1/status/{task_id}", paths)
        self.assertNotIn("/api/v1/audio/jobs", paths)
        self.assertNotIn("/api/v1/audio/jobs/{task_id}", paths)
        self.assertNotIn("/api/v1/audio/jobs/{task_id}/result", paths)


if __name__ == "__main__":
    unittest.main()
