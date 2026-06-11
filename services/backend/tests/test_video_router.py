from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import BackgroundTasks, HTTPException


class VideoRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        from services.backend.tasks import video_detect_jobs_db

        video_detect_jobs_db.clear()

    def _make_preprocessing_json(self, job_id: str) -> Path:
        path = Path("storage/jobs") / job_id / "metadata" / "preprocessing.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"job_id": job_id, "frames": []}), encoding="utf-8")
        return path

    def test_create_video_detect_job_endpoint_queues_background_job(self) -> None:
        from services.backend.routers.video import VideoStage1DetectRequest, create_video_detect_job_endpoint
        from services.backend.tasks import video_detect_jobs_db

        preprocessing_json = self._make_preprocessing_json("video-route-job-1")
        background_tasks = BackgroundTasks()

        with patch("services.backend.routers.video.validate_video_ai_python", return_value=Path(sys.executable)):
            response = create_video_detect_job_endpoint(
                background_tasks=background_tasks,
                req=VideoStage1DetectRequest(preprocessing_json=str(preprocessing_json)),
            )

        self.assertEqual(response["task_id"], "video-route-job-1")
        self.assertEqual(response["status"], "PENDING")
        self.assertIn("video-route-job-1", video_detect_jobs_db)
        self.assertEqual(video_detect_jobs_db["video-route-job-1"]["artifacts_dir"], "storage/jobs/video-route-job-1/output")
        self.assertEqual(len(background_tasks.tasks), 1)

    def test_create_video_detect_job_rejects_missing_preprocessing_json(self) -> None:
        from services.backend.routers.video import VideoStage1DetectRequest, create_video_detect_job_endpoint

        with self.assertRaises(HTTPException) as context:
            create_video_detect_job_endpoint(
                background_tasks=BackgroundTasks(),
                req=VideoStage1DetectRequest(preprocessing_json="storage/jobs/missing/metadata/preprocessing.json"),
            )

        self.assertEqual(context.exception.status_code, 400)

    def test_video_detect_status_returns_404_for_unknown_job(self) -> None:
        from services.backend.routers.video import get_video_detect_job_status

        with self.assertRaises(HTTPException) as context:
            get_video_detect_job_status("missing")

        self.assertEqual(context.exception.status_code, 404)

    def test_video_detect_result_returns_409_before_success(self) -> None:
        from services.backend.routers.video import get_video_detect_result
        from services.backend.tasks import create_video_detect_job

        create_video_detect_job("job-pending", "storage/jobs/job-pending/metadata/preprocessing.json", "storage/jobs/job-pending/output")

        with self.assertRaises(HTTPException) as context:
            get_video_detect_result("job-pending")

        self.assertEqual(context.exception.status_code, 409)

    def test_video_detect_result_reads_result_file_when_memory_result_missing(self) -> None:
        from services.backend.routers.video import get_video_detect_result
        from services.backend.tasks import create_video_detect_job, update_video_detect_job

        job_id = "job-file-result"
        result_path = Path("storage/jobs") / job_id / "output" / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"job_id": job_id, "status": "success"}
        result_path.write_text(json.dumps(payload), encoding="utf-8")

        create_video_detect_job(job_id, f"storage/jobs/{job_id}/metadata/preprocessing.json", f"storage/jobs/{job_id}/output")
        update_video_detect_job(job_id, status="SUCCEEDED", result=None, result_path=str(result_path))

        self.assertEqual(get_video_detect_result(job_id), payload)

    def test_main_does_not_register_video_ai_routes(self) -> None:
        fake_static_ffmpeg = types.SimpleNamespace(add_paths=lambda: None)

        with patch.dict(sys.modules, {"static_ffmpeg": fake_static_ffmpeg}):
            backend_main = importlib.import_module("services.backend.main")

        paths = sorted(route.path for route in backend_main.app.routes)
        self.assertNotIn("/api/v1/video-stage1/preprocess", paths)
        self.assertNotIn("/api/v1/video-stage1/detect", paths)
        self.assertNotIn("/api/v1/video-stage1/detect/jobs", paths)
        self.assertNotIn("/api/v1/video-stage1/detect/jobs/{task_id}", paths)
        self.assertNotIn("/api/v1/video-stage1/detect/jobs/{task_id}/result", paths)


if __name__ == "__main__":
    unittest.main()
