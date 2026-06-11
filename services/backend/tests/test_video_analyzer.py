from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class VideoAnalyzerTests(unittest.TestCase):
    def test_build_video_detect_command_uses_detect_module(self) -> None:
        from services.backend.services.video_analyzer import build_video_detect_command

        python_path = Path("/venv/bin/python")
        preprocessing_json = Path("storage/jobs/job-1/metadata/preprocessing.json")

        command = build_video_detect_command(
            python_executable=python_path,
            preprocessing_json=preprocessing_json,
        )

        self.assertEqual(command[0], str(python_path))
        self.assertIn("services.ai.pipelines.video_stage1.detect", command)
        self.assertIn("--preprocessing-json", command)
        self.assertIn(str(preprocessing_json), command)

    def test_get_video_ai_python_uses_env_override(self) -> None:
        from services.backend.services.video_analyzer import get_video_ai_python

        with tempfile.TemporaryDirectory() as temp_dir:
            python_path = Path(temp_dir) / "python"
            python_path.write_text("", encoding="utf-8")

            with patch.dict("os.environ", {"VERIFAKE_VIDEO_AI_PYTHON": str(python_path)}, clear=False), patch(
                "services.backend.services.video_analyzer.os.access",
                return_value=True,
            ):
                self.assertEqual(get_video_ai_python(), python_path.absolute())

    def test_run_video_detect_job_succeeds_and_stores_result(self) -> None:
        from services.backend.services.video_analyzer import run_video_detect_job
        from services.backend.tasks import create_video_detect_job, video_detect_jobs_db

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_python = temp_path / "python"
            fake_python.write_text("", encoding="utf-8")
            preprocessing_json = temp_path / "preprocessing.json"
            preprocessing_json.write_text("{}", encoding="utf-8")
            job_id = "video-job-1"
            result_path = Path("storage/jobs") / job_id / "output" / "result.json"
            video_detect_jobs_db.clear()
            create_video_detect_job(job_id, str(preprocessing_json), f"storage/jobs/{job_id}/output")

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps({"job_id": job_id, "status": "success"}), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with patch.dict("os.environ", {"VERIFAKE_VIDEO_AI_PYTHON": str(fake_python)}, clear=False), patch(
                "services.backend.services.video_analyzer.os.access",
                return_value=True,
            ), patch(
                "services.backend.services.video_analyzer.subprocess.run",
                side_effect=fake_run,
            ):
                run_video_detect_job(job_id, preprocessing_json)

            self.assertEqual(video_detect_jobs_db[job_id]["status"], "SUCCEEDED")
            self.assertEqual(video_detect_jobs_db[job_id]["result"]["job_id"], job_id)
            self.assertEqual(video_detect_jobs_db[job_id]["returncode"], 0)
            self.assertEqual(video_detect_jobs_db[job_id]["stage"], "video_stage1_detect")

    def test_run_video_detect_job_stores_failure_state(self) -> None:
        from services.backend.services.video_analyzer import run_video_detect_job
        from services.backend.tasks import create_video_detect_job, video_detect_jobs_db

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_python = temp_path / "python"
            fake_python.write_text("", encoding="utf-8")
            preprocessing_json = temp_path / "preprocessing.json"
            preprocessing_json.write_text("{}", encoding="utf-8")
            job_id = "video-job-2"
            video_detect_jobs_db.clear()
            create_video_detect_job(job_id, str(preprocessing_json), f"storage/jobs/{job_id}/output")

            with patch.dict("os.environ", {"VERIFAKE_VIDEO_AI_PYTHON": str(fake_python)}, clear=False), patch(
                "services.backend.services.video_analyzer.os.access",
                return_value=True,
            ), patch(
                "services.backend.services.video_analyzer.subprocess.run",
                return_value=subprocess.CompletedProcess(["cmd"], 1, stdout="bad", stderr="trace"),
            ):
                run_video_detect_job(job_id, preprocessing_json)

            self.assertEqual(video_detect_jobs_db[job_id]["status"], "FAILED")
            self.assertEqual(video_detect_jobs_db[job_id]["stdout"], "bad")
            self.assertEqual(video_detect_jobs_db[job_id]["stderr"], "trace")
            self.assertEqual(video_detect_jobs_db[job_id]["returncode"], 1)

    def test_run_video_detect_job_marks_timeout(self) -> None:
        from services.backend.services.video_analyzer import run_video_detect_job
        from services.backend.tasks import create_video_detect_job, video_detect_jobs_db

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_python = temp_path / "python"
            fake_python.write_text("", encoding="utf-8")
            preprocessing_json = temp_path / "preprocessing.json"
            preprocessing_json.write_text("{}", encoding="utf-8")
            job_id = "video-job-3"
            video_detect_jobs_db.clear()
            create_video_detect_job(job_id, str(preprocessing_json), f"storage/jobs/{job_id}/output")
            timeout_exc = subprocess.TimeoutExpired(cmd=["cmd"], timeout=5)
            timeout_exc.stdout = b"partial-out"
            timeout_exc.stderr = b"partial-err"

            with patch.dict("os.environ", {"VERIFAKE_VIDEO_AI_PYTHON": str(fake_python)}, clear=False), patch(
                "services.backend.services.video_analyzer.os.access",
                return_value=True,
            ), patch(
                "services.backend.services.video_analyzer.subprocess.run",
                side_effect=timeout_exc,
            ):
                run_video_detect_job(job_id, preprocessing_json)

            self.assertEqual(video_detect_jobs_db[job_id]["status"], "TIMED_OUT")
            self.assertIn("timeout", video_detect_jobs_db[job_id]["error"].lower())


if __name__ == "__main__":
    unittest.main()
