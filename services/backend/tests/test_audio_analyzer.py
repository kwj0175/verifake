from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class AudioAnalyzerTests(unittest.TestCase):
    def test_get_audio_device_defaults_to_cpu(self) -> None:
        from services.backend.services.audio_analyzer import get_audio_device

        with patch.dict("os.environ", {}, clear=False):
            with patch(
                "services.backend.services.audio_analyzer.resolve_torch_device",
                return_value="cpu",
            ):
                self.assertEqual(get_audio_device(), "cpu")

    def test_get_audio_device_uses_env_override(self) -> None:
        from services.backend.services.audio_analyzer import get_audio_device

        with patch.dict("os.environ", {"VERIFAKE_AI_DEVICE": "cuda:0"}, clear=False):
            self.assertEqual(get_audio_device(), "cuda:0")

    def test_missing_runtime_env_raises(self) -> None:
        from services.backend.services.audio_analyzer import get_audio_python

        with patch.dict("os.environ", {}, clear=False):
            with self.assertRaises(RuntimeError):
                get_audio_python()

    def test_invalid_runtime_path_raises(self) -> None:
        from services.backend.services.audio_analyzer import get_audio_python

        with patch.dict("os.environ", {"VERIFAKE_AI_PYTHON": "C:/missing/python.exe"}, clear=False):
            with self.assertRaises(RuntimeError):
                get_audio_python()

    def test_non_executable_runtime_path_raises(self) -> None:
        from services.backend.services.audio_analyzer import get_audio_python

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "python.exe"
            temp_path.write_text("", encoding="utf-8")
            with patch.dict("os.environ", {"VERIFAKE_AI_PYTHON": str(temp_path)}, clear=False), patch(
                "services.backend.services.audio_analyzer.os.access",
                return_value=False,
            ):
                with self.assertRaises(RuntimeError):
                    get_audio_python()

    def test_get_audio_python_preserves_virtualenv_symlink_path(self) -> None:
        from services.backend.services.audio_analyzer import get_audio_python

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            real_python = temp_root / "python-real"
            real_python.write_text("", encoding="utf-8")
            symlink_python = temp_root / "python"
            symlink_python.symlink_to(real_python)

            with patch.dict("os.environ", {"VERIFAKE_AI_PYTHON": str(symlink_python)}, clear=False), patch(
                "services.backend.services.audio_analyzer.os.access",
                return_value=True,
            ):
                self.assertEqual(get_audio_python(), symlink_python.absolute())

    def test_build_command_uses_audio_stage1_and_expected_paths(self) -> None:
        from services.backend.services.audio_analyzer import build_audio_stage1_command

        input_path = Path("storage/audio/job_audio.wav")
        output_dir = Path("storage/jobs/job-1/audio")
        python_path = Path("C:/venvs/ai/python.exe")

        command = build_audio_stage1_command(
            python_executable=python_path,
            input_path=input_path,
            output_dir=output_dir,
            job_id="job-1",
            device="cpu",
        )

        self.assertEqual(command[0], str(python_path))
        self.assertIn("services.ai.audio_pipeline.audio_stage1", command)
        self.assertIn("--input", command)
        self.assertIn(str(input_path), command)
        self.assertIn("--output-dir", command)
        self.assertIn(str(output_dir), command)
        self.assertIn("--request-id", command)
        self.assertIn("job-1", command)
        self.assertIn("--device", command)
        self.assertIn("cpu", command)

    def test_run_audio_job_succeeds_and_stores_result(self) -> None:
        from services.backend.services.audio_analyzer import run_audio_job
        from services.backend.tasks import audio_jobs_db, create_audio_job

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_python = temp_path / "python.exe"
            fake_python.write_text("", encoding="utf-8")
            input_path = temp_path / "input.wav"
            input_path.write_text("wav", encoding="utf-8")
            job_id = "job-1"
            output_dir = Path("storage/jobs") / job_id / "audio"
            result_path = output_dir / "audio_stage1_result.json"
            audio_jobs_db.clear()
            create_audio_job(job_id, str(input_path), str(output_dir))

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps({"request_id": job_id, "evidence_level": "sufficient"}), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with patch.dict("os.environ", {"VERIFAKE_AI_PYTHON": str(fake_python)}, clear=False), patch(
                "services.backend.services.audio_analyzer.os.access",
                return_value=True,
            ), patch(
                "services.backend.services.audio_analyzer.subprocess.run",
                side_effect=fake_run,
            ), patch(
                "services.backend.services.audio_analyzer.resolve_torch_device",
                return_value="cpu",
            ):
                run_audio_job(job_id, input_path)

            self.assertEqual(audio_jobs_db[job_id]["status"], "SUCCEEDED")
            self.assertEqual(audio_jobs_db[job_id]["result"]["request_id"], job_id)
            self.assertEqual(audio_jobs_db[job_id]["returncode"], 0)
            self.assertEqual(audio_jobs_db[job_id]["stage"], "audio_stage1")
            self.assertIsNone(audio_jobs_db[job_id]["video_path"])

    def test_run_audio_job_passes_default_cpu_device(self) -> None:
        from services.backend.services.audio_analyzer import run_audio_job
        from services.backend.tasks import audio_jobs_db, create_audio_job

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_python = temp_path / "python.exe"
            fake_python.write_text("", encoding="utf-8")
            input_path = temp_path / "input.wav"
            input_path.write_text("wav", encoding="utf-8")
            job_id = "job-device-default"
            result_path = Path("storage/jobs") / job_id / "audio" / "audio_stage1_result.json"
            audio_jobs_db.clear()
            create_audio_job(job_id, str(input_path), f"storage/jobs/{job_id}/audio")
            captured_commands: list[list[str]] = []

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                captured_commands.append(command)
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps({"request_id": job_id, "evidence_level": "sufficient"}), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with patch.dict("os.environ", {"VERIFAKE_AI_PYTHON": str(fake_python)}, clear=False), patch(
                "services.backend.services.audio_analyzer.os.access",
                return_value=True,
            ), patch(
                "services.backend.services.audio_analyzer.subprocess.run",
                side_effect=fake_run,
            ), patch(
                "services.backend.services.audio_analyzer._resolve_device_from_runtime",
                return_value="cpu",
            ):
                run_audio_job(job_id, input_path)

            self.assertEqual(len(captured_commands), 1)
            self.assertIn("--device", captured_commands[0])
            self.assertIn("cpu", captured_commands[0])

    def test_run_audio_job_passes_env_device_override(self) -> None:
        from services.backend.services.audio_analyzer import run_audio_job
        from services.backend.tasks import audio_jobs_db, create_audio_job

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_python = temp_path / "python.exe"
            fake_python.write_text("", encoding="utf-8")
            input_path = temp_path / "input.wav"
            input_path.write_text("wav", encoding="utf-8")
            job_id = "job-device-override"
            result_path = Path("storage/jobs") / job_id / "audio" / "audio_stage1_result.json"
            audio_jobs_db.clear()
            create_audio_job(job_id, str(input_path), f"storage/jobs/{job_id}/audio")
            captured_commands: list[list[str]] = []

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                captured_commands.append(command)
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps({"request_id": job_id, "evidence_level": "sufficient"}), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with patch.dict("os.environ", {"VERIFAKE_AI_PYTHON": str(fake_python), "VERIFAKE_AI_DEVICE": "cuda:0"}, clear=False), patch(
                "services.backend.services.audio_analyzer.os.access",
                return_value=True,
            ), patch(
                "services.backend.services.audio_analyzer.subprocess.run",
                side_effect=fake_run,
            ), patch(
                "services.backend.services.audio_analyzer._resolve_device_from_runtime",
                return_value="cuda:0",
            ):
                run_audio_job(job_id, input_path)

            self.assertEqual(len(captured_commands), 1)
            self.assertIn("--device", captured_commands[0])
            self.assertIn("cuda:0", captured_commands[0])

    def test_run_audio_job_passes_original_input_to_stage1_without_split(self) -> None:
        from services.backend.services.audio_analyzer import run_audio_job
        from services.backend.tasks import audio_jobs_db, create_audio_job

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_python = temp_path / "python.exe"
            fake_python.write_text("", encoding="utf-8")
            input_path = temp_path / "input.mp3"
            input_path.write_text("audio", encoding="utf-8")
            job_id = "job-original"
            result_path = Path("storage/jobs") / job_id / "audio" / "audio_stage1_result.json"
            audio_jobs_db.clear()
            create_audio_job(job_id, str(input_path), f"storage/jobs/{job_id}/audio")
            captured_commands: list[list[str]] = []

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                captured_commands.append(command)
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps({"request_id": job_id, "evidence_level": "sufficient"}), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with patch.dict("os.environ", {"VERIFAKE_AI_PYTHON": str(fake_python)}, clear=False), patch(
                "services.backend.services.audio_analyzer.os.access",
                return_value=True,
            ), patch(
                "services.backend.services.audio_analyzer.subprocess.run",
                side_effect=fake_run,
            ):
                run_audio_job(job_id, input_path)

            self.assertEqual(len(captured_commands), 1)
            self.assertIn("--input", captured_commands[0])
            input_arg_index = captured_commands[0].index("--input")
            actual_input_path = Path(captured_commands[0][input_arg_index + 1])
            self.assertEqual(actual_input_path.resolve(), input_path.resolve())

    def test_run_audio_job_stores_failure_state(self) -> None:
        from services.backend.services.audio_analyzer import run_audio_job
        from services.backend.tasks import audio_jobs_db, create_audio_job

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_python = temp_path / "python.exe"
            fake_python.write_text("", encoding="utf-8")
            input_path = temp_path / "input.wav"
            input_path.write_text("wav", encoding="utf-8")
            audio_jobs_db.clear()
            create_audio_job("job-2", str(input_path), "storage/jobs/job-2/audio")

            with patch.dict("os.environ", {"VERIFAKE_AI_PYTHON": str(fake_python)}, clear=False), patch(
                "services.backend.services.audio_analyzer.os.access",
                return_value=True,
            ), patch(
                "services.backend.services.audio_analyzer.subprocess.run",
                return_value=subprocess.CompletedProcess(["cmd"], 1, stdout="bad", stderr="trace"),
            ):
                run_audio_job("job-2", input_path)

            self.assertEqual(audio_jobs_db["job-2"]["status"], "FAILED")
            self.assertEqual(audio_jobs_db["job-2"]["stdout"], "bad")
            self.assertEqual(audio_jobs_db["job-2"]["stderr"], "trace")
            self.assertEqual(audio_jobs_db["job-2"]["returncode"], 1)
            self.assertEqual(audio_jobs_db["job-2"]["stage"], "audio_stage1")

    def test_run_audio_job_marks_timeout(self) -> None:
        from services.backend.services.audio_analyzer import run_audio_job
        from services.backend.tasks import audio_jobs_db, create_audio_job

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_python = temp_path / "python.exe"
            fake_python.write_text("", encoding="utf-8")
            input_path = temp_path / "input.wav"
            input_path.write_text("wav", encoding="utf-8")
            audio_jobs_db.clear()
            create_audio_job("job-3", str(input_path), "storage/jobs/job-3/audio")

            timeout_exc = subprocess.TimeoutExpired(cmd=["cmd"], timeout=5)
            timeout_exc.stdout = b"partial-out"
            timeout_exc.stderr = b"partial-err"

            with patch.dict("os.environ", {"VERIFAKE_AI_PYTHON": str(fake_python)}, clear=False), patch(
                "services.backend.services.audio_analyzer.os.access",
                return_value=True,
            ), patch(
                "services.backend.services.audio_analyzer.subprocess.run",
                side_effect=timeout_exc,
            ):
                run_audio_job("job-3", input_path)

            self.assertEqual(audio_jobs_db["job-3"]["status"], "TIMED_OUT")
            self.assertIn("timeout", audio_jobs_db["job-3"]["error"].lower())
            self.assertEqual(audio_jobs_db["job-3"]["stage"], "audio_stage1")

    def test_run_audio_job_missing_result_file_marks_failed(self) -> None:
        from services.backend.services.audio_analyzer import run_audio_job
        from services.backend.tasks import audio_jobs_db, create_audio_job

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_python = temp_path / "python.exe"
            fake_python.write_text("", encoding="utf-8")
            input_path = temp_path / "input.wav"
            input_path.write_text("wav", encoding="utf-8")
            audio_jobs_db.clear()
            create_audio_job("job-5", str(input_path), "storage/jobs/job-5/audio")

            with patch.dict("os.environ", {"VERIFAKE_AI_PYTHON": str(fake_python)}, clear=False), patch(
                "services.backend.services.audio_analyzer.os.access",
                return_value=True,
            ), patch(
                "services.backend.services.audio_analyzer.subprocess.run",
                return_value=subprocess.CompletedProcess(["cmd"], 0, stdout="ok", stderr=""),
            ):
                run_audio_job("job-5", input_path)

            self.assertEqual(audio_jobs_db["job-5"]["status"], "FAILED")
            self.assertEqual(audio_jobs_db["job-5"]["stage"], "audio_stage1")
            self.assertIn("결과 파일이 생성되지 않았습니다", audio_jobs_db["job-5"]["error"])

    def test_run_audio_job_invalid_result_json_marks_failed(self) -> None:
        from services.backend.services.audio_analyzer import run_audio_job
        from services.backend.tasks import audio_jobs_db, create_audio_job

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_python = temp_path / "python.exe"
            fake_python.write_text("", encoding="utf-8")
            input_path = temp_path / "input.wav"
            input_path.write_text("wav", encoding="utf-8")
            job_id = "job-6"
            result_path = Path("storage/jobs") / job_id / "audio" / "audio_stage1_result.json"
            audio_jobs_db.clear()
            create_audio_job(job_id, str(input_path), f"storage/jobs/{job_id}/audio")

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text("{invalid json", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with patch.dict("os.environ", {"VERIFAKE_AI_PYTHON": str(fake_python)}, clear=False), patch(
                "services.backend.services.audio_analyzer.os.access",
                return_value=True,
            ), patch(
                "services.backend.services.audio_analyzer.subprocess.run",
                side_effect=fake_run,
            ):
                run_audio_job(job_id, input_path)

            self.assertEqual(audio_jobs_db[job_id]["status"], "FAILED")
            self.assertEqual(audio_jobs_db[job_id]["stage"], "audio_stage1")
            self.assertIn("결과 파일을 읽을 수 없습니다", audio_jobs_db[job_id]["error"])


if __name__ == "__main__":
    unittest.main()
