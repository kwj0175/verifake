from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ProcessorTests(unittest.TestCase):
    def test_separate_streams_runs_ffmpeg_with_timeout_and_capture(self) -> None:
        from services.backend.services.processor import AUDIO_DIR, VIDEO_DIR, MEDIA_SPLIT_TIMEOUT_SEC, separate_streams

        calls: list[tuple[list[str], dict[str, object]]] = []

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

        with patch("services.backend.services.processor.subprocess.run", side_effect=fake_run):
            video_path, audio_path = separate_streams(Path("input.mp4"), "job-1")

        self.assertEqual(video_path, str(VIDEO_DIR / "job-1_video.mp4"))
        self.assertEqual(audio_path, str(AUDIO_DIR / "job-1_audio.wav"))
        self.assertEqual(len(calls), 2)
        for _, kwargs in calls:
            self.assertEqual(kwargs["check"], True)
            self.assertEqual(kwargs["capture_output"], True)
            self.assertEqual(kwargs["text"], True)
            self.assertEqual(kwargs["timeout"], MEDIA_SPLIT_TIMEOUT_SEC)

    def test_processor_exposes_only_media_separation_not_ai_preprocessing(self) -> None:
        import services.backend.services.processor as processor

        self.assertFalse(hasattr(processor, "run_video_stage1_preprocess_job"))

    def test_save_and_split_sanitizes_uploaded_filename_to_task_directory(self) -> None:
        from services.backend.services import processor

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_dir = Path(temp_dir) / "tmp"
            captured_inputs: list[Path] = []

            def fake_separate_streams(input_file: Path, job_id: str) -> tuple[str, str]:
                captured_inputs.append(input_file)
                return f"storage/video/{job_id}_video.mp4", f"storage/audio/{job_id}_audio.wav"

            with patch.object(processor, "TMP_DIR", tmp_dir), patch.object(
                processor,
                "separate_streams",
                side_effect=fake_separate_streams,
            ):
                download_dir, _, _ = processor.save_and_split("job-1", "../evil.mp4", b"video")

        self.assertEqual(download_dir, str(tmp_dir / "job-1"))
        self.assertEqual(captured_inputs, [tmp_dir / "job-1" / "evil.mp4"])


if __name__ == "__main__":
    unittest.main()
