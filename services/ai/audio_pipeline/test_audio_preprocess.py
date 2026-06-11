from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from services.ai.audio_pipeline.audio_preprocess import (
    AudioPreprocessError,
    NoAudioStreamError,
    preprocess_audio,
)


def _write_test_wav(
    path: Path,
    *,
    sample_rate: int = 22050,
    channels: int = 2,
    seconds: float = 1.0,
    amplitude: int = 4000,
) -> None:
    frame_count = int(sample_rate * seconds)
    timeline = np.linspace(0.0, seconds, frame_count, endpoint=False)
    waveform = (np.sin(2.0 * np.pi * 440.0 * timeline) * amplitude).astype(np.int16)
    if channels > 1:
        waveform = np.repeat(waveform[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(waveform.tobytes())


def _write_silent_wav(path: Path, *, sample_rate: int = 16000, seconds: float = 1.0) -> None:
    frame_count = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frame_count)


def _create_video_without_audio(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise unittest.SkipTest("ffmpeg가 없어 no-audio fixture를 만들 수 없습니다.")

    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=16x16:d=0.1",
        "-an",
        str(path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


class AudioPreprocessTests(unittest.TestCase):
    def test_synthetic_wav_creates_normalized_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.wav"
            output_dir = temp_path / "outputs"
            _write_test_wav(input_path)

            result = preprocess_audio(input_path, output_dir)

            normalized_path = Path(result["audio_preprocess"]["normalized_wav_path"])
            self.assertTrue(normalized_path.exists())
            with wave.open(str(normalized_path), "rb") as handle:
                self.assertEqual(handle.getframerate(), 16000)
                self.assertEqual(handle.getnchannels(), 1)

    def test_json_output_is_written_with_expected_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.wav"
            output_dir = temp_path / "outputs"
            json_output = temp_path / "outputs" / "audio_preprocess_result.json"
            _write_test_wav(input_path, sample_rate=16000, channels=1)

            result = preprocess_audio(input_path, output_dir, json_output_path=json_output)

            self.assertTrue(json_output.exists())
            loaded = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(loaded, result)
            self.assertIn("audio_preprocess", loaded)
            self.assertIn("limits", loaded)
            self.assertIn("original", loaded["audio_preprocess"])
            self.assertIn("normalized", loaded["audio_preprocess"])
            self.assertIn("quality_metrics", loaded["audio_preprocess"])
            self.assertIn("quality_flags", loaded["audio_preprocess"])

    def test_missing_input_path_raises_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.wav"
            with self.assertRaises(FileNotFoundError):
                preprocess_audio(missing_path, Path(temp_dir) / "outputs")

    def test_ffprobe_failure_raises_audio_preprocess_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "broken.wav"
            input_path.write_bytes(b"not-a-real-wave")

            with patch(
                "services.ai.audio_pipeline.audio_preprocess.shutil.which",
                return_value="/usr/bin/ffprobe",
            ), patch(
                "services.ai.audio_pipeline.audio_preprocess.subprocess.run",
                side_effect=subprocess.CalledProcessError(
                    1,
                    ["ffprobe", str(input_path)],
                    stderr="invalid data found",
                ),
            ):
                with self.assertRaises(AudioPreprocessError) as context:
                    preprocess_audio(input_path, temp_path / "outputs")

            self.assertIn("ffprobe 실패", str(context.exception))

    def test_video_without_audio_stream_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "video_only.mp4"
            _create_video_without_audio(input_path)

            with self.assertRaises(NoAudioStreamError):
                preprocess_audio(input_path, temp_path / "outputs")

    def test_silent_audio_generates_quality_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "silent.wav"
            _write_silent_wav(input_path)

            result = preprocess_audio(input_path, temp_path / "outputs")

            flags = result["audio_preprocess"]["quality_flags"]
            self.assertIn("SILENT_AUDIO", flags)
            self.assertIn("HIGH_SILENCE_RATIO", flags)


if __name__ == "__main__":
    unittest.main()
