from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from services.ai.audio_pipeline.audio_windowing import AudioWindowingError, run_audio_windowing


def _write_mono_wav(path: Path, *, sample_rate: int = 16000, duration_sec: float = 9.6, amplitude: int = 4000) -> None:
    frame_count = int(sample_rate * duration_sec)
    timeline = np.linspace(0.0, duration_sec, frame_count, endpoint=False)
    waveform = (np.sin(2.0 * np.pi * 220.0 * timeline) * amplitude).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(waveform.tobytes())


def _write_bad_wav(path: Path, *, sample_rate: int = 22050, channels: int = 2, duration_sec: float = 1.0) -> None:
    frame_count = int(sample_rate * duration_sec)
    timeline = np.linspace(0.0, duration_sec, frame_count, endpoint=False)
    waveform = (np.sin(2.0 * np.pi * 330.0 * timeline) * 4000).astype(np.int16)
    if channels > 1:
        waveform = np.repeat(waveform[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(waveform.tobytes())


def _write_vad_json(path: Path, *, wav_path: Path, total_duration_sec: float, speech_segments: list[dict[str, float]], quality_flags: list[str] | None = None, limits: dict[str, str | None] | None = None) -> None:
    payload = {
        "audio_vad": {
            "input_wav_path": str(wav_path),
            "total_duration_sec": total_duration_sec,
            "speech_segments": speech_segments,
            "human_speech_detected": bool(speech_segments),
            "quality_flags": quality_flags or [],
        },
        "limits": limits or {"unsupported_reason": None, "low_evidence_reason": None},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class AudioWindowingTests(unittest.TestCase):
    def test_vad_json_creates_overlapping_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            vad_json = temp_path / "audio_vad_result.json"
            _write_mono_wav(wav_path, duration_sec=9.6)
            _write_vad_json(
                vad_json,
                wav_path=wav_path,
                total_duration_sec=9.6,
                speech_segments=[{"start": 0.3, "end": 0.7, "duration": 0.4}],
            )

            result = run_audio_windowing(vad_json_path=vad_json, window_sec=4.0, hop_sec=2.0)

            windows = result["audio_windows"]["windows"]
            self.assertEqual(len(windows), 4)
            self.assertEqual((windows[0]["start"], windows[0]["end"]), (0.0, 4.0))
            self.assertEqual((windows[1]["start"], windows[1]["end"]), (2.0, 6.0))
            self.assertEqual((windows[2]["start"], windows[2]["end"]), (4.0, 8.0))
            self.assertEqual((windows[3]["start"], windows[3]["end"]), (6.0, 9.6))

    def test_window_bounds_do_not_exceed_total_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            vad_json = temp_path / "audio_vad_result.json"
            _write_mono_wav(wav_path, duration_sec=9.6)
            _write_vad_json(vad_json, wav_path=wav_path, total_duration_sec=9.6, speech_segments=[])

            result = run_audio_windowing(vad_json_path=vad_json, window_sec=4.0, hop_sec=2.0)

            for window in result["audio_windows"]["windows"]:
                self.assertGreaterEqual(window["start"], 0.0)
                self.assertLessEqual(window["end"], 9.6)
                self.assertLessEqual(window["duration"], 4.0)

    def test_speech_overlap_is_computed_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            vad_json = temp_path / "audio_vad_result.json"
            _write_mono_wav(wav_path, duration_sec=6.0)
            _write_vad_json(
                vad_json,
                wav_path=wav_path,
                total_duration_sec=6.0,
                speech_segments=[
                    {"start": 0.5, "end": 1.5, "duration": 1.0},
                    {"start": 3.0, "end": 4.5, "duration": 1.5},
                ],
            )

            result = run_audio_windowing(vad_json_path=vad_json, window_sec=4.0, hop_sec=2.0)
            windows = result["audio_windows"]["windows"]

            self.assertEqual(windows[0]["speech_overlap_sec"], 2.0)
            self.assertEqual(windows[0]["speech_coverage_ratio"], 0.5)
            self.assertTrue(windows[0]["has_speech"])
            self.assertEqual(windows[1]["speech_overlap_sec"], 1.5)
            self.assertEqual(windows[1]["speech_coverage_ratio"], 0.375)

    def test_no_speech_vad_still_creates_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            vad_json = temp_path / "audio_vad_result.json"
            _write_mono_wav(wav_path, duration_sec=5.0)
            _write_vad_json(
                vad_json,
                wav_path=wav_path,
                total_duration_sec=5.0,
                speech_segments=[],
                quality_flags=["NO_DETECTED_SPEECH"],
                limits={"unsupported_reason": "no_detected_human_speech", "low_evidence_reason": None},
            )

            result = run_audio_windowing(vad_json_path=vad_json)

            self.assertGreater(result["audio_windows"]["window_count"], 0)
            self.assertIn("NO_DETECTED_SPEECH", result["audio_windows"]["quality_flags"])
            self.assertEqual(result["limits"]["unsupported_reason"], "no_detected_human_speech")
            self.assertTrue(all(window["speech_overlap_sec"] == 0.0 for window in result["audio_windows"]["windows"]))

    def test_input_wav_mode_uses_null_overlap_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            _write_mono_wav(wav_path, duration_sec=5.0)

            result = run_audio_windowing(input_wav_path=wav_path)

            self.assertGreater(result["audio_windows"]["window_count"], 0)
            self.assertTrue(all(window["speech_overlap_sec"] is None for window in result["audio_windows"]["windows"]))
            self.assertTrue(all(window["speech_coverage_ratio"] is None for window in result["audio_windows"]["windows"]))
            self.assertTrue(all(window["has_speech"] is False for window in result["audio_windows"]["windows"]))

    def test_non_normalized_wav_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "bad.wav"
            _write_bad_wav(wav_path)

            with self.assertRaises(AudioWindowingError):
                run_audio_windowing(input_wav_path=wav_path)

    def test_json_output_and_no_window_wav_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            vad_json = temp_path / "audio_vad_result.json"
            json_output = temp_path / "outputs" / "audio_windows_result.json"
            _write_mono_wav(wav_path, duration_sec=4.5)
            _write_vad_json(vad_json, wav_path=wav_path, total_duration_sec=4.5, speech_segments=[])
            input_size = wav_path.stat().st_size

            result = run_audio_windowing(vad_json_path=vad_json, json_output_path=json_output)

            self.assertTrue(json_output.exists())
            loaded = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(loaded, result)
            self.assertEqual(wav_path.stat().st_size, input_size)
            all_files = sorted(path.relative_to(temp_path).as_posix() for path in temp_path.rglob("*") if path.is_file())
            self.assertEqual(all_files, ["audio_vad_result.json", "outputs/audio_windows_result.json", "sample_16k_mono.wav"])
            self.assertTrue(all("fake_score" not in window for window in result["audio_windows"]["windows"]))
            self.assertTrue(all("real_score" not in window for window in result["audio_windows"]["windows"]))
            self.assertTrue(all("audio_fake_prob_like" not in window for window in result["audio_windows"]["windows"]))


if __name__ == "__main__":
    unittest.main()
