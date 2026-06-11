from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from services.ai.audio_pipeline.audio_vad import AudioVadError, run_audio_vad


def _write_segmented_wav(
    path: Path,
    *,
    sample_rate: int = 16000,
    pattern: list[tuple[str, float]] | None = None,
    amplitude: int = 5000,
) -> None:
    if pattern is None:
        pattern = [("silence", 0.5), ("speech", 1.0), ("silence", 0.6), ("speech", 0.8), ("silence", 0.4)]

    chunks: list[np.ndarray] = []
    for kind, seconds in pattern:
        frame_count = int(sample_rate * seconds)
        if kind == "silence":
            chunk = np.zeros(frame_count, dtype=np.int16)
        else:
            timeline = np.linspace(0.0, seconds, frame_count, endpoint=False)
            chunk = (np.sin(2.0 * np.pi * 220.0 * timeline) * amplitude).astype(np.int16)
        chunks.append(chunk)
    waveform = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(waveform.tobytes())


def _write_stereo_wav(path: Path, *, sample_rate: int = 22050, seconds: float = 1.0) -> None:
    frame_count = int(sample_rate * seconds)
    timeline = np.linspace(0.0, seconds, frame_count, endpoint=False)
    waveform = (np.sin(2.0 * np.pi * 330.0 * timeline) * 4000).astype(np.int16)
    stereo = np.repeat(waveform[:, None], 2, axis=1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(stereo.tobytes())


class AudioVadTests(unittest.TestCase):
    def test_vad_from_preprocess_json_creates_segments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            preprocess_json = temp_path / "audio_preprocess_result.json"
            _write_segmented_wav(wav_path)
            preprocess_json.write_text(
                json.dumps(
                    {
                        "audio_preprocess": {
                            "normalized_wav_path": str(wav_path),
                            "quality_flags": [],
                        },
                        "limits": {
                            "unsupported_reason": None,
                            "low_evidence_reason": None,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch("services.ai.audio_pipeline.audio_vad._silero_vad_segments", side_effect=AudioVadError("silero_vad_not_available")):
                result = run_audio_vad(preprocess_json_path=preprocess_json)

            self.assertGreater(len(result["audio_vad"]["speech_segments"]), 0)
            self.assertEqual(result["audio_vad"]["vad_method"], "energy_fallback")
            self.assertIn("ENERGY_FALLBACK_VAD_USED", result["audio_vad"]["quality_flags"])

    def test_ratios_sum_close_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            _write_segmented_wav(wav_path)

            with patch("services.ai.audio_pipeline.audio_vad._silero_vad_segments", side_effect=AudioVadError("silero_vad_not_available")):
                result = run_audio_vad(input_wav_path=wav_path)

            stats = result["audio_vad"]["speech_stats"]
            self.assertGreaterEqual(stats["speech_ratio"], 0.0)
            self.assertLessEqual(stats["speech_ratio"], 1.0)
            self.assertGreaterEqual(stats["silence_ratio"], 0.0)
            self.assertLessEqual(stats["silence_ratio"], 1.0)
            self.assertAlmostEqual(stats["speech_ratio"] + stats["silence_ratio"], 1.0, places=3)

    def test_leading_trailing_and_pause_count_are_computed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            _write_segmented_wav(wav_path)

            with patch("services.ai.audio_pipeline.audio_vad._silero_vad_segments", side_effect=AudioVadError("silero_vad_not_available")):
                result = run_audio_vad(input_wav_path=wav_path)

            stats = result["audio_vad"]["speech_stats"]
            self.assertGreater(stats["leading_silence"], 0.0)
            self.assertGreater(stats["trailing_silence"], 0.0)
            self.assertGreaterEqual(stats["pause_count"], 1)

    def test_silent_audio_marks_human_speech_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "silent_16k_mono.wav"
            _write_segmented_wav(wav_path, pattern=[("silence", 2.0)])

            with patch("services.ai.audio_pipeline.audio_vad._silero_vad_segments", side_effect=AudioVadError("silero_vad_not_available")):
                result = run_audio_vad(input_wav_path=wav_path)

            self.assertFalse(result["audio_vad"]["human_speech_detected"])
            self.assertEqual(result["limits"]["unsupported_reason"], "no_detected_human_speech")
            self.assertIn("NO_DETECTED_SPEECH", result["audio_vad"]["quality_flags"])

    def test_short_speech_sets_low_evidence_but_keeps_detection_true(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "short_speech_16k_mono.wav"
            _write_segmented_wav(wav_path, pattern=[("silence", 0.3), ("speech", 0.3), ("silence", 1.0)])

            with patch("services.ai.audio_pipeline.audio_vad._silero_vad_segments", side_effect=AudioVadError("silero_vad_not_available")):
                result = run_audio_vad(input_wav_path=wav_path)

            self.assertTrue(result["audio_vad"]["human_speech_detected"])
            self.assertEqual(result["limits"]["low_evidence_reason"], "too_little_detected_speech")
            self.assertGreater(len(result["audio_vad"]["speech_segments"]), 0)

    def test_breathing_like_pattern_uses_gap_heuristic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "breathy_16k_mono.wav"
            _write_segmented_wav(
                wav_path,
                pattern=[("speech", 1.0), ("silence", 0.6), ("speech", 0.9)],
            )

            with patch("services.ai.audio_pipeline.audio_vad._silero_vad_segments", side_effect=AudioVadError("silero_vad_not_available")):
                result = run_audio_vad(input_wav_path=wav_path)

            pattern = result["audio_vad"]["speech_stats"]["breathing_like_pattern"]
            self.assertIsNotNone(pattern)
            self.assertTrue(pattern["detected"])
            self.assertEqual(pattern["candidate_gap_count"], 1)

    def test_breathing_like_pattern_is_none_for_continuous_speech(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "continuous_16k_mono.wav"
            _write_segmented_wav(wav_path, pattern=[("speech", 2.0)])

            with patch("services.ai.audio_pipeline.audio_vad._silero_vad_segments", side_effect=AudioVadError("silero_vad_not_available")):
                result = run_audio_vad(input_wav_path=wav_path)

            self.assertIsNone(result["audio_vad"]["speech_stats"]["breathing_like_pattern"])

    def test_breathing_like_pattern_ignores_short_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "short_gap_16k_mono.wav"
            _write_segmented_wav(
                wav_path,
                pattern=[("speech", 1.0), ("silence", 0.1), ("speech", 1.0)],
            )

            with patch("services.ai.audio_pipeline.audio_vad._silero_vad_segments", side_effect=AudioVadError("silero_vad_not_available")):
                result = run_audio_vad(input_wav_path=wav_path)

            self.assertIsNone(result["audio_vad"]["speech_stats"]["breathing_like_pattern"])

    def test_non_normalized_wav_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "bad.wav"
            _write_stereo_wav(wav_path)

            with self.assertRaises(AudioVadError):
                run_audio_vad(input_wav_path=wav_path)

    def test_json_output_and_no_extra_segment_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            json_output = temp_path / "outputs" / "audio_vad_result.json"
            _write_segmented_wav(wav_path)
            input_size = wav_path.stat().st_size

            with patch("services.ai.audio_pipeline.audio_vad._silero_vad_segments", side_effect=AudioVadError("silero_vad_not_available")):
                result = run_audio_vad(input_wav_path=wav_path, json_output_path=json_output)

            self.assertTrue(json_output.exists())
            loaded = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(loaded, result)
            self.assertTrue(wav_path.exists())
            self.assertEqual(wav_path.stat().st_size, input_size)
            all_files = sorted(path.relative_to(temp_path).as_posix() for path in temp_path.rglob("*") if path.is_file())
            self.assertEqual(all_files, ["outputs/audio_vad_result.json", "sample_16k_mono.wav"])


if __name__ == "__main__":
    unittest.main()
