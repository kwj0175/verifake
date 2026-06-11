from __future__ import annotations

import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from services.ai.audio_pipeline.antideepfake import AntiDeepfakeInferenceResult
from services.ai.audio_pipeline.audio_inference import AudioInferenceError, run_audio_inference


def _write_mono_wav(path: Path, *, sample_rate: int = 16000, duration_sec: float = 9.6, amplitude: int = 4000) -> None:
    frame_count = int(sample_rate * duration_sec)
    timeline = np.linspace(0.0, duration_sec, frame_count, endpoint=False)
    waveform = (np.sin(2.0 * np.pi * 220.0 * timeline) * amplitude).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(waveform.tobytes())


def _write_windows_json(
    path: Path,
    *,
    wav_path: Path,
    total_duration_sec: float,
    windows: list[dict[str, object]],
    source_vad_json: str | None = None,
    quality_flags: list[str] | None = None,
    limits: dict[str, str | None] | None = None,
    window_sec: float = 4.0,
    hop_sec: float = 2.0,
) -> None:
    payload = {
        "audio_windows": {
            "input_wav_path": str(wav_path),
            "source_vad_json": source_vad_json,
            "total_duration_sec": total_duration_sec,
            "window_sec": window_sec,
            "hop_sec": hop_sec,
            "window_count": len(windows),
            "windows": windows,
            "quality_flags": quality_flags or [],
        },
        "limits": limits or {"unsupported_reason": None, "low_evidence_reason": None},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class AudioInferenceTests(unittest.TestCase):
    def test_windows_json_creates_audio_inference_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            windows_json = temp_path / "audio_windows_result.json"
            json_output = temp_path / "outputs" / "audio_inference_result.json"
            _write_mono_wav(wav_path, duration_sec=8.0)
            _write_windows_json(
                windows_json,
                wav_path=wav_path,
                total_duration_sec=8.0,
                source_vad_json=str(temp_path / "audio_vad_result.json"),
                quality_flags=["ENERGY_FALLBACK_VAD_USED"],
                limits={"unsupported_reason": None, "low_evidence_reason": "short_speech_region"},
                windows=[
                    {
                        "window_id": 0,
                        "start": 0.0,
                        "end": 4.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 2.0,
                        "speech_coverage_ratio": 0.5,
                        "has_speech": True,
                    },
                    {
                        "window_id": 1,
                        "start": 2.0,
                        "end": 6.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 0.0,
                        "speech_coverage_ratio": 0.0,
                        "has_speech": False,
                    },
                ],
            )

            def fake_inference(*args: object, **kwargs: object) -> AntiDeepfakeInferenceResult:
                return AntiDeepfakeInferenceResult(
                    request_id="window-0000",
                    file_path=str(args[0]),
                    fake_logit=0.25,
                    real_logit=1.25,
                    fake_probability=0.268941,
                    real_probability=0.731059,
                    predicted_label="real",
                    score_csv_path=None,
                )

            with patch(
                "services.ai.audio_pipeline.audio_inference.run_antideepfake_inference",
                side_effect=fake_inference,
            ) as mock_inference:
                result = run_audio_inference(
                    windows_json_path=windows_json,
                    json_output_path=json_output,
                )

            self.assertTrue(json_output.exists())
            loaded = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(loaded, result)
            self.assertEqual(mock_inference.call_count, 1)
            self.assertEqual(result["audio_inference"]["window_count"], 2)
            self.assertEqual(result["audio_inference"]["scored_window_count"], 1)
            self.assertEqual(result["audio_inference"]["skipped_window_count"], 1)
            self.assertEqual(result["audio_inference"]["failed_window_count"], 0)
            self.assertEqual(result["limits"]["low_evidence_reason"], "short_speech_region")
            self.assertEqual(result["audio_inference"]["quality_flags"], ["ENERGY_FALLBACK_VAD_USED"])

    def test_window_metadata_is_preserved_and_summary_uses_scored_windows_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            windows_json = temp_path / "audio_windows_result.json"
            _write_mono_wav(wav_path, duration_sec=10.0)
            _write_windows_json(
                windows_json,
                wav_path=wav_path,
                total_duration_sec=10.0,
                source_vad_json=str(temp_path / "audio_vad_result.json"),
                windows=[
                    {
                        "window_id": 0,
                        "start": 0.0,
                        "end": 4.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 1.0,
                        "speech_coverage_ratio": 0.25,
                        "has_speech": True,
                    },
                    {
                        "window_id": 1,
                        "start": 4.0,
                        "end": 8.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 2.0,
                        "speech_coverage_ratio": 0.5,
                        "has_speech": True,
                    },
                    {
                        "window_id": 2,
                        "start": 8.0,
                        "end": 10.0,
                        "duration": 2.0,
                        "speech_overlap_sec": 0.0,
                        "speech_coverage_ratio": 0.0,
                        "has_speech": False,
                    },
                ],
            )

            scores = {
                "window-0000": (0.0, 2.0, 0.119203),
                "window-0001": (1.0, 1.0, 0.5),
            }

            def fake_inference(*args: object, **kwargs: object) -> AntiDeepfakeInferenceResult:
                request_id = str(kwargs["request_id"])
                fake_logit, real_logit, fake_prob = scores[request_id]
                return AntiDeepfakeInferenceResult(
                    request_id=request_id,
                    file_path=str(args[0]),
                    fake_logit=fake_logit,
                    real_logit=real_logit,
                    fake_probability=fake_prob,
                    real_probability=round(1.0 - fake_prob, 6),
                    predicted_label="real",
                    score_csv_path=None,
                )

            with patch(
                "services.ai.audio_pipeline.audio_inference.run_antideepfake_inference",
                side_effect=fake_inference,
            ), patch.dict(os.environ, {"VERIFAKE_FAST_AUDIO_ONCE": "0"}):
                result = run_audio_inference(windows_json_path=windows_json)

            windows = result["audio_inference"]["windows"]
            self.assertEqual(windows[0]["window_id"], 0)
            self.assertEqual((windows[0]["start"], windows[0]["end"]), (0.0, 4.0))
            self.assertEqual(windows[2]["inference_status"], "skipped_no_speech")
            self.assertAlmostEqual(windows[0]["audio_fake_prob_like"], 0.119203)
            self.assertAlmostEqual(windows[1]["audio_fake_prob_like"], 0.5)
            self.assertGreaterEqual(windows[0]["audio_fake_prob_like"], 0.0)
            self.assertLessEqual(windows[0]["audio_fake_prob_like"], 1.0)

            summary = result["audio_inference"]["score_summary"]
            self.assertAlmostEqual(summary["audio_fake_score_raw_mean"], 0.5)
            self.assertAlmostEqual(summary["audio_fake_score_raw_max"], 1.0)
            self.assertAlmostEqual(summary["audio_fake_prob_like_mean"], 0.3096015, places=6)
            self.assertAlmostEqual(summary["audio_fake_prob_like_max"], 0.5)
            self.assertEqual(result["audio_inference"]["failed_window_count"], 0)

    def test_low_speech_coverage_windows_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            windows_json = temp_path / "audio_windows_result.json"
            _write_mono_wav(wav_path, duration_sec=4.0)
            _write_windows_json(
                windows_json,
                wav_path=wav_path,
                total_duration_sec=4.0,
                source_vad_json=str(temp_path / "audio_vad_result.json"),
                windows=[
                    {
                        "window_id": 0,
                        "start": 0.0,
                        "end": 4.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 0.1,
                        "speech_coverage_ratio": 0.025,
                        "has_speech": True,
                    }
                ],
            )

            with patch("services.ai.audio_pipeline.audio_inference.run_antideepfake_inference") as mock_inference:
                result = run_audio_inference(windows_json_path=windows_json)

            self.assertEqual(mock_inference.call_count, 0)
            self.assertEqual(result["audio_inference"]["windows"][0]["inference_status"], "skipped_low_speech_coverage")

    def test_stage_unsupported_skips_all_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            windows_json = temp_path / "audio_windows_result.json"
            _write_mono_wav(wav_path, duration_sec=4.0)
            _write_windows_json(
                windows_json,
                wav_path=wav_path,
                total_duration_sec=4.0,
                source_vad_json=str(temp_path / "audio_vad_result.json"),
                limits={"unsupported_reason": "no_detected_human_speech", "low_evidence_reason": None},
                windows=[
                    {
                        "window_id": 0,
                        "start": 0.0,
                        "end": 4.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 0.0,
                        "speech_coverage_ratio": 0.0,
                        "has_speech": False,
                    }
                ],
            )

            with patch("services.ai.audio_pipeline.audio_inference.run_antideepfake_inference") as mock_inference:
                result = run_audio_inference(windows_json_path=windows_json)

            self.assertEqual(mock_inference.call_count, 0)
            self.assertEqual(result["audio_inference"]["windows"][0]["inference_status"], "skipped_stage_unsupported")

    def test_invalid_window_is_kept_with_invalid_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            windows_json = temp_path / "audio_windows_result.json"
            _write_mono_wav(wav_path, duration_sec=4.0)
            _write_windows_json(
                windows_json,
                wav_path=wav_path,
                total_duration_sec=4.0,
                source_vad_json=str(temp_path / "audio_vad_result.json"),
                windows=[
                    {
                        "window_id": 0,
                        "start": 2.0,
                        "end": 1.0,
                        "duration": -1.0,
                        "speech_overlap_sec": 0.0,
                        "speech_coverage_ratio": 0.0,
                        "has_speech": True,
                    }
                ],
            )

            with patch("services.ai.audio_pipeline.audio_inference.run_antideepfake_inference") as mock_inference:
                result = run_audio_inference(windows_json_path=windows_json)

            self.assertEqual(mock_inference.call_count, 0)
            self.assertEqual(result["audio_inference"]["windows"][0]["inference_status"], "skipped_invalid_window")

    def test_final_and_suspicious_fields_are_not_created(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            windows_json = temp_path / "audio_windows_result.json"
            _write_mono_wav(wav_path, duration_sec=4.0)
            _write_windows_json(
                windows_json,
                wav_path=wav_path,
                total_duration_sec=4.0,
                source_vad_json=str(temp_path / "audio_vad_result.json"),
                windows=[
                    {
                        "window_id": 0,
                        "start": 0.0,
                        "end": 4.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 1.0,
                        "speech_coverage_ratio": 0.25,
                        "has_speech": True,
                    }
                ],
            )

            with patch(
                "services.ai.audio_pipeline.audio_inference.run_antideepfake_inference",
                return_value=AntiDeepfakeInferenceResult(
                    request_id="window-0000",
                    file_path=str(wav_path),
                    fake_logit=0.1,
                    real_logit=0.9,
                    fake_probability=0.31,
                    real_probability=0.69,
                    predicted_label="real",
                    score_csv_path=None,
                ),
            ):
                result = run_audio_inference(windows_json_path=windows_json)

            self.assertNotIn("top_suspicious_audio_segments", result["audio_inference"])
            self.assertNotIn("audio_fake_prob", result["audio_inference"])
            self.assertNotIn("audio_uncertainty", result["audio_inference"])
            self.assertNotIn("fake_score", result["audio_inference"]["windows"][0])
            self.assertNotIn("segment_score", result["audio_inference"]["windows"][0])

    def test_failed_model_error_is_not_counted_as_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "normalized" / "sample_16k_mono.wav"
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            windows_json = temp_path / "audio_windows_result.json"
            _write_mono_wav(wav_path, duration_sec=4.0)
            _write_windows_json(
                windows_json,
                wav_path=wav_path,
                total_duration_sec=4.0,
                source_vad_json=str(temp_path / "audio_vad_result.json"),
                quality_flags=["ENERGY_FALLBACK_VAD_USED"],
                windows=[
                    {
                        "window_id": 0,
                        "start": 0.0,
                        "end": 4.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 1.0,
                        "speech_coverage_ratio": 0.25,
                        "has_speech": True,
                    }
                ],
            )

            with patch(
                "services.ai.audio_pipeline.audio_inference.run_antideepfake_inference",
                side_effect=RuntimeError("model runtime failed"),
            ):
                result = run_audio_inference(windows_json_path=windows_json)

            self.assertEqual(result["audio_inference"]["scored_window_count"], 0)
            self.assertEqual(result["audio_inference"]["skipped_window_count"], 0)
            self.assertEqual(result["audio_inference"]["failed_window_count"], 1)
            self.assertEqual(result["audio_inference"]["windows"][0]["inference_status"], "failed_model_error")
            self.assertEqual(
                result["audio_inference"]["windows"][0]["model_error"],
                "RuntimeError: model runtime failed",
            )
            self.assertEqual(result["audio_inference"]["model_errors"], ["RuntimeError: model runtime failed"])
            self.assertIn("INFERENCE_FAILED", result["audio_inference"]["quality_flags"])

    def test_input_wav_path_must_stay_under_stage_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            outside_wav_dir = temp_path.parent / "outside_audio"
            outside_wav_dir.mkdir(parents=True, exist_ok=True)
            wav_path = outside_wav_dir / "sample_16k_mono.wav"
            windows_json = temp_path / "audio_windows_result.json"
            _write_mono_wav(wav_path, duration_sec=4.0)
            _write_windows_json(
                windows_json,
                wav_path=wav_path,
                total_duration_sec=4.0,
                source_vad_json=None,
                windows=[
                    {
                        "window_id": 0,
                        "start": 0.0,
                        "end": 4.0,
                        "duration": 4.0,
                        "speech_overlap_sec": None,
                        "speech_coverage_ratio": None,
                        "has_speech": True,
                    }
                ],
            )

            with self.assertRaises(AudioInferenceError):
                run_audio_inference(windows_json_path=windows_json, skip_no_speech_windows=False)

    def test_checkpoint_missing_in_real_inference_path_raises_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            windows_json = temp_path / "audio_windows_result.json"
            _write_mono_wav(wav_path, duration_sec=4.0)
            _write_windows_json(
                windows_json,
                wav_path=wav_path,
                total_duration_sec=4.0,
                source_vad_json=str(temp_path / "audio_vad_result.json"),
                windows=[
                    {
                        "window_id": 0,
                        "start": 0.0,
                        "end": 4.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 1.0,
                        "speech_coverage_ratio": 0.25,
                        "has_speech": True,
                    }
                ],
            )

            with self.assertRaises(FileNotFoundError):
                run_audio_inference(
                    windows_json_path=windows_json,
                    checkpoint_path=temp_path / "missing.ckpt",
                )

    def test_preprocess_vad_windowing_are_not_reexecuted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample_16k_mono.wav"
            windows_json = temp_path / "audio_windows_result.json"
            _write_mono_wav(wav_path, duration_sec=4.0)
            _write_windows_json(
                windows_json,
                wav_path=wav_path,
                total_duration_sec=4.0,
                source_vad_json=str(temp_path / "audio_vad_result.json"),
                windows=[
                    {
                        "window_id": 0,
                        "start": 0.0,
                        "end": 4.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 1.0,
                        "speech_coverage_ratio": 0.25,
                        "has_speech": True,
                    }
                ],
            )

            with patch(
                "services.ai.audio_pipeline.audio_preprocess.preprocess_audio",
                side_effect=AssertionError("preprocess should not run"),
            ), patch(
                "services.ai.audio_pipeline.audio_vad.run_audio_vad",
                side_effect=AssertionError("vad should not run"),
            ), patch(
                "services.ai.audio_pipeline.audio_windowing.run_audio_windowing",
                side_effect=AssertionError("windowing should not run"),
            ), patch(
                "services.ai.audio_pipeline.audio_inference.run_antideepfake_inference",
                return_value=AntiDeepfakeInferenceResult(
                    request_id="window-0000",
                    file_path=str(wav_path),
                    fake_logit=0.1,
                    real_logit=0.9,
                    fake_probability=0.31,
                    real_probability=0.69,
                    predicted_label="real",
                    score_csv_path=None,
                ),
            ):
                result = run_audio_inference(windows_json_path=windows_json)

            self.assertEqual(result["audio_inference"]["windows"][0]["inference_status"], "scored")


if __name__ == "__main__":
    unittest.main()
