from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from services.ai.audio_pipeline import run_audio_stage1


def _preprocess_result(*, input_path: Path, normalized_wav_path: Path) -> dict[str, Any]:
    return {
        "audio_preprocess": {
            "input_path": str(input_path),
            "normalized_wav_path": str(normalized_wav_path),
            "original": {
                "codec": "mp3",
                "bitrate": 128000,
                "duration_sec": 8.0,
                "sample_rate": None,
                "channel_count": None,
            },
            "normalized": {
                "codec": "pcm_s16le",
                "duration_sec": 8.0,
                "sample_rate": 16000,
                "channel_count": 1,
            },
            "quality_flags": ["LOW_BITRATE_SOURCE", "ENERGY_FALLBACK_VAD_USED"],
        },
        "limits": {"unsupported_reason": None, "low_evidence_reason": None},
    }


def _vad_result(*, quality_flags: list[str] | None = None, limits: dict[str, str | None] | None = None) -> dict[str, Any]:
    return {
        "audio_vad": {
            "human_speech_detected": True,
            "quality_flags": quality_flags or ["LONG_LEADING_SILENCE", "LOW_SPEECH_RATIO"],
            "speech_stats": {
                "speech_duration_sec": 3.0,
                "silence_duration_sec": 5.0,
                "speech_ratio": 0.375,
                "silence_ratio": 0.625,
                "pause_count": 2,
                "leading_silence": 0.5,
                "trailing_silence": 0.7,
                "breathing_like_pattern": {
                    "detected": True,
                    "candidate_gap_count": 1,
                    "candidate_gap_durations_sec": [0.6],
                },
            },
        },
        "limits": limits or {"unsupported_reason": None, "low_evidence_reason": None},
    }


def _windows_result(normalized_wav_path: Path) -> dict[str, Any]:
    return {
        "audio_windows": {
            "input_wav_path": str(normalized_wav_path),
            "source_vad_json": str(normalized_wav_path.parent / "audio_vad_result.json"),
            "total_duration_sec": 8.0,
            "window_sec": 4.0,
            "hop_sec": 2.0,
            "window_count": 2,
            "windows": [],
            "quality_flags": ["LOW_SPEECH_RATIO"],
        },
        "limits": {"unsupported_reason": None, "low_evidence_reason": None},
    }


def _inference_result(normalized_wav_path: Path, *, limits: dict[str, str | None] | None = None) -> dict[str, Any]:
    return {
        "audio_inference": {
            "input_wav_path": str(normalized_wav_path),
            "source_windows_json": str(normalized_wav_path.parent / "audio_windows_result.json"),
            "source_vad_json": str(normalized_wav_path.parent / "audio_vad_result.json"),
            "model_name": "AntiDeepfake",
            "window_count": 3,
            "scored_window_count": 2,
            "skipped_window_count": 1,
            "failed_window_count": 0,
            "score_summary": {
                "audio_fake_score_raw_mean": 0.5,
                "audio_fake_score_raw_max": 0.8,
                "audio_fake_prob_like_mean": 0.4,
                "audio_fake_prob_like_max": 0.6,
            },
            "windows": [
                {
                    "window_id": 0,
                    "start": 0.0,
                    "end": 4.0,
                    "duration": 4.0,
                    "audio_fake_score_raw": 0.2,
                    "audio_real_score_raw": 1.1,
                    "audio_fake_prob_like": 0.2,
                    "inference_status": "scored",
                },
                {
                    "window_id": 1,
                    "start": 2.0,
                    "end": 6.0,
                    "duration": 4.0,
                    "audio_fake_score_raw": 0.8,
                    "audio_real_score_raw": 0.4,
                    "audio_fake_prob_like": 0.6,
                    "inference_status": "scored",
                },
                {
                    "window_id": 2,
                    "start": 4.0,
                    "end": 8.0,
                    "duration": 4.0,
                    "audio_fake_score_raw": None,
                    "audio_real_score_raw": None,
                    "audio_fake_prob_like": None,
                    "inference_status": "skipped_no_speech",
                },
            ],
            "quality_flags": ["LOW_SPEECH_RATIO", "INFERENCE_FAILED"],
        },
        "limits": limits or {"unsupported_reason": None, "low_evidence_reason": None},
    }


def _segments_result(normalized_wav_path: Path, *, limits: dict[str, str | None] | None = None) -> dict[str, Any]:
    return {
        "audio_summary": {
            "input_wav_path": str(normalized_wav_path),
            "source_inference_json": str(normalized_wav_path.parent / "audio_inference_result.json"),
            "source_windows_json": str(normalized_wav_path.parent / "audio_windows_result.json"),
            "source_vad_json": str(normalized_wav_path.parent / "audio_vad_result.json"),
            "model_name": "AntiDeepfake",
            "score_summary": {
                "audio_fake_prob_like_mean": 0.4,
                "audio_fake_prob_like_max": 0.6,
                "audio_fake_prob_like_variance": 0.04,
                "audio_fake_score_raw_mean": 0.5,
                "audio_fake_score_raw_max": 0.8,
                "audio_fake_score_raw_variance": 0.09,
            },
            "top_suspicious_audio_segments": [
                {
                    "segment_id": 0,
                    "start": 0.0,
                    "end": 6.0,
                    "duration": 6.0,
                    "window_ids": [0, 1],
                    "window_count": 2,
                    "max_fake_prob_like": 0.6,
                    "mean_fake_prob_like": 0.4,
                    "max_fake_score_raw": 0.8,
                    "mean_fake_score_raw": 0.5,
                    "score_variance": 0.04,
                    "reason": "consecutive_windows_above_threshold",
                }
            ],
            "quality_flags": ["LOW_SPEECH_RATIO", "NO_SCORED_WINDOWS_FOR_SUMMARY"],
        },
        "limits": limits or {"unsupported_reason": None, "low_evidence_reason": None},
    }


class AudioStage1Tests(unittest.TestCase):
    def test_run_audio_stage1_returns_final_schema_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "sample.mp3"
            input_path.write_bytes(b"input")
            output_dir = temp_path / "artifacts"
            normalized_wav_path = output_dir / "normalized" / "sample_16k_mono.wav"

            with patch(
                "services.ai.audio_pipeline.audio_stage1.preprocess_audio",
                return_value=_preprocess_result(input_path=input_path, normalized_wav_path=normalized_wav_path),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_vad",
                return_value=_vad_result(),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_windowing",
                return_value=_windows_result(normalized_wav_path),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_inference",
                return_value=_inference_result(normalized_wav_path),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_segments",
                return_value=_segments_result(normalized_wav_path),
            ):
                result = run_audio_stage1(input_path=input_path, output_dir=output_dir, request_id="req-1")

            output_json = output_dir / "audio_stage1_result.json"
            self.assertTrue(output_json.exists())
            self.assertEqual(json.loads(output_json.read_text(encoding="utf-8")), result)
            self.assertEqual(result["request_id"], "req-1")
            self.assertEqual(result["file_path"], str(input_path.resolve()))
            self.assertEqual(result["evidence_level"], "sufficient")
            self.assertEqual(result["original_metadata"]["sample_rate_hz"], 16000)
            self.assertEqual(result["original_metadata"]["channel_count"], 1)
            self.assertAlmostEqual(result["audio_fake_score_raw"], 0.5)
            self.assertAlmostEqual(result["audio_real_score_raw"], 0.75)
            self.assertAlmostEqual(result["audio_fake_prob_like"], 0.4)
            self.assertAlmostEqual(result["audio_uncertainty"], 0.16)
            self.assertEqual(result["quality_flags"], ["low_bitrate_source", "long_leading_silence", "low_speech_ratio"])
            self.assertEqual(len(result["top_suspicious_audio_segments"]), 1)
            self.assertEqual(result["top_suspicious_audio_segments"][0]["rank"], 1)
            self.assertAlmostEqual(result["top_suspicious_audio_segments"][0]["fake_score_raw"], 0.8)
            self.assertAlmostEqual(result["top_suspicious_audio_segments"][0]["real_score_raw"], 0.4)
            self.assertAlmostEqual(result["top_suspicious_audio_segments"][0]["fake_prob_like"], 0.6)

    def test_unsupported_content_maps_to_unsupported_evidence_and_full_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "sample.wav"
            input_path.write_bytes(b"input")
            output_dir = temp_path / "artifacts"
            normalized_wav_path = output_dir / "normalized" / "sample_16k_mono.wav"
            unsupported_limits = {"unsupported_reason": "no_detected_human_speech", "low_evidence_reason": None}
            unsupported_summary = _segments_result(normalized_wav_path, limits=unsupported_limits)["audio_summary"]

            with patch(
                "services.ai.audio_pipeline.audio_stage1.preprocess_audio",
                return_value=_preprocess_result(input_path=input_path, normalized_wav_path=normalized_wav_path),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_vad",
                return_value={
                    "audio_vad": {
                        "human_speech_detected": False,
                        "quality_flags": ["NO_DETECTED_SPEECH"],
                        "speech_stats": {
                            "speech_duration_sec": 0.0,
                            "silence_duration_sec": 8.0,
                            "speech_ratio": 0.0,
                            "silence_ratio": 1.0,
                            "pause_count": 0,
                            "leading_silence": 8.0,
                            "trailing_silence": 8.0,
                            "breathing_like_pattern": None,
                        },
                    },
                    "limits": unsupported_limits,
                },
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_windowing",
                return_value=_windows_result(normalized_wav_path),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_inference",
                return_value=_inference_result(normalized_wav_path, limits=unsupported_limits),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_segments",
                return_value={
                    "audio_summary": {
                        **unsupported_summary,
                        "score_summary": {
                            "audio_fake_prob_like_mean": None,
                            "audio_fake_prob_like_max": None,
                            "audio_fake_prob_like_variance": None,
                            "audio_fake_score_raw_mean": None,
                            "audio_fake_score_raw_max": None,
                            "audio_fake_score_raw_variance": None,
                        },
                        "top_suspicious_audio_segments": [],
                    },
                    "limits": unsupported_limits,
                },
            ):
                result = run_audio_stage1(input_path=input_path, output_dir=output_dir, request_id="req-unsupported")

            self.assertEqual(result["evidence_level"], "unsupported_content")
            self.assertFalse(result["human_speech_detected"])
            self.assertEqual(result["audio_uncertainty"], 1.0)
            self.assertIn("no_human_speech", result["quality_flags"])

    def test_low_evidence_maps_to_low_evidence_and_uncertainty_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "sample.wav"
            input_path.write_bytes(b"input")
            output_dir = temp_path / "artifacts"
            normalized_wav_path = output_dir / "normalized" / "sample_16k_mono.wav"
            low_limits = {"unsupported_reason": None, "low_evidence_reason": "too_little_detected_speech"}

            with patch(
                "services.ai.audio_pipeline.audio_stage1.preprocess_audio",
                return_value=_preprocess_result(input_path=input_path, normalized_wav_path=normalized_wav_path),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_vad",
                return_value=_vad_result(limits=low_limits),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_windowing",
                return_value=_windows_result(normalized_wav_path),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_inference",
                return_value=_inference_result(normalized_wav_path, limits=low_limits),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_segments",
                return_value=_segments_result(normalized_wav_path, limits=low_limits),
            ):
                result = run_audio_stage1(input_path=input_path, output_dir=output_dir, request_id="req-low")

            self.assertEqual(result["evidence_level"], "low_evidence")
            self.assertGreaterEqual(result["audio_uncertainty"], 0.6)

    def test_no_scored_windows_forces_full_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "sample.wav"
            input_path.write_bytes(b"input")
            output_dir = temp_path / "artifacts"
            normalized_wav_path = output_dir / "normalized" / "sample_16k_mono.wav"
            inference_result = _inference_result(normalized_wav_path)
            inference_result["audio_inference"]["windows"] = [
                {
                    "window_id": 0,
                    "start": 0.0,
                    "end": 4.0,
                    "duration": 4.0,
                    "audio_fake_score_raw": None,
                    "audio_real_score_raw": None,
                    "audio_fake_prob_like": None,
                    "inference_status": "skipped_no_speech",
                }
            ]
            inference_result["audio_inference"]["scored_window_count"] = 0
            inference_result["audio_inference"]["skipped_window_count"] = 1
            inference_result["audio_inference"]["score_summary"] = {
                "audio_fake_score_raw_mean": None,
                "audio_fake_score_raw_max": None,
                "audio_fake_prob_like_mean": None,
                "audio_fake_prob_like_max": None,
            }
            segments_result = _segments_result(normalized_wav_path)
            segments_result["audio_summary"]["score_summary"] = {
                "audio_fake_prob_like_mean": None,
                "audio_fake_prob_like_max": None,
                "audio_fake_prob_like_variance": None,
                "audio_fake_score_raw_mean": None,
                "audio_fake_score_raw_max": None,
                "audio_fake_score_raw_variance": None,
            }
            segments_result["audio_summary"]["top_suspicious_audio_segments"] = []

            with patch(
                "services.ai.audio_pipeline.audio_stage1.preprocess_audio",
                return_value=_preprocess_result(input_path=input_path, normalized_wav_path=normalized_wav_path),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_vad",
                return_value=_vad_result(),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_windowing",
                return_value=_windows_result(normalized_wav_path),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_inference",
                return_value=inference_result,
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_segments",
                return_value=segments_result,
            ):
                result = run_audio_stage1(input_path=input_path, output_dir=output_dir, request_id="req-none")

            self.assertEqual(result["evidence_level"], "low_evidence")
            self.assertEqual(result["audio_uncertainty"], 1.0)
            self.assertIsNone(result["audio_fake_score_raw"])
            self.assertIsNone(result["audio_real_score_raw"])
            self.assertIsNone(result["audio_fake_prob_like"])
            self.assertEqual(result["scored_window_count"], 0)
            self.assertEqual(result["skipped_window_count"], 1)
            self.assertEqual(result["top_suspicious_audio_segments"], [])

    def test_failed_inference_degrades_evidence_level(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "sample.wav"
            input_path.write_bytes(b"input")
            output_dir = temp_path / "artifacts"
            normalized_wav_path = output_dir / "normalized" / "sample_16k_mono.wav"
            inference_result = _inference_result(normalized_wav_path)
            inference_result["audio_inference"]["failed_window_count"] = 1
            inference_result["audio_inference"]["model_errors"] = ["RuntimeError: model runtime failed"]
            inference_result["audio_inference"]["windows"][1]["inference_status"] = "failed_model_error"
            inference_result["audio_inference"]["windows"][1]["model_error"] = "RuntimeError: model runtime failed"
            inference_result["audio_inference"]["windows"][1]["audio_fake_score_raw"] = None
            inference_result["audio_inference"]["windows"][1]["audio_real_score_raw"] = None
            inference_result["audio_inference"]["windows"][1]["audio_fake_prob_like"] = None
            inference_result["audio_inference"]["scored_window_count"] = 1

            with patch(
                "services.ai.audio_pipeline.audio_stage1.preprocess_audio",
                return_value=_preprocess_result(input_path=input_path, normalized_wav_path=normalized_wav_path),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_vad",
                return_value=_vad_result(),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_windowing",
                return_value=_windows_result(normalized_wav_path),
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_inference",
                return_value=inference_result,
            ), patch(
                "services.ai.audio_pipeline.audio_stage1.run_audio_segments",
                return_value=_segments_result(normalized_wav_path),
            ):
                result = run_audio_stage1(input_path=input_path, output_dir=output_dir, request_id="req-failed")

            self.assertEqual(result["evidence_level"], "low_evidence")
            self.assertEqual(result["audio_model_error"], "RuntimeError: model runtime failed")


if __name__ == "__main__":
    unittest.main()
