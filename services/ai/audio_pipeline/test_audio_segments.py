from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.ai.audio_pipeline.audio_segments import AudioSegmentsError, run_audio_segments


def _write_inference_json(
    path: Path,
    *,
    input_wav_path: str = "/tmp/sample.wav",
    source_windows_json: str = "/tmp/audio_windows_result.json",
    source_vad_json: str | None = "/tmp/audio_vad_result.json",
    model_name: str = "AntiDeepfake",
    windows: list[dict[str, object]],
    quality_flags: list[str] | None = None,
    limits: dict[str, str | None] | None = None,
) -> None:
    payload = {
        "audio_inference": {
            "input_wav_path": input_wav_path,
            "source_windows_json": source_windows_json,
            "source_vad_json": source_vad_json,
            "model_name": model_name,
            "window_sec": 4.0,
            "hop_sec": 2.0,
            "window_count": len(windows),
            "scored_window_count": sum(1 for window in windows if window["inference_status"] == "scored"),
            "skipped_window_count": sum(
                1 for window in windows if str(window["inference_status"]).startswith("skipped_")
            ),
            "failed_window_count": sum(1 for window in windows if window["inference_status"] == "failed_model_error"),
            "score_summary": {},
            "windows": windows,
            "quality_flags": quality_flags or [],
        },
        "limits": limits or {"unsupported_reason": None, "low_evidence_reason": None},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class AudioSegmentsTests(unittest.TestCase):
    def test_inference_json_creates_audio_segments_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            inference_json = temp_path / "audio_inference_result.json"
            json_output = temp_path / "outputs" / "audio_segments_result.json"
            _write_inference_json(
                inference_json,
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
                        "audio_fake_score_raw": 0.25,
                        "audio_real_score_raw": 1.25,
                        "audio_fake_prob_like": 0.6,
                        "inference_status": "scored",
                    }
                ],
            )

            result = run_audio_segments(inference_json_path=inference_json, json_output_path=json_output)

            self.assertTrue(json_output.exists())
            loaded = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(loaded, result)
            self.assertEqual(result["audio_summary"]["window_count"], 1)
            self.assertEqual(result["audio_summary"]["scored_window_count"], 1)
            self.assertEqual(result["limits"]["low_evidence_reason"], "short_speech_region")
            self.assertEqual(result["audio_summary"]["quality_flags"], ["ENERGY_FALLBACK_VAD_USED"])

    def test_only_scored_windows_become_suspicious_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            inference_json = Path(temp_dir) / "audio_inference_result.json"
            _write_inference_json(
                inference_json,
                windows=[
                    {
                        "window_id": 0,
                        "start": 0.0,
                        "end": 4.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 1.0,
                        "speech_coverage_ratio": 0.25,
                        "has_speech": True,
                        "audio_fake_score_raw": 1.0,
                        "audio_real_score_raw": 0.0,
                        "audio_fake_prob_like": 0.7,
                        "inference_status": "scored",
                    },
                    {
                        "window_id": 1,
                        "start": 2.0,
                        "end": 6.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 1.0,
                        "speech_coverage_ratio": 0.25,
                        "has_speech": False,
                        "audio_fake_score_raw": None,
                        "audio_real_score_raw": None,
                        "audio_fake_prob_like": None,
                        "inference_status": "skipped_no_speech",
                    },
                    {
                        "window_id": 2,
                        "start": 6.0,
                        "end": 10.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 1.0,
                        "speech_coverage_ratio": 0.25,
                        "has_speech": True,
                        "audio_fake_score_raw": None,
                        "audio_real_score_raw": None,
                        "audio_fake_prob_like": None,
                        "inference_status": "failed_model_error",
                    },
                ],
            )

            result = run_audio_segments(inference_json_path=inference_json)
            self.assertEqual(result["audio_summary"]["suspicious_window_count"], 1)
            self.assertEqual(result["audio_summary"]["suspicious_segment_count"], 1)
            self.assertEqual(result["audio_summary"]["top_suspicious_audio_segments"][0]["window_ids"], [0])

    def test_threshold_is_inclusive_at_exact_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            inference_json = Path(temp_dir) / "audio_inference_result.json"
            _write_inference_json(
                inference_json,
                windows=[
                    {
                        "window_id": 0,
                        "start": 0.0,
                        "end": 4.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 1.0,
                        "speech_coverage_ratio": 0.25,
                        "has_speech": True,
                        "audio_fake_score_raw": 1.0,
                        "audio_real_score_raw": 0.0,
                        "audio_fake_prob_like": 0.5,
                        "inference_status": "scored",
                    }
                ],
            )
            result = run_audio_segments(inference_json_path=inference_json, suspicious_threshold=0.5)
            self.assertEqual(result["audio_summary"]["suspicious_window_count"], 1)

    def test_overlapping_or_small_gap_windows_merge_into_one_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            inference_json = Path(temp_dir) / "audio_inference_result.json"
            _write_inference_json(
                inference_json,
                windows=[
                    {
                        "window_id": 0,
                        "start": 0.0,
                        "end": 4.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 1.0,
                        "speech_coverage_ratio": 0.25,
                        "has_speech": True,
                        "audio_fake_score_raw": 2.0,
                        "audio_real_score_raw": -1.0,
                        "audio_fake_prob_like": 0.7,
                        "inference_status": "scored",
                    },
                    {
                        "window_id": 1,
                        "start": 4.4,
                        "end": 8.0,
                        "duration": 3.6,
                        "speech_overlap_sec": 1.0,
                        "speech_coverage_ratio": 0.25,
                        "has_speech": True,
                        "audio_fake_score_raw": 1.0,
                        "audio_real_score_raw": -1.0,
                        "audio_fake_prob_like": 0.8,
                        "inference_status": "scored",
                    },
                ],
            )
            result = run_audio_segments(inference_json_path=inference_json, max_merge_gap_sec=0.5)
            segments = result["audio_summary"]["top_suspicious_audio_segments"]
            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0]["window_ids"], [0, 1])
            self.assertEqual(segments[0]["reason"], "consecutive_windows_above_threshold")

    def test_large_gap_windows_stay_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            inference_json = Path(temp_dir) / "audio_inference_result.json"
            _write_inference_json(
                inference_json,
                windows=[
                    {
                        "window_id": 0,
                        "start": 0.0,
                        "end": 4.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 1.0,
                        "speech_coverage_ratio": 0.25,
                        "has_speech": True,
                        "audio_fake_score_raw": 2.0,
                        "audio_real_score_raw": -1.0,
                        "audio_fake_prob_like": 0.7,
                        "inference_status": "scored",
                    },
                    {
                        "window_id": 1,
                        "start": 6.0,
                        "end": 10.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 1.0,
                        "speech_coverage_ratio": 0.25,
                        "has_speech": True,
                        "audio_fake_score_raw": 3.0,
                        "audio_real_score_raw": -1.0,
                        "audio_fake_prob_like": 0.8,
                        "inference_status": "scored",
                    },
                ],
            )
            result = run_audio_segments(inference_json_path=inference_json, max_merge_gap_sec=0.5)
            self.assertEqual(result["audio_summary"]["suspicious_segment_count"], 2)

    def test_top_k_and_ranking_are_applied_after_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            inference_json = Path(temp_dir) / "audio_inference_result.json"
            _write_inference_json(
                inference_json,
                windows=[
                    {"window_id": 0, "start": 0.0, "end": 4.0, "duration": 4.0, "speech_overlap_sec": 1.0, "speech_coverage_ratio": 0.25, "has_speech": True, "audio_fake_score_raw": 1.0, "audio_real_score_raw": -1.0, "audio_fake_prob_like": 0.6, "inference_status": "scored"},
                    {"window_id": 1, "start": 10.0, "end": 14.0, "duration": 4.0, "speech_overlap_sec": 1.0, "speech_coverage_ratio": 0.25, "has_speech": True, "audio_fake_score_raw": 2.0, "audio_real_score_raw": -1.0, "audio_fake_prob_like": 0.9, "inference_status": "scored"},
                    {"window_id": 2, "start": 20.0, "end": 24.0, "duration": 4.0, "speech_overlap_sec": 1.0, "speech_coverage_ratio": 0.25, "has_speech": True, "audio_fake_score_raw": 1.5, "audio_real_score_raw": -1.0, "audio_fake_prob_like": 0.75, "inference_status": "scored"},
                ],
            )
            result = run_audio_segments(inference_json_path=inference_json, top_k=2)
            segments = result["audio_summary"]["top_suspicious_audio_segments"]
            self.assertEqual(len(segments), 2)
            self.assertEqual(segments[0]["window_ids"], [1])
            self.assertEqual(segments[1]["window_ids"], [2])
            self.assertEqual(result["audio_summary"]["suspicious_segment_count"], 3)

    def test_score_summary_uses_population_variance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            inference_json = Path(temp_dir) / "audio_inference_result.json"
            _write_inference_json(
                inference_json,
                windows=[
                    {"window_id": 0, "start": 0.0, "end": 4.0, "duration": 4.0, "speech_overlap_sec": 1.0, "speech_coverage_ratio": 0.25, "has_speech": True, "audio_fake_score_raw": 1.0, "audio_real_score_raw": -1.0, "audio_fake_prob_like": 0.2, "inference_status": "scored"},
                    {"window_id": 1, "start": 4.0, "end": 8.0, "duration": 4.0, "speech_overlap_sec": 1.0, "speech_coverage_ratio": 0.25, "has_speech": True, "audio_fake_score_raw": 3.0, "audio_real_score_raw": -1.0, "audio_fake_prob_like": 0.6, "inference_status": "scored"},
                ],
            )
            result = run_audio_segments(inference_json_path=inference_json, suspicious_threshold=0.5)
            summary = result["audio_summary"]["score_summary"]
            self.assertAlmostEqual(summary["audio_fake_prob_like_mean"], 0.4)
            self.assertAlmostEqual(summary["audio_fake_prob_like_max"], 0.6)
            self.assertAlmostEqual(summary["audio_fake_prob_like_variance"], 0.04)
            self.assertAlmostEqual(summary["audio_fake_score_raw_mean"], 2.0)
            self.assertAlmostEqual(summary["audio_fake_score_raw_max"], 3.0)
            self.assertAlmostEqual(summary["audio_fake_score_raw_variance"], 1.0)

    def test_no_scored_windows_sets_null_summary_and_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            inference_json = Path(temp_dir) / "audio_inference_result.json"
            _write_inference_json(
                inference_json,
                quality_flags=["ENERGY_FALLBACK_VAD_USED"],
                windows=[
                    {"window_id": 0, "start": 0.0, "end": 4.0, "duration": 4.0, "speech_overlap_sec": 0.0, "speech_coverage_ratio": 0.0, "has_speech": False, "audio_fake_score_raw": None, "audio_real_score_raw": None, "audio_fake_prob_like": None, "inference_status": "skipped_no_speech"},
                    {"window_id": 1, "start": 4.0, "end": 8.0, "duration": 4.0, "speech_overlap_sec": 0.0, "speech_coverage_ratio": 0.0, "has_speech": False, "audio_fake_score_raw": None, "audio_real_score_raw": None, "audio_fake_prob_like": None, "inference_status": "failed_model_error"},
                ],
            )
            result = run_audio_segments(inference_json_path=inference_json)
            summary = result["audio_summary"]["score_summary"]
            self.assertIsNone(summary["audio_fake_prob_like_mean"])
            self.assertIsNone(summary["audio_fake_prob_like_variance"])
            self.assertIn("NO_SCORED_WINDOWS_FOR_SUMMARY", result["audio_summary"]["quality_flags"])

    def test_no_windows_sets_null_summary_and_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            inference_json = Path(temp_dir) / "audio_inference_result.json"
            _write_inference_json(inference_json, windows=[])
            result = run_audio_segments(inference_json_path=inference_json)
            summary = result["audio_summary"]["score_summary"]
            self.assertIsNone(summary["audio_fake_score_raw_mean"])
            self.assertIn("NO_WINDOWS_FOR_SUMMARY", result["audio_summary"]["quality_flags"])

    def test_skipped_and_failed_windows_are_excluded_from_segment_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            inference_json = Path(temp_dir) / "audio_inference_result.json"
            _write_inference_json(
                inference_json,
                windows=[
                    {"window_id": 0, "start": 0.0, "end": 4.0, "duration": 4.0, "speech_overlap_sec": 1.0, "speech_coverage_ratio": 0.25, "has_speech": True, "audio_fake_score_raw": 1.0, "audio_real_score_raw": -1.0, "audio_fake_prob_like": 0.9, "inference_status": "scored"},
                    {"window_id": 1, "start": 2.0, "end": 6.0, "duration": 4.0, "speech_overlap_sec": 1.0, "speech_coverage_ratio": 0.25, "has_speech": False, "audio_fake_score_raw": None, "audio_real_score_raw": None, "audio_fake_prob_like": None, "inference_status": "skipped_no_speech"},
                    {"window_id": 2, "start": 4.0, "end": 8.0, "duration": 4.0, "speech_overlap_sec": 1.0, "speech_coverage_ratio": 0.25, "has_speech": True, "audio_fake_score_raw": None, "audio_real_score_raw": None, "audio_fake_prob_like": None, "inference_status": "failed_model_error"},
                ],
            )
            result = run_audio_segments(inference_json_path=inference_json)
            segment = result["audio_summary"]["top_suspicious_audio_segments"][0]
            self.assertEqual(segment["window_ids"], [0])
            self.assertEqual(segment["window_count"], 1)

    def test_limits_and_quality_flags_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            inference_json = Path(temp_dir) / "audio_inference_result.json"
            _write_inference_json(
                inference_json,
                quality_flags=["ENERGY_FALLBACK_VAD_USED", "INFERENCE_FAILED"],
                limits={"unsupported_reason": None, "low_evidence_reason": "short_speech_region"},
                windows=[],
            )
            result = run_audio_segments(inference_json_path=inference_json)
            self.assertEqual(result["limits"]["low_evidence_reason"], "short_speech_region")
            self.assertIn("ENERGY_FALLBACK_VAD_USED", result["audio_summary"]["quality_flags"])
            self.assertIn("INFERENCE_FAILED", result["audio_summary"]["quality_flags"])

    def test_inference_is_not_rerun_and_checkpoint_not_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            inference_json = Path(temp_dir) / "audio_inference_result.json"
            _write_inference_json(
                inference_json,
                windows=[
                    {"window_id": 0, "start": 0.0, "end": 4.0, "duration": 4.0, "speech_overlap_sec": 1.0, "speech_coverage_ratio": 0.25, "has_speech": True, "audio_fake_score_raw": 2.0, "audio_real_score_raw": -1.0, "audio_fake_prob_like": 0.8, "inference_status": "scored"},
                ],
            )
            with patch(
                "services.ai.audio_pipeline.audio_inference.run_audio_inference",
                side_effect=AssertionError("stage 4 should not rerun"),
            ):
                result = run_audio_segments(inference_json_path=inference_json)
            self.assertEqual(result["audio_summary"]["scored_window_count"], 1)

    def test_final_fusion_fields_are_not_created(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            inference_json = Path(temp_dir) / "audio_inference_result.json"
            _write_inference_json(
                inference_json,
                windows=[
                    {"window_id": 0, "start": 0.0, "end": 4.0, "duration": 4.0, "speech_overlap_sec": 1.0, "speech_coverage_ratio": 0.25, "has_speech": True, "audio_fake_score_raw": 2.0, "audio_real_score_raw": -1.0, "audio_fake_prob_like": 0.8, "inference_status": "scored"},
                ],
            )
            result = run_audio_segments(inference_json_path=inference_json)
            self.assertNotIn("overall", result)
            self.assertNotIn("video", result)
            self.assertNotIn("audio", result)

    def test_invalid_count_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audio_inference_result.json"
            payload = {
                "audio_inference": {
                    "input_wav_path": "/tmp/sample.wav",
                    "source_windows_json": "/tmp/audio_windows_result.json",
                    "source_vad_json": "/tmp/audio_vad_result.json",
                    "model_name": "AntiDeepfake",
                    "window_sec": 4.0,
                    "hop_sec": 2.0,
                    "window_count": 2,
                    "scored_window_count": 1,
                    "skipped_window_count": 0,
                    "failed_window_count": 0,
                    "score_summary": {},
                    "windows": [],
                    "quality_flags": [],
                },
                "limits": {"unsupported_reason": None, "low_evidence_reason": None},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(AudioSegmentsError):
                run_audio_segments(inference_json_path=path)

    def test_unknown_inference_status_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            inference_json = Path(temp_dir) / "audio_inference_result.json"
            _write_inference_json(
                inference_json,
                windows=[
                    {
                        "window_id": 0,
                        "start": 0.0,
                        "end": 4.0,
                        "duration": 4.0,
                        "speech_overlap_sec": 1.0,
                        "speech_coverage_ratio": 0.25,
                        "has_speech": True,
                        "audio_fake_score_raw": 1.0,
                        "audio_real_score_raw": 0.0,
                        "audio_fake_prob_like": 0.8,
                        "inference_status": "unknown_status",
                    }
                ],
            )
            with self.assertRaises(AudioSegmentsError):
                run_audio_segments(inference_json_path=inference_json)

    def test_non_object_root_raises_audio_segments_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audio_inference_result.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(AudioSegmentsError):
                run_audio_segments(inference_json_path=path)

    def test_non_object_audio_inference_raises_audio_segments_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audio_inference_result.json"
            payload = {
                "audio_inference": [],
                "limits": {"unsupported_reason": None, "low_evidence_reason": None},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(AudioSegmentsError):
                run_audio_segments(inference_json_path=path)

    def test_non_list_windows_raises_audio_segments_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audio_inference_result.json"
            payload = {
                "audio_inference": {
                    "input_wav_path": "/tmp/sample.wav",
                    "source_windows_json": "/tmp/audio_windows_result.json",
                    "source_vad_json": "/tmp/audio_vad_result.json",
                    "model_name": "AntiDeepfake",
                    "window_sec": 4.0,
                    "hop_sec": 2.0,
                    "window_count": 1,
                    "scored_window_count": 0,
                    "skipped_window_count": 0,
                    "failed_window_count": 0,
                    "score_summary": {},
                    "windows": {},
                    "quality_flags": [],
                },
                "limits": {"unsupported_reason": None, "low_evidence_reason": None},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(AudioSegmentsError):
                run_audio_segments(inference_json_path=path)

    def test_null_windows_raises_audio_segments_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audio_inference_result.json"
            payload = {
                "audio_inference": {
                    "input_wav_path": "/tmp/sample.wav",
                    "source_windows_json": "/tmp/audio_windows_result.json",
                    "source_vad_json": "/tmp/audio_vad_result.json",
                    "model_name": "AntiDeepfake",
                    "window_sec": 4.0,
                    "hop_sec": 2.0,
                    "window_count": 0,
                    "scored_window_count": 0,
                    "skipped_window_count": 0,
                    "failed_window_count": 0,
                    "score_summary": {},
                    "windows": None,
                    "quality_flags": [],
                },
                "limits": {"unsupported_reason": None, "low_evidence_reason": None},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(AudioSegmentsError):
                run_audio_segments(inference_json_path=path)

    def test_null_quality_flags_raises_audio_segments_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audio_inference_result.json"
            payload = {
                "audio_inference": {
                    "input_wav_path": "/tmp/sample.wav",
                    "source_windows_json": "/tmp/audio_windows_result.json",
                    "source_vad_json": "/tmp/audio_vad_result.json",
                    "model_name": "AntiDeepfake",
                    "window_sec": 4.0,
                    "hop_sec": 2.0,
                    "window_count": 0,
                    "scored_window_count": 0,
                    "skipped_window_count": 0,
                    "failed_window_count": 0,
                    "score_summary": {},
                    "windows": [],
                    "quality_flags": None,
                },
                "limits": {"unsupported_reason": None, "low_evidence_reason": None},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(AudioSegmentsError):
                run_audio_segments(inference_json_path=path)

    def test_excessive_window_count_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audio_inference_result.json"
            windows = [
                {
                    "window_id": index,
                    "start": float(index),
                    "end": float(index + 1),
                    "duration": 1.0,
                    "speech_overlap_sec": 1.0,
                    "speech_coverage_ratio": 1.0,
                    "has_speech": True,
                    "audio_fake_score_raw": 1.0,
                    "audio_real_score_raw": 0.0,
                    "audio_fake_prob_like": 0.8,
                    "inference_status": "scored",
                }
                for index in range(513)
            ]
            _write_inference_json(path, windows=windows)
            with self.assertRaises(AudioSegmentsError):
                run_audio_segments(inference_json_path=path)


if __name__ == "__main__":
    unittest.main()
