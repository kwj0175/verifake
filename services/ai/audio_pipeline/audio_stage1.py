"""End-to-end stage1 audio pipeline orchestration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from .audio_inference import run_audio_inference
from .audio_preprocess import preprocess_audio
from .audio_segments import (
    DEFAULT_MAX_MERGE_GAP_SEC,
    DEFAULT_SUSPICIOUS_THRESHOLD,
    DEFAULT_TOP_K,
    run_audio_segments,
)
from .audio_vad import run_audio_vad
from .audio_windowing import DEFAULT_HOP_SEC, DEFAULT_WINDOW_SEC, run_audio_windowing
from .schemas import AudioAnalysisResult, EvidenceLevel, OriginalAudioMetadata, QualityFlag, SuspiciousAudioSegment


class AudioStage1Error(RuntimeError):
    """Stage1 orchestration failure."""


FINAL_RESULT_FILENAME = "audio_stage1_result.json"
QUALITY_FLAG_MAP: dict[str, QualityFlag] = {
    "LONG_LEADING_SILENCE": QualityFlag.LONG_LEADING_SILENCE,
    "LONG_TRAILING_SILENCE": QualityFlag.LONG_TRAILING_SILENCE,
    "HIGH_SILENCE_RATIO": QualityFlag.HIGH_SILENCE_RATIO,
    "LOW_BITRATE_SOURCE": QualityFlag.LOW_BITRATE_SOURCE,
    "LOW_SPEECH_RATIO": QualityFlag.LOW_SPEECH_RATIO,
    "NO_DETECTED_SPEECH": QualityFlag.NO_HUMAN_SPEECH,
    "TOO_SHORT": QualityFlag.TOO_SHORT,
    "CLIPPING_DETECTED": QualityFlag.CLIPPING_DETECTED,
}


def _default_json_output_path(output_dir: Path) -> Path:
    return output_dir / FINAL_RESULT_FILENAME


def _resolve_limits(*results: dict[str, Any]) -> dict[str, Any]:
    merged = {"unsupported_reason": None, "low_evidence_reason": None}
    for result in results:
        limits = result.get("limits", {})
        if limits.get("unsupported_reason") is not None:
            merged["unsupported_reason"] = limits["unsupported_reason"]
        if limits.get("low_evidence_reason") is not None:
            merged["low_evidence_reason"] = limits["low_evidence_reason"]
    return merged


def _map_original_metadata(preprocess_result: dict[str, Any]) -> OriginalAudioMetadata:
    audio_preprocess = preprocess_result.get("audio_preprocess", {})
    original = audio_preprocess.get("original", {})
    normalized = audio_preprocess.get("normalized", {})
    return OriginalAudioMetadata(
        codec=original.get("codec"),
        bitrate=original.get("bitrate"),
        duration_sec=float(original.get("duration_sec") or normalized.get("duration_sec") or 0.0),
        sample_rate_hz=int(original.get("sample_rate") or normalized.get("sample_rate") or 16000),
        channel_count=int(original.get("channel_count") or normalized.get("channel_count") or 1),
    )


def _map_quality_flags(*, limits: dict[str, Any], results: list[dict[str, Any]]) -> list[QualityFlag]:
    ordered_flags: list[QualityFlag] = []
    seen: set[QualityFlag] = set()
    if limits.get("unsupported_reason") == "no_detected_human_speech":
        seen.add(QualityFlag.NO_HUMAN_SPEECH)
        ordered_flags.append(QualityFlag.NO_HUMAN_SPEECH)
    for result in results:
        for payload in result.values():
            if not isinstance(payload, dict):
                continue
            for raw_flag in payload.get("quality_flags", []):
                mapped = QUALITY_FLAG_MAP.get(str(raw_flag))
                if mapped is not None and mapped not in seen:
                    seen.add(mapped)
                    ordered_flags.append(mapped)
    return ordered_flags


def _map_evidence_level(limits: dict[str, Any]) -> EvidenceLevel:
    if limits.get("unsupported_reason") is not None:
        return EvidenceLevel.UNSUPPORTED_CONTENT
    if limits.get("low_evidence_reason") is not None:
        return EvidenceLevel.LOW_EVIDENCE
    return EvidenceLevel.SUFFICIENT


def _apply_runtime_evidence_limits(
    *,
    limits: dict[str, Any],
    scored_window_count: int,
    failed_window_count: int,
) -> dict[str, Any]:
    resolved_limits = dict(limits)
    if resolved_limits.get("unsupported_reason") is not None:
        return resolved_limits
    if failed_window_count > 0 and resolved_limits.get("low_evidence_reason") is None:
        resolved_limits["low_evidence_reason"] = "stage4_inference_failed"
    if scored_window_count <= 0 and resolved_limits.get("low_evidence_reason") is None:
        resolved_limits["low_evidence_reason"] = "no_scored_audio_windows"
    return resolved_limits


def _scored_windows(inference_result: dict[str, Any]) -> list[dict[str, Any]]:
    windows = inference_result.get("audio_inference", {}).get("windows", [])
    return [window for window in windows if window.get("inference_status") == "scored"]


def _mean_or_none(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _score_values(windows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for window in windows:
        value = window.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _first_model_error(inference_result: dict[str, Any]) -> str | None:
    audio_inference = inference_result.get("audio_inference", {})
    model_errors = audio_inference.get("model_errors") or []
    for error in model_errors:
        if error:
            return str(error)
    for window in audio_inference.get("windows", []):
        if window.get("inference_status") == "failed_model_error" and window.get("model_error"):
            return str(window["model_error"])
    return None


def _compute_uncertainty(*, limits: dict[str, Any], score_summary: dict[str, Any], scored_window_count: int) -> float:
    if limits.get("unsupported_reason") is not None or scored_window_count <= 0:
        return 1.0
    variance = float(score_summary.get("audio_fake_prob_like_variance") or 0.0)
    scaled = min(1.0, max(0.0, variance * 4.0))
    if limits.get("low_evidence_reason") is not None:
        return max(0.6, scaled)
    return scaled


def _map_suspicious_segments(
    *,
    inference_result: dict[str, Any],
    segments_result: dict[str, Any],
) -> list[SuspiciousAudioSegment]:
    windows_by_id = {
        int(window["window_id"]): window
        for window in _scored_windows(inference_result)
        if "window_id" in window
    }
    mapped_segments: list[SuspiciousAudioSegment] = []
    raw_segments = segments_result.get("audio_summary", {}).get("top_suspicious_audio_segments", [])
    for rank, raw_segment in enumerate(raw_segments, start=1):
        window_ids = [int(window_id) for window_id in raw_segment.get("window_ids", [])]
        candidate_windows = [windows_by_id[window_id] for window_id in window_ids if window_id in windows_by_id]
        if candidate_windows:
            anchor_window = max(candidate_windows, key=lambda window: float(window.get("audio_fake_prob_like") or 0.0))
            fake_score_raw = float(anchor_window.get("audio_fake_score_raw") or 0.0)
            real_score_raw = float(anchor_window.get("audio_real_score_raw") or 0.0)
            fake_prob_like = float(anchor_window.get("audio_fake_prob_like") or 0.0)
        else:
            fake_score_raw = float(raw_segment.get("max_fake_score_raw") or 0.0)
            real_score_raw = 0.0
            fake_prob_like = float(raw_segment.get("max_fake_prob_like") or 0.0)
        mapped_segments.append(
            SuspiciousAudioSegment(
                start_sec=float(raw_segment["start"]),
                end_sec=float(raw_segment["end"]),
                fake_score_raw=fake_score_raw,
                real_score_raw=real_score_raw,
                fake_prob_like=fake_prob_like,
                rank=rank,
            )
        )
    return mapped_segments


def save_audio_stage1_result(result: dict[str, Any], json_output_path: Path) -> Path:
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_output_path


def run_audio_stage1(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    request_id: str | None = None,
    json_output_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    hparams_path: str | Path | None = None,
    python_executable: str = sys.executable,
    device: str | None = None,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    suspicious_threshold: float = DEFAULT_SUSPICIOUS_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    max_merge_gap_sec: float = DEFAULT_MAX_MERGE_GAP_SEC,
) -> dict[str, Any]:
    resolved_input_path = Path(input_path).expanduser().resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_request_id = request_id or resolved_input_path.stem

    preprocess_json_path = resolved_output_dir / "audio_preprocess_result.json"
    vad_json_path = resolved_output_dir / "audio_vad_result.json"
    windows_json_path = resolved_output_dir / "audio_windows_result.json"
    inference_json_path = resolved_output_dir / "audio_inference_result.json"
    segments_json_path = resolved_output_dir / "audio_segments_result.json"

    preprocess_result = preprocess_audio(
        resolved_input_path,
        resolved_output_dir,
        json_output_path=preprocess_json_path,
    )
    vad_result = run_audio_vad(
        preprocess_json_path=preprocess_json_path,
        json_output_path=vad_json_path,
    )
    windowing_result = run_audio_windowing(
        vad_json_path=vad_json_path,
        json_output_path=windows_json_path,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )
    inference_result = run_audio_inference(
        windows_json_path=windows_json_path,
        json_output_path=inference_json_path,
        checkpoint_path=checkpoint_path,
        hparams_path=hparams_path,
        python_executable=python_executable,
        device=device,
    )
    segments_result = run_audio_segments(
        inference_json_path=inference_json_path,
        json_output_path=segments_json_path,
        suspicious_threshold=suspicious_threshold,
        top_k=top_k,
        max_merge_gap_sec=max_merge_gap_sec,
    )

    limits = _resolve_limits(preprocess_result, vad_result, windowing_result, inference_result, segments_result)
    stats = vad_result.get("audio_vad", {}).get("speech_stats", {})
    scored_windows = _scored_windows(inference_result)
    audio_inference = inference_result.get("audio_inference", {})
    scored_window_count = len(scored_windows)
    failed_window_count = int(audio_inference.get("failed_window_count") or 0)
    skipped_window_count = int(audio_inference.get("skipped_window_count") or 0)
    limits = _apply_runtime_evidence_limits(
        limits=limits,
        scored_window_count=scored_window_count,
        failed_window_count=failed_window_count,
    )
    evidence_level = _map_evidence_level(limits)
    fake_scores = _score_values(scored_windows, "audio_fake_score_raw")
    real_scores = _score_values(scored_windows, "audio_real_score_raw")
    fake_probs = _score_values(scored_windows, "audio_fake_prob_like")
    score_summary = segments_result.get("audio_summary", {}).get("score_summary", {})

    final_result = AudioAnalysisResult(
        request_id=resolved_request_id,
        file_path=str(resolved_input_path),
        original_metadata=_map_original_metadata(preprocess_result),
        audio_fake_score_raw=_mean_or_none(fake_scores),
        audio_real_score_raw=_mean_or_none(real_scores),
        audio_fake_prob_like=_mean_or_none(fake_probs),
        audio_uncertainty=_compute_uncertainty(
            limits=limits,
            score_summary=score_summary,
            scored_window_count=scored_window_count,
        ),
        scored_window_count=scored_window_count,
        failed_window_count=failed_window_count,
        skipped_window_count=skipped_window_count,
        audio_model_error=_first_model_error(inference_result),
        human_speech_detected=bool(vad_result.get("audio_vad", {}).get("human_speech_detected", False)),
        evidence_level=evidence_level,
        speech_duration_sec=float(stats.get("speech_duration_sec") or 0.0),
        speech_ratio=float(stats.get("speech_ratio") or 0.0),
        silence_ratio=float(stats.get("silence_ratio") or 0.0),
        pause_count=int(stats.get("pause_count") or 0),
        leading_silence_sec=float(stats.get("leading_silence") or 0.0),
        trailing_silence_sec=float(stats.get("trailing_silence") or 0.0),
        quality_flags=_map_quality_flags(
            limits=limits,
            results=[preprocess_result, vad_result, windowing_result, inference_result, segments_result],
        ),
        top_suspicious_audio_segments=_map_suspicious_segments(
            inference_result=inference_result,
            segments_result=segments_result,
        ),
    )
    final_payload = final_result.model_dump(mode="json")
    output_path = (
        Path(json_output_path).expanduser().resolve()
        if json_output_path
        else _default_json_output_path(resolved_output_dir)
    )
    save_audio_stage1_result(final_payload, output_path)
    return final_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="오디오 stage1 전체 파이프라인을 순차 실행합니다.")
    parser.add_argument("--input", required=True, help="입력 오디오 또는 영상 파일 경로")
    parser.add_argument("--output-dir", required=True, help="stage1 artifact 출력 디렉터리")
    parser.add_argument("--request-id", default=None, help="최종 결과 request_id")
    parser.add_argument("--json-output", default=None, help="최종 stage1 결과 JSON 저장 경로")
    parser.add_argument("--checkpoint-path", default=None, help="AntiDeepfake checkpoint 경로")
    parser.add_argument("--hparams-path", default=None, help="AntiDeepfake hparams 경로")
    parser.add_argument("--python-executable", default=sys.executable, help="vendored runtime Python 경로")
    parser.add_argument("--device", default=None, help="예: cpu, cuda:0, mps")
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC, help="window 길이(초)")
    parser.add_argument("--hop-sec", type=float, default=DEFAULT_HOP_SEC, help="window hop 길이(초)")
    parser.add_argument(
        "--suspicious-threshold",
        type=float,
        default=DEFAULT_SUSPICIOUS_THRESHOLD,
        help="audio_fake_prob_like suspicious threshold",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="저장할 top suspicious segment 개수")
    parser.add_argument(
        "--max-merge-gap-sec",
        type=float,
        default=DEFAULT_MAX_MERGE_GAP_SEC,
        help="suspicious window merge gap",
    )
    args = parser.parse_args()

    result = run_audio_stage1(
        input_path=args.input,
        output_dir=args.output_dir,
        request_id=args.request_id,
        json_output_path=args.json_output,
        checkpoint_path=args.checkpoint_path,
        hparams_path=args.hparams_path,
        python_executable=args.python_executable,
        device=args.device,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        suspicious_threshold=args.suspicious_threshold,
        top_k=args.top_k,
        max_merge_gap_sec=args.max_merge_gap_sec,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
