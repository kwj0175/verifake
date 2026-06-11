"""Stage 5 suspicious audio segment merge and summary generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pvariance
from typing import Any

DEFAULT_SUSPICIOUS_THRESHOLD = 0.5
DEFAULT_TOP_K = 5
DEFAULT_MAX_MERGE_GAP_SEC = 0.5
DEFAULT_MAX_WINDOWS = 512
ALLOWED_INFERENCE_STATUSES = {
    "scored",
    "skipped_no_speech",
    "skipped_low_speech_coverage",
    "skipped_invalid_window",
    "skipped_stage_unsupported",
    "failed_model_error",
}


class AudioSegmentsError(RuntimeError):
    """stage-5 suspicious segment merge / summary failure."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise AudioSegmentsError(f"JSON 파일을 읽을 수 없습니다: {path}") from exc


def _resolve_inputs(*, inference_json_path: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved_inference_json = Path(inference_json_path).expanduser().resolve()
    payload = _load_json(resolved_inference_json)
    if not isinstance(payload, dict):
        raise AudioSegmentsError("inference JSON의 최상위 구조는 object여야 합니다.")
    try:
        audio_inference = payload["audio_inference"]
        limits = dict(payload.get("limits", {}))
    except KeyError as exc:
        raise AudioSegmentsError("inference JSON에 필요한 필드가 없습니다.") from exc
    if not isinstance(audio_inference, dict):
        raise AudioSegmentsError("audio_inference 필드는 object여야 합니다.")
    return resolved_inference_json, audio_inference, limits


def _validate_inference_payload(
    audio_inference: dict[str, Any], *, suspicious_threshold: float, top_k: int, max_merge_gap_sec: float
) -> tuple[
    str,
    str,
    str | None,
    str | None,
    float,
    int,
    int,
    int,
    int,
    list[dict[str, Any]],
    list[str],
]:
    if not 0.0 <= suspicious_threshold <= 1.0:
        raise AudioSegmentsError("suspicious_threshold는 0.0 이상 1.0 이하여야 합니다.")
    if top_k <= 0:
        raise AudioSegmentsError("top_k는 1 이상이어야 합니다.")
    if max_merge_gap_sec < 0.0:
        raise AudioSegmentsError("max_merge_gap_sec는 0 이상이어야 합니다.")

    required_keys = {
        "input_wav_path",
        "source_windows_json",
        "source_vad_json",
        "model_name",
        "window_count",
        "scored_window_count",
        "skipped_window_count",
        "failed_window_count",
        "score_summary",
        "windows",
        "quality_flags",
    }
    missing = sorted(required_keys - set(audio_inference))
    if missing:
        raise AudioSegmentsError(
            f"inference JSON에 필요한 audio_inference 필드가 없습니다: {', '.join(missing)}"
        )

    input_wav_path = str(audio_inference["input_wav_path"])
    source_windows_json = str(audio_inference["source_windows_json"])
    source_vad_json = audio_inference.get("source_vad_json")
    if source_vad_json is not None:
        source_vad_json = str(source_vad_json)
    model_name = str(audio_inference["model_name"])
    window_count = int(audio_inference["window_count"])
    scored_window_count = int(audio_inference["scored_window_count"])
    skipped_window_count = int(audio_inference["skipped_window_count"])
    failed_window_count = int(audio_inference["failed_window_count"])
    raw_windows = audio_inference["windows"]
    raw_quality_flags = audio_inference.get("quality_flags", [])
    raw_score_summary = audio_inference["score_summary"]

    if not isinstance(raw_windows, list):
        raise AudioSegmentsError("audio_inference.windows 는 배열이어야 합니다.")
    if not isinstance(raw_quality_flags, list):
        raise AudioSegmentsError("audio_inference.quality_flags 는 배열이어야 합니다.")
    if not isinstance(raw_score_summary, dict):
        raise AudioSegmentsError("audio_inference.score_summary 는 object여야 합니다.")

    windows = list(raw_windows)
    quality_flags = list(raw_quality_flags)

    if window_count != len(windows):
        raise AudioSegmentsError("inference JSON의 window_count와 실제 windows 길이가 일치하지 않습니다.")
    if window_count > DEFAULT_MAX_WINDOWS:
        raise AudioSegmentsError(f"window 개수가 너무 많습니다: {window_count} > {DEFAULT_MAX_WINDOWS}")

    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise AudioSegmentsError(f"windows[{index}] 는 object여야 합니다.")
        status = window.get("inference_status")
        if status not in ALLOWED_INFERENCE_STATUSES:
            raise AudioSegmentsError(f"지원하지 않는 inference_status 입니다: {status}")

    actual_scored = sum(1 for window in windows if window.get("inference_status") == "scored")
    actual_skipped = sum(
        1 for window in windows if str(window.get("inference_status", "")).startswith("skipped_")
    )
    actual_failed = sum(1 for window in windows if window.get("inference_status") == "failed_model_error")

    if scored_window_count != actual_scored:
        raise AudioSegmentsError("inference JSON의 scored_window_count가 실제 상태와 일치하지 않습니다.")
    if skipped_window_count != actual_skipped:
        raise AudioSegmentsError("inference JSON의 skipped_window_count가 실제 상태와 일치하지 않습니다.")
    if failed_window_count != actual_failed:
        raise AudioSegmentsError("inference JSON의 failed_window_count가 실제 상태와 일치하지 않습니다.")

    return (
        input_wav_path,
        source_windows_json,
        source_vad_json,
        model_name,
        float(audio_inference.get("window_sec", 0.0)),
        window_count,
        scored_window_count,
        skipped_window_count,
        failed_window_count,
        windows,
        quality_flags,
    )


def _default_json_output_for_inference(inference_json_path: Path) -> Path:
    return inference_json_path.parent / "audio_segments_result.json"


def _collect_scored_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored_windows: list[dict[str, Any]] = []
    for window in windows:
        if window.get("inference_status") != "scored":
            continue
        required = {
            "window_id",
            "start",
            "end",
            "duration",
            "audio_fake_score_raw",
            "audio_fake_prob_like",
        }
        missing = sorted(required - set(window))
        if missing:
            raise AudioSegmentsError(f"scored window에 필요한 필드가 없습니다: {', '.join(missing)}")
        if window["audio_fake_score_raw"] is None or window["audio_fake_prob_like"] is None:
            raise AudioSegmentsError("scored window의 score 필드는 null일 수 없습니다.")
        scored_windows.append(window)
    return scored_windows


def _is_suspicious_window(window: dict[str, Any], *, suspicious_threshold: float) -> bool:
    return float(window["audio_fake_prob_like"]) >= suspicious_threshold


def _sorted_suspicious_windows(
    scored_windows: list[dict[str, Any]], *, suspicious_threshold: float
) -> list[dict[str, Any]]:
    suspicious_windows = [
        window for window in scored_windows if _is_suspicious_window(window, suspicious_threshold=suspicious_threshold)
    ]
    return sorted(suspicious_windows, key=lambda window: (float(window["start"]), int(window["window_id"])))


def _merge_suspicious_windows(
    suspicious_windows: list[dict[str, Any]], *, max_merge_gap_sec: float
) -> list[list[dict[str, Any]]]:
    if not suspicious_windows:
        return []

    grouped: list[list[dict[str, Any]]] = [[suspicious_windows[0]]]
    current_end = float(suspicious_windows[0]["end"])

    for window in suspicious_windows[1:]:
        next_start = float(window["start"])
        next_end = float(window["end"])
        if next_start <= current_end + max_merge_gap_sec + 1e-9:
            grouped[-1].append(window)
            current_end = max(current_end, next_end)
        else:
            grouped.append([window])
            current_end = next_end
    return grouped


def _population_variance(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(pvariance(values))


def _build_segment(grouped_windows: list[dict[str, Any]], *, segment_id: int) -> dict[str, Any]:
    fake_probs = [float(window["audio_fake_prob_like"]) for window in grouped_windows]
    fake_scores = [float(window["audio_fake_score_raw"]) for window in grouped_windows]
    start = float(grouped_windows[0]["start"])
    end = max(float(window["end"]) for window in grouped_windows)
    return {
        "segment_id": segment_id,
        "start": round(start, 6),
        "end": round(end, 6),
        "duration": round(end - start, 6),
        "window_ids": [int(window["window_id"]) for window in grouped_windows],
        "window_count": len(grouped_windows),
        "max_fake_prob_like": round(max(fake_probs), 6),
        "mean_fake_prob_like": round(mean(fake_probs), 6),
        "max_fake_score_raw": round(max(fake_scores), 6),
        "mean_fake_score_raw": round(mean(fake_scores), 6),
        "score_variance": round(_population_variance(fake_probs) or 0.0, 6),
        "reason": "single_window_above_threshold"
        if len(grouped_windows) == 1
        else "consecutive_windows_above_threshold",
    }


def _rank_segments(segments: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    ranked = sorted(
        segments,
        key=lambda segment: (
            -float(segment["max_fake_prob_like"]),
            -float(segment["mean_fake_prob_like"]),
            -float(segment["max_fake_score_raw"]),
            float(segment["start"]),
        ),
    )
    return ranked[:top_k]


def _build_audio_score_summary(
    *,
    scored_windows: list[dict[str, Any]],
    suspicious_windows: list[dict[str, Any]],
    suspicious_segments: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    if not scored_windows:
        return {
            "audio_fake_prob_like_mean": None,
            "audio_fake_prob_like_max": None,
            "audio_fake_prob_like_variance": None,
            "audio_fake_score_raw_mean": None,
            "audio_fake_score_raw_max": None,
            "audio_fake_score_raw_variance": None,
            "suspicious_window_count": len(suspicious_windows),
            "suspicious_segment_count": len(suspicious_segments),
        }

    fake_probs = [float(window["audio_fake_prob_like"]) for window in scored_windows]
    fake_scores = [float(window["audio_fake_score_raw"]) for window in scored_windows]
    fake_prob_variance = _population_variance(fake_probs)
    fake_score_variance = _population_variance(fake_scores)
    return {
        "audio_fake_prob_like_mean": round(mean(fake_probs), 6),
        "audio_fake_prob_like_max": round(max(fake_probs), 6),
        "audio_fake_prob_like_variance": round(fake_prob_variance, 6) if fake_prob_variance is not None else None,
        "audio_fake_score_raw_mean": round(mean(fake_scores), 6),
        "audio_fake_score_raw_max": round(max(fake_scores), 6),
        "audio_fake_score_raw_variance": round(fake_score_variance, 6) if fake_score_variance is not None else None,
        "suspicious_window_count": len(suspicious_windows),
        "suspicious_segment_count": len(suspicious_segments),
    }


def save_audio_segments_result(result: dict[str, Any], json_output_path: Path) -> Path:
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_output_path


def run_audio_segments(
    *,
    inference_json_path: str | Path,
    json_output_path: str | Path | None = None,
    suspicious_threshold: float = DEFAULT_SUSPICIOUS_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    max_merge_gap_sec: float = DEFAULT_MAX_MERGE_GAP_SEC,
) -> dict[str, Any]:
    resolved_inference_json, audio_inference, limits = _resolve_inputs(inference_json_path=inference_json_path)
    (
        input_wav_path,
        source_windows_json,
        source_vad_json,
        model_name,
        _window_sec,
        window_count,
        scored_window_count,
        skipped_window_count,
        failed_window_count,
        windows,
        quality_flags,
    ) = _validate_inference_payload(
        audio_inference,
        suspicious_threshold=suspicious_threshold,
        top_k=top_k,
        max_merge_gap_sec=max_merge_gap_sec,
    )

    all_quality_flags = list(quality_flags)
    if not windows:
        all_quality_flags = list(dict.fromkeys([*all_quality_flags, "NO_WINDOWS_FOR_SUMMARY"]))

    scored_windows = _collect_scored_windows(windows)
    if windows and not scored_windows:
        all_quality_flags = list(dict.fromkeys([*all_quality_flags, "NO_SCORED_WINDOWS_FOR_SUMMARY"]))

    suspicious_windows = _sorted_suspicious_windows(
        scored_windows,
        suspicious_threshold=suspicious_threshold,
    )
    grouped_windows = _merge_suspicious_windows(
        suspicious_windows,
        max_merge_gap_sec=max_merge_gap_sec,
    )
    all_segments = [
        _build_segment(group, segment_id=index)
        for index, group in enumerate(grouped_windows)
    ]
    top_segments = _rank_segments(all_segments, top_k=top_k)
    score_summary = _build_audio_score_summary(
        scored_windows=scored_windows,
        suspicious_windows=suspicious_windows,
        suspicious_segments=all_segments,
    )

    result = {
        "audio_summary": {
            "input_wav_path": input_wav_path,
            "source_inference_json": str(resolved_inference_json),
            "source_windows_json": source_windows_json,
            "source_vad_json": source_vad_json,
            "model_name": model_name,
            "suspicious_threshold": float(suspicious_threshold),
            "top_k": int(top_k),
            "max_merge_gap_sec": float(max_merge_gap_sec),
            "window_count": window_count,
            "scored_window_count": scored_window_count,
            "skipped_window_count": skipped_window_count,
            "failed_window_count": failed_window_count,
            "suspicious_window_count": len(suspicious_windows),
            "suspicious_segment_count": len(all_segments),
            "score_summary": score_summary,
            "top_suspicious_audio_segments": top_segments,
            "quality_flags": all_quality_flags,
        },
        "limits": {
            "unsupported_reason": limits.get("unsupported_reason"),
            "low_evidence_reason": limits.get("low_evidence_reason"),
        },
    }

    output_path = (
        Path(json_output_path).expanduser().resolve()
        if json_output_path
        else _default_json_output_for_inference(resolved_inference_json)
    )
    save_audio_segments_result(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="오디오 suspicious segment merge / summary만 단독 실행합니다.")
    parser.add_argument("--inference-json", required=True, help="4단계 inference JSON 경로")
    parser.add_argument("--json-output", default=None, help="segments 결과 JSON 저장 경로")
    parser.add_argument(
        "--suspicious-threshold",
        type=float,
        default=DEFAULT_SUSPICIOUS_THRESHOLD,
        help="audio_fake_prob_like >= threshold 인 scored window를 suspicious 로 봅니다.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="저장할 top suspicious segment 개수")
    parser.add_argument(
        "--max-merge-gap-sec",
        type=float,
        default=DEFAULT_MAX_MERGE_GAP_SEC,
        help="겹치거나 이 gap 이하인 suspicious window는 병합합니다.",
    )
    args = parser.parse_args()

    result = run_audio_segments(
        inference_json_path=args.inference_json,
        json_output_path=args.json_output,
        suspicious_threshold=args.suspicious_threshold,
        top_k=args.top_k,
        max_merge_gap_sec=args.max_merge_gap_sec,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
