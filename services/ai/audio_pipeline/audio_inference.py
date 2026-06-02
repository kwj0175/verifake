"""Window-level AntiDeepfake inference for stage 4 of the audio pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import wave
from pathlib import Path
from statistics import mean
from typing import Any

from .antideepfake import DEFAULT_CHECKPOINT_PATH, DEFAULT_HPARAMS_PATH, run_antideepfake_inference
from services.ai.common.runtime_probe import resolve_torch_device

DEFAULT_MIN_SPEECH_COVERAGE = 0.05
DEFAULT_MAX_WINDOWS = 512


class AudioInferenceError(RuntimeError):
    """stage-4 audio inference failure."""


def _fast_audio_once_enabled() -> bool:
    return os.getenv("VERIFAKE_FAST_AUDIO_ONCE", "1").strip().lower() not in {"0", "false", "no"}


def _format_model_error(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise AudioInferenceError(f"JSON 파일을 읽을 수 없습니다: {path}") from exc


def _resolve_inputs(*, windows_json_path: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any], Path]:
    resolved_windows_json = Path(windows_json_path).expanduser().resolve()
    payload = _load_json(resolved_windows_json)

    try:
        audio_windows = payload["audio_windows"]
        limits = dict(payload.get("limits", {}))
        wav_path = Path(audio_windows["input_wav_path"]).expanduser().resolve()
    except KeyError as exc:
        raise AudioInferenceError("windows JSON에 필요한 필드가 없습니다.") from exc

    return resolved_windows_json, audio_windows, limits, wav_path


def _is_under_directory(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _validate_input_wav_location(
    *, windows_json_path: Path, source_vad_json: str | None, wav_path: Path
) -> None:
    allowed_roots = [windows_json_path.parent.resolve()]
    if source_vad_json:
        allowed_roots.append(Path(source_vad_json).expanduser().resolve().parent)

    if any(_is_under_directory(wav_path, root) for root in allowed_roots):
        return

    roots = ", ".join(str(root) for root in allowed_roots)
    raise AudioInferenceError(
        "stage-4 inference는 stage-3가 생성한 artifact 경로만 허용합니다. "
        f"input_wav_path={wav_path}, allowed_roots=[{roots}]"
    )


def _validate_normalized_wav(wav_path: Path) -> dict[str, Any]:
    if not wav_path.exists():
        raise FileNotFoundError(f"정규화 WAV를 찾을 수 없습니다: {wav_path}")
    if wav_path.suffix.lower() != ".wav":
        raise AudioInferenceError("입력 파일은 WAV 형식이어야 합니다.")

    with wave.open(str(wav_path), "rb") as handle:
        sample_rate = handle.getframerate()
        channel_count = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        duration_sec = float(frame_count) / float(sample_rate) if sample_rate > 0 else 0.0

    if sample_rate != 16000:
        raise AudioInferenceError("stage-4 inference 입력 WAV는 16kHz여야 합니다.")
    if channel_count != 1:
        raise AudioInferenceError("stage-4 inference 입력 WAV는 mono여야 합니다.")
    if duration_sec <= 0.0:
        raise AudioInferenceError("stage-4 inference 입력 WAV duration이 0 이하입니다.")
    if sample_width != 2:
        raise AudioInferenceError("stage-4 inference 입력 WAV는 16-bit PCM이어야 합니다.")

    return {
        "sample_rate": sample_rate,
        "channel_count": channel_count,
        "duration_sec": duration_sec,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
    }


def _validate_windows_payload(
    audio_windows: dict[str, Any], *, total_duration_sec: float
) -> tuple[str | None, float, float, list[dict[str, Any]], list[str]]:
    required_keys = {
        "input_wav_path",
        "source_vad_json",
        "total_duration_sec",
        "window_sec",
        "hop_sec",
        "window_count",
        "windows",
        "quality_flags",
    }
    missing = sorted(required_keys - set(audio_windows))
    if missing:
        raise AudioInferenceError(
            f"windows JSON에 필요한 audio_windows 필드가 없습니다: {', '.join(missing)}"
        )

    source_vad_json = audio_windows.get("source_vad_json")
    window_sec = float(audio_windows["window_sec"])
    hop_sec = float(audio_windows["hop_sec"])
    windows = list(audio_windows["windows"])
    quality_flags = list(audio_windows.get("quality_flags", []))
    declared_duration = float(audio_windows["total_duration_sec"])
    declared_window_count = int(audio_windows["window_count"])

    if abs(declared_duration - total_duration_sec) > 0.05:
        raise AudioInferenceError("windows JSON의 total_duration_sec와 실제 wav duration이 일치하지 않습니다.")
    if declared_window_count != len(windows):
        raise AudioInferenceError("windows JSON의 window_count와 실제 windows 길이가 일치하지 않습니다.")
    if declared_window_count > DEFAULT_MAX_WINDOWS:
        raise AudioInferenceError(
            f"window 개수가 너무 많습니다: {declared_window_count} > {DEFAULT_MAX_WINDOWS}"
        )

    return source_vad_json, window_sec, hop_sec, windows, quality_flags


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _build_window_result(
    window: dict[str, Any],
    *,
    total_duration_sec: float,
    min_speech_coverage: float,
    skip_no_speech_windows: bool,
    unsupported_reason: str | None,
) -> dict[str, Any]:
    required_keys = {
        "window_id",
        "start",
        "end",
        "duration",
        "speech_overlap_sec",
        "speech_coverage_ratio",
        "has_speech",
    }
    missing = sorted(required_keys - set(window))
    if missing:
        raise AudioInferenceError(f"window row에 필요한 필드가 없습니다: {', '.join(missing)}")

    start = float(window["start"])
    end = float(window["end"])
    duration = float(window["duration"])
    speech_overlap_sec = window["speech_overlap_sec"]
    speech_coverage_ratio = window["speech_coverage_ratio"]
    has_speech = bool(window["has_speech"])

    result = {
        "window_id": int(window["window_id"]),
        "start": start,
        "end": end,
        "duration": duration,
        "speech_overlap_sec": speech_overlap_sec,
        "speech_coverage_ratio": speech_coverage_ratio,
        "has_speech": has_speech,
        "audio_fake_score_raw": None,
        "audio_real_score_raw": None,
        "audio_fake_prob_like": None,
        "inference_status": "pending",
        "model_error": None,
    }

    if unsupported_reason is not None:
        result["inference_status"] = "skipped_stage_unsupported"
        return result

    if start < 0.0 or end > total_duration_sec or duration <= 0.0 or end <= start:
        result["inference_status"] = "skipped_invalid_window"
        return result

    computed_duration = round(end - start, 6)
    if abs(computed_duration - duration) > 0.01:
        result["inference_status"] = "skipped_invalid_window"
        return result

    if skip_no_speech_windows and not has_speech:
        result["inference_status"] = "skipped_no_speech"
        return result

    if speech_coverage_ratio is not None:
        if not _is_number(speech_coverage_ratio):
            result["inference_status"] = "skipped_invalid_window"
            return result
        if float(speech_coverage_ratio) < min_speech_coverage:
            result["inference_status"] = "skipped_low_speech_coverage"
            return result

    if speech_overlap_sec is not None and not _is_number(speech_overlap_sec):
        result["inference_status"] = "skipped_invalid_window"
        return result

    result["inference_status"] = "scored"
    return result


def _ensure_unique_window_ids(windows: list[dict[str, Any]]) -> None:
    seen: set[int] = set()
    for window in windows:
        try:
            window_id = int(window["window_id"])
        except KeyError as exc:
            raise AudioInferenceError("window row에 window_id가 없습니다.") from exc
        except (TypeError, ValueError) as exc:
            raise AudioInferenceError("window_id는 정수여야 합니다.") from exc
        if window_id in seen:
            raise AudioInferenceError(f"중복된 window_id가 있습니다: {window_id}")
        seen.add(window_id)


def _extract_window_clip(
    *, source_wav_path: Path, window_row: dict[str, Any], output_path: Path
) -> Path:
    with wave.open(str(source_wav_path), "rb") as source_handle:
        sample_rate = source_handle.getframerate()
        channel_count = source_handle.getnchannels()
        sample_width = source_handle.getsampwidth()
        total_frames = source_handle.getnframes()

        start_frame = int(round(float(window_row["start"]) * sample_rate))
        end_frame = int(round(float(window_row["end"]) * sample_rate))
        frame_count = end_frame - start_frame

        if start_frame < 0 or end_frame > total_frames or frame_count <= 0:
            raise AudioInferenceError(
                f"window clip frame 범위가 유효하지 않습니다: window_id={window_row['window_id']}"
            )

        source_handle.setpos(start_frame)
        frames = source_handle.readframes(frame_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as output_handle:
        output_handle.setnchannels(channel_count)
        output_handle.setsampwidth(sample_width)
        output_handle.setframerate(sample_rate)
        output_handle.writeframes(frames)
    return output_path


def _default_json_output_for_windows(windows_json_path: Path) -> Path:
    return windows_json_path.parent / "audio_inference_result.json"


def _build_score_summary(windows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    scored_windows = [window for window in windows if window["inference_status"] == "scored"]
    fake_scores = [float(window["audio_fake_score_raw"]) for window in scored_windows]
    fake_probs = [float(window["audio_fake_prob_like"]) for window in scored_windows]

    return {
        "audio_fake_score_raw_mean": round(mean(fake_scores), 6) if fake_scores else None,
        "audio_fake_score_raw_max": round(max(fake_scores), 6) if fake_scores else None,
        "audio_fake_prob_like_mean": round(mean(fake_probs), 6) if fake_probs else None,
        "audio_fake_prob_like_max": round(max(fake_probs), 6) if fake_probs else None,
    }


def save_audio_inference_result(result: dict[str, Any], json_output_path: Path) -> Path:
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_output_path


def run_audio_inference(
    *,
    windows_json_path: str | Path,
    json_output_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    hparams_path: str | Path | None = None,
    python_executable: str = sys.executable,
    device: str | None = None,
    skip_no_speech_windows: bool = True,
    min_speech_coverage: float = DEFAULT_MIN_SPEECH_COVERAGE,
) -> dict[str, Any]:
    if not device:
        explicit_device = os.getenv("VERIFAKE_AI_DEVICE", "").strip()
        if explicit_device:
            device = explicit_device

    if not device:
        device = resolve_torch_device(default="cpu", cuda_index_env_var="VERIFAKE_CUDA_DEVICE")

    if min_speech_coverage < 0.0 or min_speech_coverage > 1.0:
        raise AudioInferenceError("min_speech_coverage는 0.0 이상 1.0 이하여야 합니다.")

    resolved_windows_json, audio_windows, limits, wav_path = _resolve_inputs(
        windows_json_path=windows_json_path
    )
    metadata = _validate_normalized_wav(wav_path)
    total_duration_sec = round(float(metadata["duration_sec"]), 6)

    source_vad_json, window_sec, hop_sec, windows, quality_flags = _validate_windows_payload(
        audio_windows,
        total_duration_sec=total_duration_sec,
    )
    _validate_input_wav_location(
        windows_json_path=resolved_windows_json,
        source_vad_json=source_vad_json,
        wav_path=wav_path,
    )
    _ensure_unique_window_ids(windows)

    unsupported_reason = limits.get("unsupported_reason")
    window_rows = [
        _build_window_result(
            window,
            total_duration_sec=total_duration_sec,
            min_speech_coverage=min_speech_coverage,
            skip_no_speech_windows=skip_no_speech_windows,
            unsupported_reason=unsupported_reason,
        )
        for window in windows
    ]

    if unsupported_reason is None:
        with tempfile.TemporaryDirectory(prefix="audio_inference_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            scored_rows = [row for row in window_rows if row["inference_status"] == "scored"]
            if scored_rows and _fast_audio_once_enabled():
                try:
                    inference_result = run_antideepfake_inference(
                        wav_path,
                        request_id="full-audio",
                        checkpoint_path=checkpoint_path or DEFAULT_CHECKPOINT_PATH,
                        hparams_path=hparams_path or DEFAULT_HPARAMS_PATH,
                        python_executable=python_executable,
                        device=device,
                    )
                except FileNotFoundError:
                    raise
                except Exception as exc:
                    model_error = _format_model_error(exc)
                    for window_row in scored_rows:
                        window_row["inference_status"] = "failed_model_error"
                        window_row["model_error"] = model_error
                else:
                    for window_row in scored_rows:
                        window_row["audio_fake_score_raw"] = inference_result.fake_logit
                        window_row["audio_real_score_raw"] = inference_result.real_logit
                        # NOTE: audio_fake_prob_like is a softmax-based probability-like score,
                        # not a calibrated probability.
                        window_row["audio_fake_prob_like"] = inference_result.fake_probability
            else:
                for window_row in scored_rows:
                    clip_path = temp_dir_path / (
                        f"window_{window_row['window_id']:04d}_"
                        f"{window_row['start']:.6f}_{window_row['end']:.6f}.wav"
                    )
                    _extract_window_clip(source_wav_path=wav_path, window_row=window_row, output_path=clip_path)
                    request_id = f"window-{window_row['window_id']:04d}"
                    try:
                        inference_result = run_antideepfake_inference(
                            clip_path,
                            request_id=request_id,
                            checkpoint_path=checkpoint_path or DEFAULT_CHECKPOINT_PATH,
                            hparams_path=hparams_path or DEFAULT_HPARAMS_PATH,
                            python_executable=python_executable,
                            device=device,
                        )
                    except FileNotFoundError:
                        raise
                    except Exception as exc:
                        window_row["inference_status"] = "failed_model_error"
                        window_row["model_error"] = _format_model_error(exc)
                        continue

                    window_row["audio_fake_score_raw"] = inference_result.fake_logit
                    window_row["audio_real_score_raw"] = inference_result.real_logit
                    # NOTE: audio_fake_prob_like is a softmax-based probability-like score,
                    # not a calibrated probability.
                    window_row["audio_fake_prob_like"] = inference_result.fake_probability

    scored_window_count = sum(1 for window in window_rows if window["inference_status"] == "scored")
    failed_window_count = sum(1 for window in window_rows if window["inference_status"] == "failed_model_error")
    skipped_window_count = sum(
        1 for window in window_rows if str(window["inference_status"]).startswith("skipped_")
    )
    all_quality_flags = list(quality_flags)
    if failed_window_count > 0:
        all_quality_flags = list(dict.fromkeys([*all_quality_flags, "INFERENCE_FAILED"]))
    model_errors = sorted(
        {
            str(window["model_error"])
            for window in window_rows
            if window.get("inference_status") == "failed_model_error" and window.get("model_error")
        }
    )

    result = {
        "audio_inference": {
            "input_wav_path": str(wav_path),
            "source_windows_json": str(resolved_windows_json),
            "source_vad_json": source_vad_json,
            "model_name": "AntiDeepfake",
            "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()) if checkpoint_path else None,
            "window_sec": window_sec,
            "hop_sec": hop_sec,
            "window_count": len(window_rows),
            "scored_window_count": scored_window_count,
            "skipped_window_count": skipped_window_count,
            "failed_window_count": failed_window_count,
            "score_summary": _build_score_summary(window_rows),
            "model_errors": model_errors,
            "windows": window_rows,
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
        else _default_json_output_for_windows(resolved_windows_json)
    )
    save_audio_inference_result(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="오디오 window inference만 단독 실행합니다.")
    parser.add_argument("--windows-json", required=True, help="3단계 window JSON 경로")
    parser.add_argument("--json-output", default=None, help="inference 결과 JSON 저장 경로")
    parser.add_argument("--checkpoint-path", default=None, help="AntiDeepfake checkpoint 경로")
    parser.add_argument("--hparams-path", default=None, help="AntiDeepfake hparams 경로")
    parser.add_argument("--python-executable", default=sys.executable, help="vendored runtime Python 경로")
    parser.add_argument("--device", default=None, help="예: cpu, cuda:0, mps")
    parser.add_argument(
        "--skip-no-speech-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="speech 없는 window를 skip할지 여부",
    )
    parser.add_argument(
        "--min-speech-coverage",
        type=float,
        default=DEFAULT_MIN_SPEECH_COVERAGE,
        help="이 값보다 speech coverage가 낮은 window는 skip합니다.",
    )
    args = parser.parse_args()

    result = run_audio_inference(
        windows_json_path=args.windows_json,
        json_output_path=args.json_output,
        checkpoint_path=args.checkpoint_path,
        hparams_path=args.hparams_path,
        python_executable=args.python_executable,
        device=args.device,
        skip_no_speech_windows=args.skip_no_speech_windows,
        min_speech_coverage=args.min_speech_coverage,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
