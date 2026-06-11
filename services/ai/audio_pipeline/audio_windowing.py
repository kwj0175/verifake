"""정규화된 WAV 전체 타임라인 기준으로 overlapping window metadata를 생성한다."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any


DEFAULT_WINDOW_SEC = 4.0
DEFAULT_HOP_SEC = 2.0
MIN_WINDOW_DURATION_SEC = 1.0
MIN_WINDOW_SPEECH_COVERAGE = 0.05


class AudioWindowingError(RuntimeError):
    """windowing 단계 실패를 나타내는 기본 예외."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise AudioWindowingError(f"JSON 파일을 읽을 수 없습니다: {path}") from exc


def _resolve_inputs(
    *,
    vad_json_path: str | Path | None,
    input_wav_path: str | Path | None,
) -> tuple[Path, float | None, list[dict[str, Any]], list[str], dict[str, Any], str | None]:
    if vad_json_path is None and input_wav_path is None:
        raise AudioWindowingError("--vad-json 또는 --input-wav 중 하나는 반드시 필요합니다.")
    if vad_json_path is not None and input_wav_path is not None:
        raise AudioWindowingError("--vad-json 과 --input-wav 는 동시에 사용할 수 없습니다.")

    if vad_json_path is not None:
        vad_json = Path(vad_json_path).expanduser().resolve()
        payload = _load_json(vad_json)
        try:
            audio_vad = payload["audio_vad"]
            wav_path = Path(audio_vad["input_wav_path"]).expanduser().resolve()
            total_duration_sec = float(audio_vad["total_duration_sec"])
            speech_segments = list(audio_vad.get("speech_segments", []))
            quality_flags = list(audio_vad.get("quality_flags", []))
            limits = dict(payload.get("limits", {}))
        except KeyError as exc:
            raise AudioWindowingError("VAD JSON에 필요한 필드가 없습니다.") from exc
        return wav_path, total_duration_sec, speech_segments, quality_flags, limits, str(vad_json)

    wav_path = Path(input_wav_path).expanduser().resolve()
    return wav_path, None, [], [], {"unsupported_reason": None, "low_evidence_reason": None}, None


def _validate_normalized_wav(wav_path: Path) -> dict[str, Any]:
    if not wav_path.exists():
        raise FileNotFoundError(f"정규화 WAV를 찾을 수 없습니다: {wav_path}")
    if wav_path.suffix.lower() != ".wav":
        raise AudioWindowingError("입력 파일은 WAV 형식이어야 합니다.")

    with wave.open(str(wav_path), "rb") as handle:
        sample_rate = handle.getframerate()
        channel_count = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        duration_sec = float(frame_count) / float(sample_rate) if sample_rate > 0 else 0.0

    if sample_rate != 16000:
        raise AudioWindowingError("stage-3 windowing 입력 WAV는 16kHz여야 합니다.")
    if channel_count != 1:
        raise AudioWindowingError("stage-3 windowing 입력 WAV는 mono여야 합니다.")
    if duration_sec <= 0.0:
        raise AudioWindowingError("stage-3 windowing 입력 WAV duration이 0 이하입니다.")
    if sample_width != 2:
        raise AudioWindowingError("stage-3 windowing 입력 WAV는 16-bit PCM이어야 합니다.")

    return {
        "sample_rate": sample_rate,
        "channel_count": channel_count,
        "duration_sec": duration_sec,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
    }


def _normalize_segment(segment: dict[str, Any]) -> tuple[float, float]:
    start = float(segment["start"])
    end = float(segment["end"])
    if end <= start:
        raise AudioWindowingError("speech segment end가 start보다 커야 합니다.")
    return start, end


def _merge_segments(segments: list[dict[str, Any]]) -> list[tuple[float, float]]:
    if not segments:
        return []
    normalized = sorted(_normalize_segment(segment) for segment in segments)
    merged: list[list[float]] = [[normalized[0][0], normalized[0][1]]]
    for start, end in normalized[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _generate_windows(
    total_duration_sec: float,
    *,
    window_sec: float,
    hop_sec: float,
    min_window_duration_sec: float,
) -> tuple[list[dict[str, float]], list[str]]:
    if window_sec <= 0.0:
        raise AudioWindowingError("window_sec는 0보다 커야 합니다.")
    if hop_sec <= 0.0:
        raise AudioWindowingError("hop_sec는 0보다 커야 합니다.")
    if min_window_duration_sec <= 0.0:
        raise AudioWindowingError("min_window_duration_sec는 0보다 커야 합니다.")

    windows: list[dict[str, float]] = []
    flags: list[str] = []
    start = 0.0
    index = 0

    while start < total_duration_sec:
        end = min(total_duration_sec, start + window_sec)
        duration = end - start
        if duration < min_window_duration_sec:
            flags.append("SHORT_FINAL_WINDOW_SKIPPED")
            break

        windows.append(
            {
                "window_id": index,
                "start": round(start, 6),
                "end": round(end, 6),
                "duration": round(duration, 6),
            }
        )

        if end >= total_duration_sec:
            break

        start = round(start + hop_sec, 6)
        index += 1

    if not windows:
        flags.append("NO_WINDOWS_CREATED")
    return windows, list(dict.fromkeys(flags))


def _compute_speech_overlap(
    *,
    window_start: float,
    window_end: float,
    window_duration: float,
    merged_segments: list[tuple[float, float]] | None,
) -> tuple[float | None, float | None, bool]:
    if merged_segments is None:
        return None, None, False

    overlap_sec = 0.0
    for seg_start, seg_end in merged_segments:
        if seg_end <= window_start or seg_start >= window_end:
            continue
        overlap_sec += max(0.0, min(window_end, seg_end) - max(window_start, seg_start))

    overlap_sec = round(overlap_sec, 6)
    coverage_ratio = round(overlap_sec / window_duration, 6) if window_duration > 0 else 0.0
    has_speech = coverage_ratio >= MIN_WINDOW_SPEECH_COVERAGE
    return overlap_sec, coverage_ratio, has_speech


def _default_json_output_for_vad(vad_json_path: Path) -> Path:
    return vad_json_path.parent / "audio_windows_result.json"


def _default_json_output_for_wav(wav_path: Path) -> Path:
    return wav_path.parent / "audio_windows_result.json"


def save_audio_windowing_result(result: dict[str, Any], json_output_path: Path) -> Path:
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_output_path


def run_audio_windowing(
    *,
    vad_json_path: str | Path | None = None,
    input_wav_path: str | Path | None = None,
    json_output_path: str | Path | None = None,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    min_window_duration_sec: float = MIN_WINDOW_DURATION_SEC,
) -> dict[str, Any]:
    wav_path, total_duration_from_vad, speech_segments, quality_flags, limits, source_vad_json = _resolve_inputs(
        vad_json_path=vad_json_path,
        input_wav_path=input_wav_path,
    )
    metadata = _validate_normalized_wav(wav_path)
    total_duration_sec = round(float(metadata["duration_sec"]), 6)

    if total_duration_from_vad is not None and abs(total_duration_from_vad - total_duration_sec) > 0.05:
        raise AudioWindowingError("VAD JSON의 total_duration_sec와 실제 wav duration이 일치하지 않습니다.")

    windows, windowing_flags = _generate_windows(
        total_duration_sec,
        window_sec=window_sec,
        hop_sec=hop_sec,
        min_window_duration_sec=min_window_duration_sec,
    )
    merged_segments = _merge_segments(speech_segments) if source_vad_json is not None else None

    window_rows: list[dict[str, Any]] = []
    for window in windows:
        overlap_sec, coverage_ratio, has_speech = _compute_speech_overlap(
            window_start=float(window["start"]),
            window_end=float(window["end"]),
            window_duration=float(window["duration"]),
            merged_segments=merged_segments,
        )
        window_rows.append(
            {
                **window,
                "speech_overlap_sec": overlap_sec,
                "speech_coverage_ratio": coverage_ratio,
                "has_speech": has_speech,
            }
        )

    all_quality_flags = list(dict.fromkeys([*quality_flags, *windowing_flags]))
    result = {
        "audio_windows": {
            "input_wav_path": str(wav_path),
            "source_vad_json": source_vad_json,
            "total_duration_sec": total_duration_sec,
            "window_sec": float(window_sec),
            "hop_sec": float(hop_sec),
            "window_count": len(window_rows),
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
        else _default_json_output_for_vad(Path(vad_json_path).expanduser().resolve())
        if vad_json_path
        else _default_json_output_for_wav(wav_path)
    )
    save_audio_windowing_result(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="오디오 sliding window metadata만 단독 실행합니다.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--vad-json", help="2단계 VAD JSON 경로")
    group.add_argument("--input-wav", help="normalized 16kHz mono WAV 경로")
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC, help="window 길이(초)")
    parser.add_argument("--hop-sec", type=float, default=DEFAULT_HOP_SEC, help="window hop 길이(초)")
    parser.add_argument("--json-output", default=None, help="windowing 결과 JSON 저장 경로")
    args = parser.parse_args()

    result = run_audio_windowing(
        vad_json_path=args.vad_json,
        input_wav_path=args.input_wav,
        json_output_path=args.json_output,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
