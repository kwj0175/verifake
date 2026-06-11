"""정규화된 WAV 전체 타임라인에 대해 speech / non-speech 태깅을 수행한다."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np


PAUSE_THRESHOLD_SEC = 0.5
MIN_SPEECH_DURATION_SEC = 1.0
MIN_SPEECH_RATIO = 0.05
LOW_SPEECH_RATIO_THRESHOLD = 0.05
HIGH_SILENCE_RATIO_THRESHOLD = 0.95
LONG_SILENCE_THRESHOLD_SEC = 3.0
FRAME_MS = 30
HOP_MS = 10
ENERGY_THRESHOLD_FLOOR = 0.01
ENERGY_NOISE_MULTIPLIER = 3.0
MIN_SPEECH_SEGMENT_SEC = 0.2
MIN_SILENCE_GAP_SEC = 0.3
BREATH_GAP_MIN_SEC = 0.3
BREATH_GAP_MAX_SEC = 2.0


class AudioVadError(RuntimeError):
    """VAD 단계 실패를 나타내는 기본 예외."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise AudioVadError(f"JSON 파일을 읽을 수 없습니다: {path}") from exc


def _resolve_inputs(
    *,
    preprocess_json_path: str | Path | None,
    input_wav_path: str | Path | None,
) -> tuple[Path, list[str], dict[str, Any], str | None]:
    if preprocess_json_path is None and input_wav_path is None:
        raise AudioVadError("--preprocess-json 또는 --input-wav 중 하나는 반드시 필요합니다.")
    if preprocess_json_path is not None and input_wav_path is not None:
        raise AudioVadError("--preprocess-json 과 --input-wav 는 동시에 사용할 수 없습니다.")

    if preprocess_json_path is not None:
        preprocess_path = Path(preprocess_json_path).expanduser().resolve()
        payload = _load_json(preprocess_path)
        try:
            wav_path = payload["audio_preprocess"]["normalized_wav_path"]
        except KeyError as exc:
            raise AudioVadError(
                "preprocess JSON에 audio_preprocess.normalized_wav_path 가 없습니다."
            ) from exc
        quality_flags = list(payload.get("audio_preprocess", {}).get("quality_flags", []))
        limits = dict(payload.get("limits", {}))
        return Path(wav_path).expanduser().resolve(), quality_flags, limits, str(preprocess_path)

    if input_wav_path is None:
        raise AudioVadError("--input-wav 경로가 비어 있습니다.")
    wav_path = Path(input_wav_path).expanduser().resolve()
    return wav_path, [], {"unsupported_reason": None, "low_evidence_reason": None}, None


def _validate_normalized_wav(wav_path: Path) -> dict[str, Any]:
    if not wav_path.exists():
        raise FileNotFoundError(f"정규화 WAV를 찾을 수 없습니다: {wav_path}")
    if wav_path.suffix.lower() != ".wav":
        raise AudioVadError("입력 파일은 WAV 형식이어야 합니다.")

    with wave.open(str(wav_path), "rb") as handle:
        sample_rate = handle.getframerate()
        channel_count = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        duration_sec = float(frame_count) / float(sample_rate) if sample_rate > 0 else 0.0

    if sample_rate != 16000:
        raise AudioVadError("stage-2 VAD 입력 WAV는 16kHz여야 합니다.")
    if channel_count != 1:
        raise AudioVadError("stage-2 VAD 입력 WAV는 mono여야 합니다.")
    if duration_sec <= 0.0:
        raise AudioVadError("stage-2 VAD 입력 WAV duration이 0 이하입니다.")
    if sample_width != 2:
        raise AudioVadError("stage-2 VAD 입력 WAV는 16-bit PCM이어야 합니다.")

    return {
        "sample_rate": sample_rate,
        "channel_count": channel_count,
        "duration_sec": duration_sec,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
    }


def _read_waveform(wav_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as handle:
        sample_rate = handle.getframerate()
        raw_frames = handle.readframes(handle.getnframes())
    waveform = np.frombuffer(raw_frames, dtype="<i2").astype(np.float32) / 32768.0
    return waveform, sample_rate


def _silero_vad_segments(wav_path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        import importlib

        silero_vad = importlib.import_module("silero_vad")
    except ImportError as exc:
        raise AudioVadError("silero_vad_not_available") from exc

    read_audio = silero_vad.read_audio
    load_silero_vad = silero_vad.load_silero_vad
    get_speech_timestamps = silero_vad.get_speech_timestamps

    audio = read_audio(str(wav_path), sampling_rate=16000)
    model = load_silero_vad()
    timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=16000,
        return_seconds=True,
    )
    segments: list[dict[str, Any]] = []
    for item in timestamps:
        start = float(item["start"])
        end = float(item["end"])
        duration = max(0.0, end - start)
        segments.append(
            {
                "start": round(start, 6),
                "end": round(end, 6),
                "duration": round(duration, 6),
                "confidence": None,
                "method": "silero",
            }
        )
    return segments, "silero"


def _segments_from_mask(mask: np.ndarray, hop_sec: float, frame_sec: float, total_duration_sec: float) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    in_segment = False
    segment_start = 0.0
    for index, is_speech in enumerate(mask.tolist()):
        if is_speech and not in_segment:
            in_segment = True
            segment_start = index * hop_sec
        elif not is_speech and in_segment:
            in_segment = False
            segment_end = min(total_duration_sec, index * hop_sec + frame_sec)
            segments.append((segment_start, segment_end))
    if in_segment:
        segments.append((segment_start, total_duration_sec))
    return segments


def _merge_close_segments(segments: list[tuple[float, float]], *, min_silence_gap_sec: float) -> list[tuple[float, float]]:
    if not segments:
        return []
    merged = [list(segments[0])]
    for start, end in segments[1:]:
        gap = start - float(merged[-1][1])
        if gap < min_silence_gap_sec:
            merged[-1][1] = max(float(merged[-1][1]), end)
        else:
            merged.append([start, end])
    return [(float(start), float(end)) for start, end in merged]


def _energy_vad_segments(wav_path: Path) -> tuple[list[dict[str, Any]], str]:
    waveform, sample_rate = _read_waveform(wav_path)
    if waveform.size == 0:
        return [], "energy_fallback"

    frame_size = max(1, int(sample_rate * FRAME_MS / 1000.0))
    hop_size = max(1, int(sample_rate * HOP_MS / 1000.0))
    frame_sec = frame_size / sample_rate
    hop_sec = hop_size / sample_rate
    total_duration_sec = waveform.size / sample_rate

    rms_values: list[float] = []
    for start in range(0, max(1, waveform.size - frame_size + 1), hop_size):
        frame = waveform[start : start + frame_size]
        if frame.size == 0:
            continue
        rms_values.append(float(np.sqrt(np.mean(np.square(frame)))))

    if not rms_values:
        return [], "energy_fallback"

    rms_array = np.asarray(rms_values, dtype=np.float32)
    noise_floor = float(np.percentile(rms_array, 20))
    threshold = max(ENERGY_THRESHOLD_FLOOR, noise_floor * ENERGY_NOISE_MULTIPLIER)
    speech_mask = rms_array >= threshold

    raw_segments = _segments_from_mask(speech_mask, hop_sec, frame_sec, total_duration_sec)
    merged_segments = _merge_close_segments(raw_segments, min_silence_gap_sec=MIN_SILENCE_GAP_SEC)
    min_speech_duration = MIN_SPEECH_SEGMENT_SEC

    segments: list[dict[str, Any]] = []
    for start, end in merged_segments:
        duration = end - start
        if duration < min_speech_duration:
            continue
        segments.append(
            {
                "start": round(start, 6),
                "end": round(end, 6),
                "duration": round(duration, 6),
                "confidence": None,
                "method": "energy_fallback",
            }
        )
    return segments, "energy_fallback"


def _run_vad(wav_path: Path) -> tuple[list[dict[str, Any]], str, list[str]]:
    try:
        segments, method = _silero_vad_segments(wav_path)
        return segments, method, []
    except AudioVadError as exc:
        if str(exc) != "silero_vad_not_available":
            raise
    segments, method = _energy_vad_segments(wav_path)
    return segments, method, ["ENERGY_FALLBACK_VAD_USED"]


def _compute_speech_stats(
    *,
    total_duration_sec: float,
    speech_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    speech_duration_sec = round(sum(float(seg["duration"]) for seg in speech_segments), 6)
    silence_duration_sec = round(max(0.0, total_duration_sec - speech_duration_sec), 6)
    speech_ratio = speech_duration_sec / total_duration_sec if total_duration_sec > 0 else 0.0
    silence_ratio = silence_duration_sec / total_duration_sec if total_duration_sec > 0 else 0.0

    if speech_segments:
        leading_silence = round(float(speech_segments[0]["start"]), 6)
        trailing_silence = round(max(0.0, total_duration_sec - float(speech_segments[-1]["end"])), 6)
        pause_count = 0
        for previous, current in zip(speech_segments, speech_segments[1:]):
            gap = float(current["start"]) - float(previous["end"])
            if gap >= PAUSE_THRESHOLD_SEC:
                pause_count += 1
    else:
        leading_silence = round(total_duration_sec, 6)
        trailing_silence = round(total_duration_sec, 6)
        pause_count = 0

    breathing_like_pattern = _build_breathing_like_pattern(speech_segments)

    return {
        "speech_duration_sec": speech_duration_sec,
        "silence_duration_sec": silence_duration_sec,
        "speech_ratio": round(speech_ratio, 6),
        "silence_ratio": round(silence_ratio, 6),
        "pause_count": pause_count,
        "leading_silence": leading_silence,
        "trailing_silence": trailing_silence,
        "breathing_like_pattern": breathing_like_pattern,
        "breathing_like_pattern_note": "approximate gap-based heuristic derived from detected speech pauses",
    }


def _build_breathing_like_pattern(speech_segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(speech_segments) < 2:
        return None

    candidate_gap_durations: list[float] = []
    for previous, current in zip(speech_segments, speech_segments[1:]):
        gap_duration_sec = float(current["start"]) - float(previous["end"])
        if BREATH_GAP_MIN_SEC <= gap_duration_sec <= BREATH_GAP_MAX_SEC:
            candidate_gap_durations.append(round(gap_duration_sec, 6))

    if not candidate_gap_durations:
        return None

    return {
        "detected": True,
        "method": "vad_gap_heuristic",
        "candidate_gap_count": len(candidate_gap_durations),
        "candidate_gap_durations_sec": candidate_gap_durations,
    }


def _update_quality_flags(
    existing_flags: list[str],
    *,
    stats: dict[str, Any],
    speech_segments: list[dict[str, Any]],
    fallback_flags: list[str],
) -> list[str]:
    flags = list(existing_flags)
    flags.extend(fallback_flags)

    if not speech_segments:
        flags.append("NO_DETECTED_SPEECH")
    if stats["speech_ratio"] < LOW_SPEECH_RATIO_THRESHOLD:
        flags.append("LOW_SPEECH_RATIO")
    if stats["silence_ratio"] >= HIGH_SILENCE_RATIO_THRESHOLD:
        flags.append("HIGH_SILENCE_RATIO")
    if stats["leading_silence"] >= LONG_SILENCE_THRESHOLD_SEC:
        flags.append("LONG_LEADING_SILENCE")
    if stats["trailing_silence"] >= LONG_SILENCE_THRESHOLD_SEC:
        flags.append("LONG_TRAILING_SILENCE")

    return list(dict.fromkeys(flags))


def _decide_limits(
    *,
    stats: dict[str, Any],
    speech_segments: list[dict[str, Any]],
    existing_limits: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    limits = {
        "unsupported_reason": existing_limits.get("unsupported_reason"),
        "low_evidence_reason": existing_limits.get("low_evidence_reason"),
    }

    if not speech_segments:
        limits["unsupported_reason"] = "no_detected_human_speech"
        return False, limits

    if stats["speech_duration_sec"] < MIN_SPEECH_DURATION_SEC or stats["speech_ratio"] < MIN_SPEECH_RATIO:
        limits["low_evidence_reason"] = "too_little_detected_speech"
        return True, limits

    return True, limits


def _default_json_output_for_preprocess(preprocess_json_path: Path) -> Path:
    return preprocess_json_path.parent / "audio_vad_result.json"


def _default_json_output_for_wav(wav_path: Path) -> Path:
    return wav_path.parent / "audio_vad_result.json"


def save_audio_vad_result(result: dict[str, Any], json_output_path: Path) -> Path:
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_output_path


def run_audio_vad(
    *,
    preprocess_json_path: str | Path | None = None,
    input_wav_path: str | Path | None = None,
    json_output_path: str | Path | None = None,
) -> dict[str, Any]:
    wav_path, existing_flags, existing_limits, preprocess_json = _resolve_inputs(
        preprocess_json_path=preprocess_json_path,
        input_wav_path=input_wav_path,
    )
    normalized_metadata = _validate_normalized_wav(wav_path)
    speech_segments, vad_method, fallback_flags = _run_vad(wav_path)
    stats = _compute_speech_stats(
        total_duration_sec=float(normalized_metadata["duration_sec"]),
        speech_segments=speech_segments,
    )
    quality_flags = _update_quality_flags(
        existing_flags,
        stats=stats,
        speech_segments=speech_segments,
        fallback_flags=fallback_flags,
    )
    human_speech_detected, limits = _decide_limits(
        stats=stats,
        speech_segments=speech_segments,
        existing_limits=existing_limits,
    )

    result = {
        "audio_vad": {
            "input_wav_path": str(wav_path),
            "vad_method": vad_method,
            "total_duration_sec": round(float(normalized_metadata["duration_sec"]), 6),
            "human_speech_detected": human_speech_detected,
            "speech_segments": speech_segments,
            "speech_stats": stats,
            "quality_flags": quality_flags,
        },
        "limits": limits,
    }

    output_path = (
        Path(json_output_path).expanduser().resolve()
        if json_output_path
        else _default_json_output_for_preprocess(Path(preprocess_json).expanduser().resolve())
        if preprocess_json
        else _default_json_output_for_wav(wav_path)
    )
    save_audio_vad_result(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="오디오 VAD / 음성 구간 태깅만 단독 실행합니다.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preprocess-json", help="1단계 전처리 JSON 경로")
    group.add_argument("--input-wav", help="normalized 16kHz mono WAV 경로")
    parser.add_argument("--json-output", default=None, help="VAD 결과 JSON 저장 경로")
    args = parser.parse_args()

    result = run_audio_vad(
        preprocess_json_path=args.preprocess_json,
        input_wav_path=args.input_wav,
        json_output_path=args.json_output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
