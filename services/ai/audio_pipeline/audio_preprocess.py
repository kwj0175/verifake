"""오디오 입력을 내부 분석용 WAV로 정규화하는 전처리 모듈."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
SUPPORTED_INPUT_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS

NORMALIZED_SAMPLE_RATE = 16000
NORMALIZED_CHANNELS = 1
NORMALIZED_CODEC = "pcm_s16le"
TOO_SHORT_THRESHOLD_SEC = 1.0
LOW_BITRATE_THRESHOLD_BPS = 64_000
SILENCE_THRESHOLD = 1e-3
SILENT_AUDIO_RMS_THRESHOLD = 5e-4
HIGH_SILENCE_RATIO_THRESHOLD = 0.98
CLIPPING_RATIO_THRESHOLD = 1e-3


class AudioPreprocessError(RuntimeError):
    """오디오 전처리 실패를 나타내는 기본 예외."""


class UnsupportedInputError(AudioPreprocessError):
    """지원하지 않는 입력 형식일 때 발생한다."""


class NoAudioStreamError(AudioPreprocessError):
    """입력 파일에 오디오 스트림이 없을 때 발생한다."""


@dataclass(frozen=True)
class ProbeMetadata:
    """ffprobe 결과에서 필요한 메타데이터만 추린다."""

    input_path: str
    file_extension: str
    container_format: str | None
    video_stream_exists: bool
    audio_stream_index: int | None
    codec: str | None
    bitrate: int | None
    duration_sec: float | None
    sample_rate: int | None
    channel_count: int | None
    has_audio_stream: bool


def _require_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise AudioPreprocessError(f"{name} 실행 파일을 찾을 수 없습니다.")
    return binary


def _encoding_to_codec_name(encoding: str | None, bits_per_sample: int | None) -> str | None:
    if encoding is None:
        return None
    normalized_encoding = encoding.lower()
    if normalized_encoding == "pcm_s" and bits_per_sample == 16:
        return "pcm_s16le"
    return normalized_encoding


def _parse_optional_int(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_float(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_ffprobe(input_path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        if input_path.suffix.lower() == ".wav":
            return _run_wave_probe(input_path)
        return _run_torchaudio_probe(input_path)

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        (
            "format=format_name,bit_rate,duration:"
            "stream=index,codec_type,codec_name,bit_rate,duration,sample_rate,channels"
        ),
        "-of",
        "json",
        str(input_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        detail = f" stderr: {stderr}" if stderr else ""
        raise AudioPreprocessError(
            "오디오 메타데이터를 읽을 수 없습니다. "
            f"ffprobe 실패: {input_path}{detail}"
        ) from exc

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AudioPreprocessError(
            "ffprobe 출력 JSON을 파싱할 수 없습니다. "
            f"input: {input_path}"
        ) from exc


def _run_torchaudio_probe(input_path: Path) -> dict[str, Any]:
    import importlib

    torchaudio = importlib.import_module("torchaudio")

    try:
        info = torchaudio.info(str(input_path))
    except (RuntimeError, OSError) as exc:
        raise AudioPreprocessError(
            f"오디오 메타데이터를 읽을 수 없습니다: {input_path}"
        ) from exc

    sample_rate = int(info.sample_rate or 0)
    channel_count = int(info.num_channels or 0)
    bits_per_sample = int(info.bits_per_sample or 0)
    duration_sec = None
    if sample_rate > 0 and info.num_frames > 0:
        duration_sec = float(info.num_frames) / float(sample_rate)

    codec_name = _encoding_to_codec_name(getattr(info, "encoding", None), bits_per_sample)
    estimated_bitrate = None
    if sample_rate > 0 and channel_count > 0 and bits_per_sample > 0:
        estimated_bitrate = sample_rate * channel_count * bits_per_sample

    return {
        "format": {
            "format_name": input_path.suffix.lower().lstrip("."),
            "bit_rate": estimated_bitrate,
            "duration": duration_sec,
        },
        "streams": [
            {
                "index": 0,
                "codec_type": "audio",
                "codec_name": codec_name,
                "bit_rate": estimated_bitrate,
                "duration": duration_sec,
                "sample_rate": sample_rate,
                "channels": channel_count,
            }
        ],
    }


def _run_wave_probe(input_path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(input_path), "rb") as handle:
            sample_rate = handle.getframerate()
            channel_count = handle.getnchannels()
            bits_per_sample = handle.getsampwidth() * 8
            frame_count = handle.getnframes()
    except (wave.Error, OSError) as exc:
        raise AudioPreprocessError(
            f"wave 모듈로 오디오 메타데이터를 읽을 수 없습니다: {input_path}"
        ) from exc

    duration_sec = float(frame_count) / float(sample_rate) if sample_rate > 0 else None
    estimated_bitrate = None
    if sample_rate > 0 and channel_count > 0 and bits_per_sample > 0:
        estimated_bitrate = sample_rate * channel_count * bits_per_sample

    return {
        "format": {
            "format_name": "wav",
            "bit_rate": estimated_bitrate,
            "duration": duration_sec,
        },
        "streams": [
            {
                "index": 0,
                "codec_type": "audio",
                "codec_name": "pcm_s16le" if bits_per_sample == 16 else "unknown",
                "bit_rate": estimated_bitrate,
                "duration": duration_sec,
                "sample_rate": sample_rate,
                "channels": channel_count,
            }
        ],
    }


def _extract_probe_metadata(input_path: Path) -> ProbeMetadata:
    probe = _run_ffprobe(input_path)
    streams = probe.get("streams", [])
    format_info = probe.get("format", {})
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)

    return ProbeMetadata(
        input_path=str(input_path.resolve()),
        file_extension=input_path.suffix.lower(),
        container_format=format_info.get("format_name"),
        video_stream_exists=video_stream is not None,
        audio_stream_index=_parse_optional_int(audio_stream.get("index")) if audio_stream else None,
        codec=audio_stream.get("codec_name") if audio_stream else None,
        bitrate=(
            _parse_optional_int(audio_stream.get("bit_rate"))
            if audio_stream is not None
            else _parse_optional_int(format_info.get("bit_rate"))
        ),
        duration_sec=(
            _parse_optional_float(audio_stream.get("duration"))
            if audio_stream is not None
            else _parse_optional_float(format_info.get("duration"))
        ),
        sample_rate=_parse_optional_int(audio_stream.get("sample_rate")) if audio_stream else None,
        channel_count=_parse_optional_int(audio_stream.get("channels")) if audio_stream else None,
        has_audio_stream=audio_stream is not None,
    )


def _validate_input_file(input_path: Path) -> ProbeMetadata:
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        raise UnsupportedInputError(
            f"지원하지 않는 입력 확장자입니다: {input_path.suffix.lower()}"
        )

    metadata = _extract_probe_metadata(input_path)
    if not metadata.has_audio_stream:
        raise NoAudioStreamError("입력 파일에 오디오 스트림이 없습니다.")
    return metadata


def _normalize_audio(input_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        if input_path.suffix.lower() == ".wav":
            _normalize_wave_with_stdlib(input_path, output_path)
            return
        _normalize_audio_with_torchaudio(input_path, output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        NORMALIZED_CODEC,
        "-ar",
        str(NORMALIZED_SAMPLE_RATE),
        "-ac",
        str(NORMALIZED_CHANNELS),
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise AudioPreprocessError(
            "오디오 정규화에 실패했습니다. "
            f"stderr: {exc.stderr.strip()}"
        ) from exc


def _normalize_audio_with_torchaudio(input_path: Path, output_path: Path) -> None:
    import importlib

    torchaudio = importlib.import_module("torchaudio")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        waveform, sample_rate = torchaudio.load(str(input_path))
    except (RuntimeError, OSError) as exc:
        raise AudioPreprocessError(
            f"torchaudio로 입력 파일을 읽을 수 없습니다: {input_path}"
        ) from exc

    if waveform.ndim != 2 or waveform.numel() == 0:
        raise AudioPreprocessError("입력 오디오 파형이 비어 있습니다.")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != NORMALIZED_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=NORMALIZED_SAMPLE_RATE,
        )

    torchaudio.save(
        str(output_path),
        waveform,
        NORMALIZED_SAMPLE_RATE,
        format="wav",
        encoding="PCM_S",
        bits_per_sample=16,
    )


def _normalize_wave_with_stdlib(input_path: Path, output_path: Path) -> None:
    try:
        with wave.open(str(input_path), "rb") as handle:
            sample_rate = handle.getframerate()
            channel_count = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frame_count = handle.getnframes()
            raw_frames = handle.readframes(frame_count)
    except (wave.Error, OSError) as exc:
        raise AudioPreprocessError(f"WAV 입력을 읽을 수 없습니다: {input_path}") from exc

    if sample_width != 2:
        raise AudioPreprocessError("stdlib WAV fallback은 16-bit PCM WAV만 지원합니다.")
    if channel_count <= 0 or sample_rate <= 0:
        raise AudioPreprocessError("입력 WAV 메타데이터가 올바르지 않습니다.")

    waveform = np.frombuffer(raw_frames, dtype="<i2").astype(np.float32)
    if channel_count > 1:
        waveform = waveform.reshape(-1, channel_count).mean(axis=1)
    waveform /= 32768.0

    if sample_rate != NORMALIZED_SAMPLE_RATE and waveform.size > 0:
        source_positions = np.arange(waveform.size, dtype=np.float32)
        target_length = max(1, int(round(waveform.size * NORMALIZED_SAMPLE_RATE / sample_rate)))
        target_positions = np.linspace(0.0, waveform.size - 1, target_length, dtype=np.float32)
        waveform = np.interp(target_positions, source_positions, waveform).astype(np.float32)

    waveform = np.clip(waveform, -1.0, 1.0)
    pcm_waveform = np.round(waveform * 32767.0).astype("<i2")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(NORMALIZED_CHANNELS)
        handle.setsampwidth(2)
        handle.setframerate(NORMALIZED_SAMPLE_RATE)
        handle.writeframes(pcm_waveform.tobytes())


def _read_pcm_waveform(wav_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as handle:
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        channel_count = handle.getnchannels()
        frame_count = handle.getnframes()
        raw_frames = handle.readframes(frame_count)

    if sample_width != 2:
        raise AudioPreprocessError("정규화된 WAV가 16-bit PCM이 아닙니다.")

    waveform = np.frombuffer(raw_frames, dtype="<i2").astype(np.float32)
    if channel_count > 1:
        waveform = waveform.reshape(-1, channel_count).mean(axis=1)
    waveform /= 32768.0
    return waveform, sample_rate


def _compute_quality_metrics(wav_path: Path) -> dict[str, float | None]:
    waveform, _ = _read_pcm_waveform(wav_path)
    if waveform.size == 0:
        return {
            "rms_level": None,
            "clipping_ratio": None,
            "silence_ratio_estimate": None,
        }

    rms = float(np.sqrt(np.mean(np.square(waveform))))
    clipping_ratio = float(np.mean(np.abs(waveform) >= 0.999))
    silence_ratio = float(np.mean(np.abs(waveform) <= SILENCE_THRESHOLD))
    return {
        "rms_level": rms,
        "clipping_ratio": clipping_ratio,
        "silence_ratio_estimate": silence_ratio,
    }


def _build_quality_flags(
    *,
    original_metadata: ProbeMetadata,
    normalized_duration_sec: float,
    file_size_bytes: int,
    quality_metrics: dict[str, float | None],
) -> list[str]:
    flags: list[str] = []
    if normalized_duration_sec <= 0.0:
        flags.append("EMPTY_AUDIO")
    if normalized_duration_sec < TOO_SHORT_THRESHOLD_SEC:
        flags.append("TOO_SHORT")
    if file_size_bytes <= 0:
        flags.append("EMPTY_AUDIO")
    if original_metadata.bitrate is not None and original_metadata.bitrate < LOW_BITRATE_THRESHOLD_BPS:
        flags.append("LOW_BITRATE_SOURCE")

    rms = quality_metrics.get("rms_level")
    if rms is not None and rms <= SILENT_AUDIO_RMS_THRESHOLD:
        flags.append("SILENT_AUDIO")

    silence_ratio = quality_metrics.get("silence_ratio_estimate")
    if silence_ratio is not None and silence_ratio >= HIGH_SILENCE_RATIO_THRESHOLD:
        flags.append("HIGH_SILENCE_RATIO")

    clipping_ratio = quality_metrics.get("clipping_ratio")
    if clipping_ratio is not None and clipping_ratio >= CLIPPING_RATIO_THRESHOLD:
        flags.append("CLIPPING_DETECTED")

    return list(dict.fromkeys(flags))


def _verify_normalized_audio(wav_path: Path) -> dict[str, Any]:
    if not wav_path.exists():
        raise AudioPreprocessError("정규화된 WAV 파일이 생성되지 않았습니다.")

    file_size_bytes = wav_path.stat().st_size
    if file_size_bytes <= 0:
        raise AudioPreprocessError("정규화된 WAV 파일 크기가 0입니다.")

    metadata = _extract_probe_metadata(wav_path)
    if not metadata.has_audio_stream:
        raise AudioPreprocessError("정규화된 WAV에서 오디오 스트림을 찾을 수 없습니다.")
    if metadata.sample_rate != NORMALIZED_SAMPLE_RATE:
        raise AudioPreprocessError("정규화된 WAV의 sample rate가 16kHz가 아닙니다.")
    if metadata.channel_count != NORMALIZED_CHANNELS:
        raise AudioPreprocessError("정규화된 WAV의 channel count가 mono가 아닙니다.")
    if metadata.duration_sec is None or metadata.duration_sec <= 0.0:
        raise AudioPreprocessError("정규화된 WAV의 duration이 0 이하입니다.")
    if metadata.codec != NORMALIZED_CODEC:
        raise AudioPreprocessError("정규화된 WAV codec이 pcm_s16le가 아닙니다.")

    return {
        "format": "wav",
        "codec": metadata.codec,
        "sample_rate": metadata.sample_rate,
        "channel_count": metadata.channel_count,
        "duration_sec": metadata.duration_sec,
        "file_size_bytes": file_size_bytes,
    }


def _default_normalized_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / "normalized" / f"{input_path.stem}_16k_mono.wav"


def _default_json_path(output_dir: Path) -> Path:
    return output_dir / "audio_preprocess_result.json"


def save_preprocess_result(result: dict[str, Any], json_output_path: Path) -> Path:
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return json_output_path


def preprocess_audio(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    json_output_path: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    original = _validate_input_file(input_path)
    normalized_wav_path = _default_normalized_path(input_path, output_dir)
    _normalize_audio(input_path, normalized_wav_path)
    normalized = _verify_normalized_audio(normalized_wav_path)
    quality_metrics = _compute_quality_metrics(normalized_wav_path)
    quality_flags = _build_quality_flags(
        original_metadata=original,
        normalized_duration_sec=float(normalized["duration_sec"]),
        file_size_bytes=int(normalized["file_size_bytes"]),
        quality_metrics=quality_metrics,
    )

    result = {
        "audio_preprocess": {
            "input_path": str(input_path),
            "file_extension": original.file_extension,
            "normalized_wav_path": str(normalized_wav_path),
            "original": {
                "codec": original.codec,
                "bitrate": original.bitrate,
                "duration_sec": original.duration_sec,
                "sample_rate": original.sample_rate,
                "channel_count": original.channel_count,
                "has_audio_stream": original.has_audio_stream,
                "container_format": original.container_format,
                "video_stream_exists": original.video_stream_exists,
                "audio_stream_index": original.audio_stream_index,
            },
            "normalized": normalized,
            "quality_metrics": quality_metrics,
            "quality_flags": quality_flags,
        },
        "limits": {
            "unsupported_reason": None,
            "low_evidence_reason": None,
        },
    }

    json_output = Path(json_output_path).expanduser().resolve() if json_output_path else _default_json_path(output_dir)
    save_preprocess_result(result, json_output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="오디오 전처리만 단독 실행합니다.")
    parser.add_argument("--input", required=True, help="입력 오디오 또는 영상 파일 경로")
    parser.add_argument("--output-dir", required=True, help="정규화 WAV를 저장할 출력 디렉터리")
    parser.add_argument(
        "--json-output",
        default=None,
        help="전처리 결과 JSON 저장 경로 (기본값: <output-dir>/audio_preprocess_result.json)",
    )
    args = parser.parse_args()

    result = preprocess_audio(args.input, args.output_dir, json_output_path=args.json_output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
