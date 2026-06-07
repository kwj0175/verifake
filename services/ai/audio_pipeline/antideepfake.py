"""Thin audio-pipeline wrapper around the vendored AntiDeepfake inference script.

This intentionally keeps the upstream model/runtime path intact. The wrapper only:
- resolves repo-relative config/checkpoint paths,
- writes a temporary protocol CSV that AntiDeepfake already knows how to read,
- invokes the vendored inference script,
- parses the emitted score CSV.

TODO(verifake): map these raw AntiDeepfake scores into the fuller AudioAnalysisResult
schema once the rest of the audio evidence pipeline is implemented.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import subprocess
import sys
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from services.ai.common.runtime_probe import resolve_torch_device


REPO_ROOT = Path(__file__).resolve().parents[3]
ANTIDEEPFAKE_ROOT = REPO_ROOT / "services/ai/antideepfake"
ANTIDEEPFAKE_MAIN = ANTIDEEPFAKE_ROOT / "main.py"
DEFAULT_HPARAMS_PATH = ANTIDEEPFAKE_ROOT / "hparams/mms_300m_audio_pipeline.yaml"
DEFAULT_CHECKPOINT_PATH = (
    REPO_ROOT / "services/ai/checkpoints/audio/antideepfake/mms_300m.ckpt"
)

PROTOCOL_COLUMNS = [
    "ID",
    "Label",
    "Duration",
    "SampleRate",
    "Path",
    "Attack",
    "Speaker",
    "Proportion",
    "AudioChannel",
    "AudioEncoding",
    "AudioBitSample",
    "Language",
]


@dataclass(frozen=True)
class AudioFileMetadata:
    duration_seconds: float
    sample_rate: int
    channels: int
    encoding: str
    bits_per_sample: int


@dataclass(frozen=True)
class AntiDeepfakeInferenceResult:
    request_id: str
    file_path: str
    fake_logit: float
    real_logit: float
    fake_probability: float
    real_probability: float
    predicted_label: str
    score_csv_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _resolve_existing_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _read_audio_metadata(audio_path: Path) -> AudioFileMetadata:
    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as handle:
                sample_rate = handle.getframerate()
                channels = handle.getnchannels()
                bits_per_sample = handle.getsampwidth() * 8
                frame_count = handle.getnframes()
        except (wave.Error, OSError) as exc:
            raise RuntimeError(f"Failed to read WAV metadata: {audio_path}") from exc

        duration_seconds = float(frame_count) / float(sample_rate) if sample_rate > 0 else 0.0
        encoding = "PCM_S" if bits_per_sample else "unknown"
        return AudioFileMetadata(
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=channels or 1,
            encoding=encoding,
            bits_per_sample=bits_per_sample,
        )

    import importlib

    torchaudio = importlib.import_module("torchaudio")

    info = torchaudio.info(str(audio_path))
    sample_rate = int(info.sample_rate)
    channels = int(info.num_channels or 1)
    bits_per_sample = int(info.bits_per_sample or 0)
    duration_seconds = 0.0
    if sample_rate > 0 and info.num_frames > 0:
        duration_seconds = float(info.num_frames) / float(sample_rate)

    return AudioFileMetadata(
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        encoding=str(info.encoding or "unknown"),
        bits_per_sample=bits_per_sample,
    )


def _protocol_audio_path(audio_path: Path) -> str:
    return str(audio_path.resolve().as_posix())


def _write_protocol_csv(
    protocol_path: Path,
    audio_path: Path,
    metadata: AudioFileMetadata,
    request_id: str,
) -> None:
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    path_value = _protocol_audio_path(audio_path)
    base_row = {
        "Duration": f"{metadata.duration_seconds:.6f}",
        "SampleRate": str(metadata.sample_rate),
        "Path": path_value,
        "Attack": "-",
        "Speaker": request_id,
        "Proportion": "service",
        "AudioChannel": str(metadata.channels),
        "AudioEncoding": metadata.encoding,
        "AudioBitSample": str(metadata.bits_per_sample),
        "Language": "UNK",
    }
    rows = [
        {
            "ID": f"{request_id}-fake-probe",
            "Label": "fake",
            **base_row,
        },
        {
            "ID": f"{request_id}-real-probe",
            "Label": "real",
            **base_row,
        },
    ]

    with protocol_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROTOCOL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _stable_softmax(fake_logit: float, real_logit: float) -> tuple[float, float]:
    pivot = max(fake_logit, real_logit)
    exp_fake = math.exp(fake_logit - pivot)
    exp_real = math.exp(real_logit - pivot)
    total = exp_fake + exp_real
    return exp_fake / total, exp_real / total


def _parse_score_row(score_csv_path: Path, request_id: str) -> AntiDeepfakeInferenceResult:
    with score_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"AntiDeepfake score file is empty: {score_csv_path}")

    target_row = next(
        (row for row in rows if row.get("ID") == f"{request_id}-fake-probe"),
        rows[0],
    )

    try:
        fake_logit, real_logit = ast.literal_eval(target_row["Score"])
    except (SyntaxError, ValueError, KeyError) as exc:
        raise RuntimeError(
            f"Failed to parse AntiDeepfake scores from {score_csv_path}"
        ) from exc

    fake_logit = float(fake_logit)
    real_logit = float(real_logit)
    fake_probability, real_probability = _stable_softmax(fake_logit, real_logit)

    return AntiDeepfakeInferenceResult(
        request_id=request_id,
        file_path=target_row.get("Path", ""),
        fake_logit=fake_logit,
        real_logit=real_logit,
        fake_probability=fake_probability,
        real_probability=real_probability,
        predicted_label="real" if real_probability >= 0.5 else "fake",
        score_csv_path=str(score_csv_path),
    )


def _build_command(
    *,
    python_executable: str,
    hparams_path: Path,
    base_path: Path,
    protocol_path: Path,
    output_folder: Path,
    score_path: Path,
    checkpoint_path: Path,
    device: str | None,
) -> list[str]:
    command = [
        python_executable,
        str(ANTIDEEPFAKE_MAIN),
        "inference",
        str(hparams_path),
        "--base_path",
        str(base_path),
        "--data_folder",
        "/",
        "--test_csv",
        str(protocol_path),
        "--output_folder",
        str(output_folder),
        "--score_path",
        str(score_path),
        "--ckpt_path",
        str(checkpoint_path),
        "--use_da",
        "False",
    ]
    if device:
        command.extend(["--device", device])
    return command


def _run_command(command: list[str]) -> None:
    try:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "AntiDeepfake inference failed. "
            f"stdout:\n{exc.stdout}\n\nstderr:\n{exc.stderr}"
        ) from exc


def run_antideepfake_inference(
    file_path: str | Path,
    *,
    request_id: str = "audio-pipeline",
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    hparams_path: str | Path = DEFAULT_HPARAMS_PATH,
    python_executable: str = sys.executable,
    artifacts_dir: str | Path | None = None,
    device: str | None = None,
) -> AntiDeepfakeInferenceResult:
    if not device:
        device = resolve_torch_device(default="cpu", cuda_index_env_var="VERIFAKE_CUDA_DEVICE")

    audio_path = _resolve_existing_file(file_path, label="audio file")
    resolved_hparams_path = _resolve_existing_file(hparams_path, label="hparams file")
    resolved_checkpoint_path = _resolve_existing_file(
        checkpoint_path,
        label=(
            "AntiDeepfake checkpoint"
            f" (expected default location: {DEFAULT_CHECKPOINT_PATH})"
        ),
    )

    metadata = _read_audio_metadata(audio_path)

    if artifacts_dir is None:
        with tempfile.TemporaryDirectory(prefix="antideepfake_") as temp_dir:
            return _run_inference_in_workspace(
                audio_path=audio_path,
                metadata=metadata,
                request_id=request_id,
                resolved_hparams_path=resolved_hparams_path,
                resolved_checkpoint_path=resolved_checkpoint_path,
                python_executable=python_executable,
                workspace=Path(temp_dir),
                keep_score_path=False,
                device=device,
            )

    workspace = Path(artifacts_dir).expanduser().resolve()
    workspace = workspace / request_id
    workspace.mkdir(parents=True, exist_ok=True)
    return _run_inference_in_workspace(
        audio_path=audio_path,
        metadata=metadata,
        request_id=request_id,
        resolved_hparams_path=resolved_hparams_path,
        resolved_checkpoint_path=resolved_checkpoint_path,
        python_executable=python_executable,
        workspace=workspace,
        keep_score_path=True,
        device=device,
    )


def _run_inference_in_workspace(
    *,
    audio_path: Path,
    metadata: AudioFileMetadata,
    request_id: str,
    resolved_hparams_path: Path,
    resolved_checkpoint_path: Path,
    python_executable: str,
    workspace: Path,
    keep_score_path: bool,
    device: str | None,
) -> AntiDeepfakeInferenceResult:
    protocol_path = workspace / "protocol.csv"
    output_folder = workspace / "output"
    score_path = output_folder / "evaluation_score.csv"

    _write_protocol_csv(protocol_path, audio_path, metadata, request_id)
    output_folder.mkdir(parents=True, exist_ok=True)

    command = _build_command(
        python_executable=python_executable,
        hparams_path=resolved_hparams_path,
        base_path=REPO_ROOT,
        protocol_path=protocol_path,
        output_folder=output_folder,
        score_path=score_path,
        checkpoint_path=resolved_checkpoint_path,
        device=device,
    )
    _run_command(command)

    result = _parse_score_row(score_path, request_id)
    if keep_score_path:
        return AntiDeepfakeInferenceResult(
            request_id=result.request_id,
            file_path=str(audio_path),
            fake_logit=result.fake_logit,
            real_logit=result.real_logit,
            fake_probability=result.fake_probability,
            real_probability=result.real_probability,
            predicted_label=result.predicted_label,
            score_csv_path=str(score_path),
        )

    return AntiDeepfakeInferenceResult(
        request_id=result.request_id,
        file_path=str(audio_path),
        fake_logit=result.fake_logit,
        real_logit=result.real_logit,
        fake_probability=result.fake_probability,
        real_probability=result.real_probability,
        predicted_label=result.predicted_label,
        score_csv_path=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AntiDeepfake via the audio pipeline wrapper.")
    parser.add_argument("file_path", help="Path to the audio file to analyze.")
    parser.add_argument("--request-id", default="audio-pipeline")
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--hparams-path", default=str(DEFAULT_HPARAMS_PATH))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Optional directory to keep the generated protocol and score CSV.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional SpeechBrain runtime device override, for example 'cpu' or 'cuda:0'.",
    )
    args = parser.parse_args()

    result = run_antideepfake_inference(
        args.file_path,
        request_id=args.request_id,
        checkpoint_path=args.checkpoint_path,
        hparams_path=args.hparams_path,
        python_executable=args.python_executable,
        artifacts_dir=args.artifacts_dir,
        device=args.device,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
