from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from services.ai.evaluation.manifest import ManifestSample


DEFAULT_FAKE_THRESHOLD = 0.5
DEFAULT_VIDEO_REVIEW_THRESHOLD = 0.3
DEFAULT_AUDIO_REAL_THRESHOLD = 0.1


@dataclass(frozen=True)
class PredictionPaths:
    video_jsonl: Path
    audio_jsonl: Path
    fusion_jsonl: Path
    combined_jsonl: Path
    combined_csv: Path
    failed_jsonl: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class PredictionRecords:
    video: dict[str, Any] | None
    audio: dict[str, Any] | None
    fusion: dict[str, Any] | None
    combined: dict[str, Any]


def _nested_get(payload: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def extract_video_fake_score(video_result: Mapping[str, Any] | None) -> float | None:
    if video_result is None:
        return None
    nested_score = _float_or_none(
        _nested_get(video_result, ("detection", "video_score", "final_fake_score"))
    )
    if nested_score is not None:
        return nested_score
    return _float_or_none(_nested_get(video_result, ("video_score", "final_fake_score")))


def extract_audio_fake_score(audio_result: Mapping[str, Any] | None) -> float | None:
    if audio_result is None:
        return None
    scored_window_count = _int_or_none(audio_result.get("scored_window_count"))
    if scored_window_count is None:
        scored_window_count = _int_or_none(
            _nested_get(audio_result, ("audio_inference", "scored_window_count"))
        )
    if scored_window_count is not None and scored_window_count <= 0:
        return None
    preferred = _float_or_none(audio_result.get("audio_fake_score"))
    if preferred is not None:
        return preferred
    return _float_or_none(audio_result.get("audio_fake_prob_like"))


def fuse_fake_scores(video_score: float, audio_score: float) -> float:
    return (float(video_score) + float(audio_score)) / 2.0


def classify_fusion_triage(
    video_score: float,
    audio_score: float,
    *,
    fake_threshold: float = DEFAULT_FAKE_THRESHOLD,
    video_review_threshold: float = DEFAULT_VIDEO_REVIEW_THRESHOLD,
    audio_real_threshold: float = DEFAULT_AUDIO_REAL_THRESHOLD,
) -> str:
    video_score = float(video_score)
    audio_score = float(audio_score)
    if audio_score >= fake_threshold:
        return "fake"
    if video_score >= video_review_threshold and audio_score <= audio_real_threshold:
        return "needs_review"
    if video_score >= fake_threshold:
        return "fake"
    return "real"


def _extract_audio_count(audio_result: Mapping[str, Any] | None, key: str) -> int | None:
    if audio_result is None:
        return None
    value = audio_result.get(key)
    if value is None:
        value = _nested_get(audio_result, ("audio_inference", key))
    return _int_or_none(value)


def _extract_audio_model_error(audio_result: Mapping[str, Any] | None) -> str | None:
    if audio_result is None:
        return None
    value = audio_result.get("audio_model_error")
    if value:
        return str(value)
    model_errors = _nested_get(audio_result, ("audio_inference", "model_errors"))
    if isinstance(model_errors, list) and model_errors:
        return str(model_errors[0])
    return None


def _base_record(sample: ManifestSample, modality: str, label: int, fake_score: float) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "dataset_name": sample.dataset_name,
        "category": sample.category,
        "type": sample.type,
        "label_name": sample.label_name,
        "split": sample.split,
        "filename": sample.filename,
        "video_path": sample.video_path,
        "audio_path": sample.audio_path,
        "modality": modality,
        "label": int(label),
        "fake_score": float(fake_score),
    }


def build_prediction_records(
    sample: ManifestSample,
    *,
    video_result: Mapping[str, Any] | None = None,
    audio_result: Mapping[str, Any] | None = None,
) -> PredictionRecords:
    video_score = extract_video_fake_score(video_result)
    audio_score = extract_audio_fake_score(audio_result)
    audio_scored_window_count = _extract_audio_count(audio_result, "scored_window_count")
    audio_failed_window_count = _extract_audio_count(audio_result, "failed_window_count")
    audio_skipped_window_count = _extract_audio_count(audio_result, "skipped_window_count")
    audio_model_error = _extract_audio_model_error(audio_result)

    video = (
        _base_record(sample, "video", sample.labels["video_label"], video_score)
        if video_score is not None
        else None
    )
    audio = (
        _base_record(sample, "audio", sample.labels["audio_label"], audio_score)
        if audio_score is not None
        else None
    )
    fusion_score = None
    fusion_triage_label = None
    if video_score is not None and audio_score is not None:
        fusion_score = fuse_fake_scores(video_score, audio_score)
        fusion_triage_label = classify_fusion_triage(video_score, audio_score)
    fusion = (
        _base_record(sample, "fusion", sample.labels["fusion_label"], fusion_score)
        if fusion_score is not None
        else None
    )
    if fusion is not None and video_score is not None and audio_score is not None:
        fusion["triage_label"] = fusion_triage_label
    combined = {
        "sample_id": sample.sample_id,
        "dataset_name": sample.dataset_name,
        "category": sample.category,
        "type": sample.type,
        "label_name": sample.label_name,
        "split": sample.split,
        "filename": sample.filename,
        "video_path": sample.video_path,
        "audio_path": sample.audio_path,
        "video_label": sample.labels["video_label"],
        "audio_label": sample.labels["audio_label"],
        "fusion_label": sample.labels["fusion_label"],
        "video_fake_score": video_score,
        "audio_fake_score": audio_score,
        "fusion_fake_score": fusion_score,
        "fusion_triage_label": fusion_triage_label,
        "audio_scored_window_count": audio_scored_window_count,
        "audio_failed_window_count": audio_failed_window_count,
        "audio_skipped_window_count": audio_skipped_window_count,
        "audio_model_error": audio_model_error,
    }
    return PredictionRecords(video=video, audio=audio, fusion=fusion, combined=combined)


def load_existing_sample_ids(paths: Iterable[str | Path]) -> set[str]:
    sample_ids: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as jsonl_file:
            for line in jsonl_file:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = row.get("sample_id")
                if sample_id:
                    sample_ids.add(str(sample_id))
    return sample_ids


def load_complete_combined_sample_ids(paths: str | Path | Iterable[str | Path]) -> set[str]:
    sample_ids: set[str] = set()
    if isinstance(paths, (str, Path)):
        raw_paths: Iterable[str | Path] = [paths]
    else:
        raw_paths = paths

    for raw_path in raw_paths:
        combined_path = Path(raw_path)
        if not combined_path.exists():
            continue
        with combined_path.open("r", encoding="utf-8") as jsonl_file:
            for line in jsonl_file:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = row.get("sample_id")
                if sample_id:
                    sample_ids.add(str(sample_id))
    return sample_ids


class PredictionWriter:
    def __init__(self, run_dir: str | Path, *, suffix: str = "") -> None:
        root = Path(run_dir)
        root.mkdir(parents=True, exist_ok=True)
        suffix = suffix.strip()
        self.paths = PredictionPaths(
            video_jsonl=root / f"predictions_video{suffix}.jsonl",
            audio_jsonl=root / f"predictions_audio{suffix}.jsonl",
            fusion_jsonl=root / f"predictions_fusion{suffix}.jsonl",
            combined_jsonl=root / f"predictions_combined{suffix}.jsonl",
            combined_csv=root / f"predictions_combined{suffix}.csv",
            failed_jsonl=root / f"failed_cases{suffix}.jsonl",
        )
        self._resume_combined_paths = [root / "predictions_combined.jsonl", self.paths.combined_jsonl]

    def processed_sample_ids(self) -> set[str]:
        return load_complete_combined_sample_ids(self._resume_combined_paths)

    def write_records(self, records: PredictionRecords) -> None:
        if records.video is not None:
            self._append_jsonl(self.paths.video_jsonl, records.video)
        if records.audio is not None:
            self._append_jsonl(self.paths.audio_jsonl, records.audio)
        if records.fusion is not None:
            self._append_jsonl(self.paths.fusion_jsonl, records.fusion)
        self._append_jsonl(self.paths.combined_jsonl, records.combined)
        self._append_csv(records.combined)

    def write_failure(self, sample: ManifestSample, *, modality: str, error: str) -> None:
        self._append_jsonl(
            self.paths.failed_jsonl,
            {
                "sample_id": sample.sample_id,
                "dataset_name": sample.dataset_name,
                "category": sample.category,
                "modality": modality,
                "filename": sample.filename,
                "video_path": sample.video_path,
                "audio_path": sample.audio_path,
                "error": error,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    @staticmethod
    def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as jsonl_file:
            jsonl_file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    def _append_csv(self, row: Mapping[str, Any]) -> None:
        self.paths.combined_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(row.keys())
        should_write_header = not self.paths.combined_csv.exists() or self.paths.combined_csv.stat().st_size == 0
        with self.paths.combined_csv.open("a", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if should_write_header:
                writer.writeheader()
            writer.writerow(dict(row))
