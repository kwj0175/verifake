from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LABELS_BY_TYPE = {
    "FakeVideo-FakeAudio": {"video_label": 1, "audio_label": 1, "fusion_label": 1},
    "FakeVideo-RealAudio": {"video_label": 1, "audio_label": 0, "fusion_label": 1},
    "RealVideo-FakeAudio": {"video_label": 0, "audio_label": 1, "fusion_label": 1},
    "RealVideo-RealAudio": {"video_label": 0, "audio_label": 0, "fusion_label": 0},
}


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    dataset_name: str
    source: str
    target1: str
    target2: str
    method: str
    category: str
    type: str
    race: str
    gender: str
    filename: str
    folder_path: str
    video_path: str
    audio_path: str
    labels: dict[str, int]
    label_name: str
    split: str = "test"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_cell(value: str) -> str:
    return value.strip().strip("\ufeff")


def _strip_dataset_prefix(folder_path: str) -> str:
    normalized = folder_path.replace("\\", "/").strip().strip("/")
    if normalized == "FakeAVCeleb":
        return ""
    prefix = "FakeAVCeleb/"
    if normalized.startswith(prefix):
        return normalized[len(prefix) :]
    return normalized


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return token.strip("-") or "unknown"


def _sample_id(*, source: str, category: str, filename: str, folder_path: str, method: str) -> str:
    stem = Path(filename).stem
    digest = hashlib.sha1(
        "|".join([source, category, filename, folder_path, method]).encode("utf-8")
    ).hexdigest()[:10]
    return "__".join(
        [_safe_token(source), _safe_token(category), _safe_token(stem), digest]
    )


def _row_to_sample(dataset_root: Path, row: list[str]) -> ManifestSample:
    if len(row) < 10:
        raise ValueError(f"FakeAVCeleb metadata row must have at least 10 columns: {row}")

    source = _clean_cell(row[0])
    target1 = _clean_cell(row[1])
    target2 = _clean_cell(row[2])
    method = _clean_cell(row[3])
    category = _clean_cell(row[4])
    sample_type = _clean_cell(row[5])
    race = _clean_cell(row[6])
    gender = _clean_cell(row[7])
    filename = _clean_cell(row[8])
    raw_folder_path = _clean_cell(row[9])

    if sample_type not in LABELS_BY_TYPE:
        raise ValueError(f"Unsupported FakeAVCeleb type: {sample_type}")
    if not filename:
        raise ValueError(f"FakeAVCeleb metadata row is missing filename: {row}")

    folder_path = _strip_dataset_prefix(raw_folder_path)
    video_path = (dataset_root / folder_path / filename).resolve()

    return ManifestSample(
        sample_id=_sample_id(
            source=source,
            category=category,
            filename=filename,
            folder_path=folder_path,
            method=method,
        ),
        dataset_name="FakeAVCeleb",
        source=source,
        target1=target1,
        target2=target2,
        method=method,
        category=category,
        type=sample_type,
        race=race,
        gender=gender,
        filename=filename,
        folder_path=folder_path,
        video_path=str(video_path),
        audio_path=str(video_path),
        labels=dict(LABELS_BY_TYPE[sample_type]),
        label_name=sample_type,
        split="test",
    )


def build_manifest(
    *,
    dataset_root: str | Path,
    metadata_csv: str | Path,
) -> list[ManifestSample]:
    root = Path(dataset_root).expanduser().resolve()
    csv_path = Path(metadata_csv).expanduser().resolve()
    samples: list[ManifestSample] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row_index, row in enumerate(reader):
            if row_index == 0:
                continue
            if not row or all(not cell.strip() for cell in row):
                continue
            samples.append(_row_to_sample(root, row))
    return samples


def summarize_manifest(samples: list[ManifestSample]) -> dict[str, Any]:
    categories = Counter(sample.category for sample in samples)
    types = Counter(sample.type for sample in samples)
    return {
        "dataset_name": "FakeAVCeleb",
        "dataset_count": len(types),
        "sample_count": len(samples),
        "category_counts": dict(sorted(categories.items())),
        "type_counts": dict(sorted(types.items())),
        "label_counts": dict(sorted(types.items())),
    }


def write_manifest_jsonl(samples: list[ManifestSample], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for sample in samples:
            jsonl_file.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    return path
