from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from services.ai.evaluation.metrics import write_metrics
from services.ai.evaluation.predictions import classify_fusion_triage, fuse_fake_scores


DATASET_TYPES = [
    "FakeVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "RealVideo-FakeAudio",
    "RealVideo-RealAudio",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prediction_row_from_combined(
    row: dict[str, Any],
    *,
    modality: str,
    label_key: str,
    score: float,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prediction = {
        "sample_id": row["sample_id"],
        "dataset_name": row["dataset_name"],
        "category": row["category"],
        "type": row["type"],
        "label_name": row["label_name"],
        "split": row["split"],
        "filename": row["filename"],
        "video_path": row["video_path"],
        "audio_path": row["audio_path"],
        "modality": modality,
        "label": int(row[label_key]),
        "fake_score": float(score),
    }
    if extra is not None:
        prediction.update(dict(extra))
    return prediction


def _merge_shard(run_dir: Path, *, shard_index: int, num_shards: int) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    suffix = f"_shard_{shard_index}_of_{num_shards}"
    video_rows = _read_jsonl(run_dir / f"predictions_video{suffix}.jsonl")
    audio_rows = _read_jsonl(run_dir / f"predictions_audio{suffix}.jsonl")
    combined_rows = _read_jsonl(run_dir / f"predictions_combined{suffix}.jsonl")

    video_by_sample = {str(row["sample_id"]): row for row in video_rows}
    audio_by_sample = {str(row["sample_id"]): row for row in audio_rows}

    merged_combined: list[dict[str, Any]] = []
    fusion_rows: list[dict[str, Any]] = []
    for row in combined_rows:
        sample_id = str(row["sample_id"])
        merged = dict(row)
        video_score = video_by_sample.get(sample_id, {}).get("fake_score")
        audio_score = audio_by_sample.get(sample_id, {}).get("fake_score")
        merged["video_fake_score"] = float(video_score) if video_score is not None else None
        merged["audio_fake_score"] = float(audio_score) if audio_score is not None else None
        if video_score is not None and audio_score is not None:
            video_score = float(video_score)
            audio_score = float(audio_score)
            fusion_score = fuse_fake_scores(video_score, audio_score)
            fusion_triage_label = classify_fusion_triage(video_score, audio_score)
            merged["fusion_fake_score"] = fusion_score
            merged["fusion_triage_label"] = fusion_triage_label
            fusion_rows.append(
                _prediction_row_from_combined(
                    merged,
                    modality="fusion",
                    label_key="fusion_label",
                    score=fusion_score,
                    extra={
                        "triage_label": fusion_triage_label,
                    },
                )
            )
        else:
            merged["fusion_fake_score"] = None
            merged["fusion_triage_label"] = None
        merged_combined.append(merged)

    _write_jsonl(run_dir / f"predictions_fusion{suffix}.jsonl", fusion_rows)
    _write_jsonl(run_dir / f"predictions_combined{suffix}.jsonl", merged_combined)
    _write_csv(run_dir / f"predictions_combined{suffix}.csv", merged_combined)
    return video_rows, audio_rows, fusion_rows, merged_combined


def aggregate(
    run_dir: Path,
    results_root: Path,
    *,
    num_shards: int,
    thresholds: dict[str, float] | None = None,
    overfit_thresholds: bool = False,
) -> None:
    all_video: list[dict[str, Any]] = []
    all_audio: list[dict[str, Any]] = []
    all_fusion: list[dict[str, Any]] = []
    all_combined: list[dict[str, Any]] = []

    for shard_index in range(num_shards):
        video_rows, audio_rows, fusion_rows, combined_rows = _merge_shard(
            run_dir,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        all_video.extend(video_rows)
        all_audio.extend(audio_rows)
        all_fusion.extend(fusion_rows)
        all_combined.extend(combined_rows)

    _write_jsonl(run_dir / "predictions_audio.jsonl", all_audio)
    _write_jsonl(run_dir / "predictions_fusion.jsonl", all_fusion)
    _write_jsonl(run_dir / "predictions_combined.jsonl", all_combined)
    _write_csv(run_dir / "predictions_combined.csv", all_combined)

    metric_paths = write_metrics(
        run_dir=run_dir,
        predictions_by_modality={
            "video": all_video,
            "audio": all_audio,
            "fusion": all_fusion,
        },
        thresholds=thresholds,
        expected_dataset_names=DATASET_TYPES,
        overfit_thresholds=overfit_thresholds,
    )
    for dataset_type, metrics_path in metric_paths.items():
        export_path = results_root / dataset_type / f"{dataset_type}.json"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(metrics_path.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate audio-only rerun shards with existing video predictions.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--video-threshold", type=float, default=0.5)
    parser.add_argument("--audio-threshold", type=float, default=0.5)
    parser.add_argument("--fusion-threshold", type=float, default=0.5)
    parser.add_argument(
        "--overfit-thresholds",
        action="store_true",
        help="Opt in to diagnostic in-sample threshold calibration; not valid for benchmark reporting",
    )
    args = parser.parse_args()

    aggregate(
        Path(args.run_dir).expanduser().resolve(),
        Path(args.results_root).expanduser().resolve(),
        num_shards=args.num_shards,
        thresholds={
            "video": args.video_threshold,
            "audio": args.audio_threshold,
            "fusion": args.fusion_threshold,
        },
        overfit_thresholds=args.overfit_thresholds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
