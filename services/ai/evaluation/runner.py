from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from services.ai.evaluation.config import EvalConfig, REPO_ROOT
from services.ai.evaluation.manifest import LABELS_BY_TYPE, ManifestSample, build_manifest, write_manifest_jsonl
from services.ai.evaluation.metrics import read_prediction_jsonl, write_metrics
from services.ai.evaluation.predictions import PredictionWriter, build_prediction_records


def _new_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    return output_root / f"{timestamp}_{uuid4().hex[:8]}"


def _run_video_pipeline(sample: ManifestSample, run_dir: Path) -> dict[str, Any]:
    from services.ai.pipelines.video_stage1.detect import run_video_stage1_detection
    from services.ai.pipelines.video_stage1.preprocess import run_video_stage1_preprocess

    storage_root = run_dir / "artifacts" / sample.sample_id / "video" / "jobs"
    run_video_stage1_preprocess(
        sample.video_path,
        job_id=sample.sample_id,
        storage_root=storage_root,
    )
    preprocessing_json_path = storage_root / sample.sample_id / "metadata" / "preprocessing.json"
    detection = run_video_stage1_detection(str(preprocessing_json_path))
    return detection


def _run_audio_pipeline(sample: ManifestSample, run_dir: Path) -> dict[str, Any]:
    from services.ai.audio_pipeline.audio_stage1 import run_audio_stage1

    output_dir = run_dir / "artifacts" / sample.sample_id / "audio"
    kwargs: dict[str, Any] = {
        "input_path": sample.audio_path,
        "output_dir": output_dir,
        "request_id": sample.sample_id,
    }
    python_executable = os.getenv("VERIFAKE_AUDIO_PYTHON")
    if python_executable:
        kwargs["python_executable"] = python_executable
    device = os.getenv("VERIFAKE_AUDIO_DEVICE")
    if device:
        kwargs["device"] = device
    return run_audio_stage1(**kwargs)


def _read_predictions(writer: PredictionWriter) -> dict[str, list[dict[str, Any]]]:
    required_fields = {"sample_id", "category", "label", "fake_score"}

    def complete_rows(path: Path) -> list[dict[str, Any]]:
        rows_by_sample_id: dict[str, dict[str, Any]] = {}
        for row in read_prediction_jsonl(path):
            if not required_fields.issubset(row):
                continue
            rows_by_sample_id[str(row["sample_id"])] = row
        return list(rows_by_sample_id.values())

    return {
        "video": complete_rows(writer.paths.video_jsonl),
        "audio": complete_rows(writer.paths.audio_jsonl),
        "fusion": complete_rows(writer.paths.fusion_jsonl),
    }


def _filter_predictions_by_type(
    predictions_by_modality: dict[str, list[dict[str, Any]]],
    sample_type: str,
) -> dict[str, list[dict[str, Any]]]:
    return {
        modality: [record for record in records if record.get("type") == sample_type]
        for modality, records in predictions_by_modality.items()
    }


def _export_final_metrics(metrics_paths: dict[str, Path], results_root: str | Path) -> dict[str, Path]:
    root = Path(results_root).expanduser().resolve()
    exported_paths: dict[str, Path] = {}
    for sample_type, metrics_path in metrics_paths.items():
        output_path = root / sample_type / f"{sample_type}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(metrics_path.read_text(encoding="utf-8"), encoding="utf-8")
        exported_paths[sample_type] = output_path
    return exported_paths


def _run_enabled_pipelines(
    sample: ManifestSample,
    run_dir: Path,
    *,
    video_enabled: bool,
    audio_enabled: bool,
    writer: PredictionWriter,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    video_result: dict[str, Any] | None = None
    audio_result: dict[str, Any] | None = None

    if video_enabled and audio_enabled:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                "video": executor.submit(_run_video_pipeline, sample, run_dir),
                "audio": executor.submit(_run_audio_pipeline, sample, run_dir),
            }
            for modality, future in futures.items():
                try:
                    result = future.result()
                except Exception as exc:
                    writer.write_failure(sample, modality=modality, error=str(exc))
                    continue
                if modality == "video":
                    video_result = result
                else:
                    audio_result = result
        return video_result, audio_result

    if video_enabled:
        try:
            video_result = _run_video_pipeline(sample, run_dir)
        except Exception as exc:
            writer.write_failure(sample, modality="video", error=str(exc))
    if audio_enabled:
        try:
            audio_result = _run_audio_pipeline(sample, run_dir)
        except Exception as exc:
            writer.write_failure(sample, modality="audio", error=str(exc))
    return video_result, audio_result


def _resolve_run_dir(config: EvalConfig, run_dir: str | Path | None) -> Path:
    resolved = Path(run_dir).expanduser().resolve() if run_dir is not None else _new_run_dir(config.output_root)
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved
    raise ValueError(f"Evaluation run_dir must be outside the repository: {REPO_ROOT}")


def run_dataset_evaluation(
    config: EvalConfig,
    limit: int | None = None,
    run_dir: str | Path | None = None,
    sample_type: str | None = None,
    results_root: str | Path | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
    overfit_thresholds: bool = False,
) -> dict[str, Any]:
    resolved_run_dir = _resolve_run_dir(config, run_dir)
    resolved_run_dir.mkdir(parents=True, exist_ok=run_dir is not None)

    if sample_type is not None and sample_type not in LABELS_BY_TYPE:
        raise ValueError(f"Unsupported FakeAVCeleb type: {sample_type}")

    samples = build_manifest(dataset_root=config.dataset_root, metadata_csv=config.metadata_csv)
    if sample_type is not None:
        samples = [sample for sample in samples if sample.type == sample_type]
    if limit is not None:
        samples = samples[:limit]
    if num_shards is not None or shard_index is not None:
        if num_shards is None or shard_index is None:
            raise ValueError("Both shard_index and num_shards must be provided for sharded evaluation")
        if num_shards <= 0:
            raise ValueError("num_shards must be greater than zero")
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError("shard_index must be between 0 and num_shards - 1")
        samples = [
            sample
            for sample_index, sample in enumerate(samples)
            if sample_index % num_shards == shard_index
        ]

    manifest_name = (
        f"manifest_shard_{shard_index}_of_{num_shards}.jsonl"
        if num_shards is not None
        else "manifest.jsonl"
    )
    manifest_jsonl = write_manifest_jsonl(samples, resolved_run_dir / manifest_name)
    shard_suffix = (
        f"_shard_{shard_index}_of_{num_shards}"
        if num_shards is not None
        else ""
    )
    writer = PredictionWriter(resolved_run_dir, suffix=shard_suffix)
    processed_sample_ids = writer.processed_sample_ids()

    for sample in samples:
        if sample.sample_id in processed_sample_ids:
            continue

        video_result, audio_result = _run_enabled_pipelines(
            sample,
            resolved_run_dir,
            video_enabled=config.video_enabled,
            audio_enabled=config.audio_enabled,
            writer=writer,
        )

        records = build_prediction_records(
            sample,
            video_result=video_result,
            audio_result=audio_result,
        )
        if records.video is not None or records.audio is not None or records.fusion is not None:
            writer.write_records(records)

    expected_dataset_names = [sample_type] if sample_type is not None else sorted({sample.type for sample in samples})
    predictions_by_modality = _read_predictions(writer)
    if sample_type is not None:
        predictions_by_modality = _filter_predictions_by_type(predictions_by_modality, sample_type)
    metrics_paths = write_metrics(
        run_dir=resolved_run_dir,
        predictions_by_modality=predictions_by_modality,
        expected_dataset_names=expected_dataset_names,
        overfit_thresholds=overfit_thresholds,
    )
    result = {
        "run_dir": str(resolved_run_dir),
        "manifest_jsonl": str(manifest_jsonl),
        "predictions": writer.paths.to_dict(),
        "metrics": {dataset: str(path) for dataset, path in metrics_paths.items()},
    }
    if results_root is not None:
        final_metrics_paths = _export_final_metrics(metrics_paths, results_root)
        result["final_metrics"] = {
            dataset: str(path) for dataset, path in final_metrics_paths.items()
        }
    return result
