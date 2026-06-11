from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ONE_CLASS_AUC_REASON = "auc_roc requires both positive and negative labels"
OVERFIT_CALIBRATION_WARNING = (
    "In-sample threshold calibration is diagnostic only and is not valid for benchmark reporting."
)


def _validate_threshold(threshold: float) -> float:
    value = float(threshold)
    if not 0.0 <= value <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")
    return value


def compute_auc_roc(labels: list[int], scores: list[float]) -> tuple[float | None, str | None]:
    positives = sum(1 for label in labels if label == 1)
    negatives = sum(1 for label in labels if label == 0)
    if positives == 0 or negatives == 0:
        return None, ONE_CLASS_AUC_REASON
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")

    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ranked):
        tie_end = index + 1
        while tie_end < len(ranked) and ranked[tie_end][1] == ranked[index][1]:
            tie_end += 1
        average_rank = (index + 1 + tie_end) / 2.0
        for rank_index in range(index, tie_end):
            original_index = ranked[rank_index][0]
            ranks[original_index] = average_rank
        index = tie_end

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc), None


def _triage_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    triage_rows = [row for row in rows if row.get("triage_label") is not None]
    if not triage_rows:
        return {}

    review_count = sum(
        1 for row in triage_rows if row.get("triage_label") == "needs_review"
    )
    auto_rows = [
        row for row in triage_rows if row.get("triage_label") in {"real", "fake"}
    ]
    auto_correct_count = sum(
        1
        for row in auto_rows
        if (1 if row.get("triage_label") == "fake" else 0) == int(row["label"])
    )
    auto_accuracy = (
        f"{auto_correct_count / len(auto_rows):.4f}"
        if auto_rows
        else "null"
    )
    return {
        "review_summary": (
            f"needs_review={review_count}/{len(triage_rows)}, "
            f"auto_accuracy={auto_accuracy}"
        )
    }


def compute_metrics(records: Iterable[dict[str, Any]], threshold: float = 0.5) -> dict[str, Any]:
    threshold = _validate_threshold(threshold)
    rows = list(records)
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["fake_score"]) for row in rows]
    predicted = [1 if score >= threshold else 0 for score in scores]
    correct_count = sum(1 for expected, actual in zip(labels, predicted) if expected == actual)
    auc, auc_reason = compute_auc_roc(labels, scores) if rows else (None, "no predictions")
    total_count = len(rows)
    result: dict[str, Any] = {
        "total_count": total_count,
        "correct_count": correct_count,
        "accuracy": correct_count / total_count if total_count else None,
        "auc_roc": auc,
        "mean_fake_score": float(sum(scores) / total_count) if total_count else None,
    }
    if auc_reason is not None:
        result["auc_roc_reason"] = auc_reason
    result.update(_triage_metrics(rows))
    return result


def _accuracy_at_threshold(records: list[dict[str, Any]], threshold: float) -> tuple[float | None, int, int]:
    predicted = [1 if float(row["fake_score"]) >= threshold else 0 for row in records]
    correct_count = sum(
        1
        for row, predicted_label in zip(records, predicted)
        if int(row["label"]) == predicted_label
    )
    total_count = len(records)
    return (correct_count / total_count if total_count else None, correct_count, total_count)


def _overfit_threshold_calibration(
    records: list[dict[str, Any]],
    *,
    reference_threshold: float,
) -> dict[str, Any]:
    candidates = {
        0.0,
        1.0,
        reference_threshold,
        *(float(row["fake_score"]) for row in records),
    }
    best_threshold = reference_threshold
    best_accuracy, best_correct_count, total_count = _accuracy_at_threshold(records, best_threshold)
    for candidate in sorted(candidates):
        accuracy, correct_count, _ = _accuracy_at_threshold(records, candidate)
        if accuracy is None:
            continue
        if best_accuracy is None or accuracy > best_accuracy:
            best_threshold = candidate
            best_accuracy = accuracy
            best_correct_count = correct_count
            continue
        if accuracy == best_accuracy and abs(candidate - reference_threshold) < abs(best_threshold - reference_threshold):
            best_threshold = candidate
            best_correct_count = correct_count

    return {
        "mode": "overfit_in_sample",
        "reporting_scope": "diagnostic_only",
        "benchmark_reporting_valid": False,
        "warning": OVERFIT_CALIBRATION_WARNING,
        "threshold": best_threshold,
        "accuracy": best_accuracy,
        "correct_count": best_correct_count,
        "total_count": total_count,
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _error_rows(records: list[dict[str, Any]], threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    for row in records:
        predicted_label = 1 if float(row["fake_score"]) >= threshold else 0
        enriched = dict(row)
        enriched["predicted_label"] = predicted_label
        if int(row["label"]) == 0 and predicted_label == 1:
            false_positives.append(enriched)
        if int(row["label"]) == 1 and predicted_label == 0:
            false_negatives.append(enriched)
    return false_positives, false_negatives


def write_metrics(
    *,
    run_dir: str | Path,
    predictions_by_modality: dict[str, list[dict[str, Any]]],
    threshold: float = 0.5,
    thresholds: Mapping[str, float] | None = None,
    expected_dataset_names: Iterable[str] | None = None,
    overfit_thresholds: bool = False,
) -> dict[str, Path]:
    root = Path(run_dir)
    metrics_dir = root / "metrics"
    errors_dir = root / "errors"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)
    default_threshold = _validate_threshold(threshold)
    modality_thresholds = {
        str(modality): _validate_threshold(value)
        for modality, value in (thresholds or {}).items()
    }

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for modality, records in predictions_by_modality.items():
        for record in records:
            grouped[str(record.get("type") or record.get("category") or record["dataset_name"])][modality].append(record)

    if expected_dataset_names is not None:
        for dataset_name in expected_dataset_names:
            grouped[str(dataset_name)]

    output_paths: dict[str, Path] = {}
    for category_name, modality_records in grouped.items():
        payload: dict[str, Any] = {
            "dataset_name": category_name,
            "results": {},
        }
        if overfit_thresholds:
            payload["calibration"] = {}
        false_positive_path = errors_dir / f"{category_name}_false_positive.jsonl"
        false_negative_path = errors_dir / f"{category_name}_false_negative.jsonl"
        false_positive_path.write_text("", encoding="utf-8")
        false_negative_path.write_text("", encoding="utf-8")

        for modality in ("video", "audio", "fusion"):
            records = modality_records.get(modality, [])
            modality_threshold = modality_thresholds.get(modality, default_threshold)
            if overfit_thresholds:
                calibration = _overfit_threshold_calibration(
                    records,
                    reference_threshold=modality_threshold,
                )
                payload["calibration"][modality] = calibration
                modality_threshold = float(calibration["threshold"])
            payload["results"][modality] = compute_metrics(
                records,
                threshold=modality_threshold,
            )

            false_positives, false_negatives = _error_rows(
                records,
                modality_threshold,
            )
            for row in false_positives:
                _append_jsonl(false_positive_path, row)
            for row in false_negatives:
                _append_jsonl(false_negative_path, row)

        output_path = metrics_dir / f"{category_name}_metrics.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        output_paths[category_name] = output_path
    return output_paths


def read_prediction_jsonl(path: str | Path) -> list[dict[str, Any]]:
    prediction_path = Path(path)
    if not prediction_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with prediction_path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows
