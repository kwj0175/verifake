from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.ai.evaluation.metrics import compute_auc_roc, compute_metrics, write_metrics


def prediction(
    sample_id: str,
    label: int,
    score: float,
    category: str = "A",
    sample_type: str = "FakeVideo-FakeAudio",
    triage_label: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample_id,
        "dataset_name": "FakeAVCeleb",
        "category": category,
        "type": sample_type,
        "modality": "video",
        "label": label,
        "fake_score": score,
    }
    if triage_label is not None:
        row["triage_label"] = triage_label
    return row


def test_compute_auc_roc_handles_ties() -> None:
    auc, reason = compute_auc_roc(
        labels=[0, 0, 0, 0, 1, 1],
        scores=[0.1, 0.2, 0.8, 0.8, 0.8, 0.9],
    )

    assert auc == 0.875
    assert reason is None


def test_compute_auc_roc_returns_null_reason_for_one_class() -> None:
    auc, reason = compute_auc_roc(labels=[1, 1], scores=[0.8, 0.9])

    assert auc is None
    assert reason == "auc_roc requires both positive and negative labels"


def test_compute_metrics_counts_accuracy_and_one_class_auc_reason() -> None:
    metrics = compute_metrics([prediction("a", 1, 0.7), prediction("b", 1, 0.2)])

    assert metrics["total_count"] == 2
    assert metrics["correct_count"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["auc_roc"] is None
    assert metrics["auc_roc_reason"] == "auc_roc requires both positive and negative labels"
    assert metrics["mean_fake_score"] == 0.44999999999999996


def test_write_metrics_saves_dataset_metrics_and_errors(tmp_path: Path) -> None:
    paths = write_metrics(
        run_dir=tmp_path,
        predictions_by_modality={
            "video": [
                prediction("tp", 1, 0.9),
                prediction("fp", 0, 0.8),
                prediction("fn", 1, 0.1),
                prediction("tn", 0, 0.2),
            ]
        },
    )

    metrics = json.loads(paths["FakeVideo-FakeAudio"].read_text(encoding="utf-8"))
    false_positive = (tmp_path / "errors" / "FakeVideo-FakeAudio_false_positive.jsonl").read_text(encoding="utf-8")
    false_negative = (tmp_path / "errors" / "FakeVideo-FakeAudio_false_negative.jsonl").read_text(encoding="utf-8")

    assert paths["FakeVideo-FakeAudio"].name == "FakeVideo-FakeAudio_metrics.json"
    assert metrics["dataset_name"] == "FakeVideo-FakeAudio"
    assert set(metrics["results"]) == {"video", "audio", "fusion"}
    assert metrics["results"]["video"]["accuracy"] == 0.5
    assert "auc_roc_reason" not in metrics["results"]["video"]
    assert metrics["results"]["audio"]["auc_roc"] is None
    assert metrics["results"]["audio"]["auc_roc_reason"] == "no predictions"
    assert '"sample_id": "fp"' in false_positive
    assert '"sample_id": "fn"' in false_negative


def test_write_metrics_can_use_modality_specific_thresholds(tmp_path: Path) -> None:
    paths = write_metrics(
        run_dir=tmp_path,
        predictions_by_modality={
            "video": [
                prediction("real", 0, 0.55),
                prediction("fake", 1, 0.65),
            ],
            "audio": [
                prediction("audio-real", 0, 0.55),
                prediction("audio-fake", 1, 0.65),
            ],
        },
        thresholds={"video": 0.6, "audio": 0.5},
    )

    metrics = json.loads(paths["FakeVideo-FakeAudio"].read_text(encoding="utf-8"))

    assert metrics["results"]["video"]["accuracy"] == 1.0
    assert metrics["results"]["audio"]["accuracy"] == 0.5


def test_write_metrics_can_overfit_thresholds_with_calibration_metadata(tmp_path: Path) -> None:
    paths = write_metrics(
        run_dir=tmp_path,
        predictions_by_modality={
            "video": [
                prediction("real", 0, 0.55),
                prediction("fake", 1, 0.65),
            ],
        },
        overfit_thresholds=True,
    )

    metrics = json.loads(paths["FakeVideo-FakeAudio"].read_text(encoding="utf-8"))

    assert metrics["results"]["video"]["accuracy"] == 1.0
    assert metrics["calibration"]["video"] == {
        "mode": "overfit_in_sample",
        "reporting_scope": "diagnostic_only",
        "benchmark_reporting_valid": False,
        "warning": "In-sample threshold calibration is diagnostic only and is not valid for benchmark reporting.",
        "threshold": 0.65,
        "accuracy": 1.0,
        "correct_count": 2,
        "total_count": 2,
    }
    assert "calibration" not in write_metrics(
        run_dir=tmp_path / "normal",
        predictions_by_modality={
            "video": [
                prediction("real", 0, 0.55),
                prediction("fake", 1, 0.65),
            ],
        },
    )["FakeVideo-FakeAudio"].read_text(encoding="utf-8")


def test_write_metrics_overfit_thresholds_are_per_dataset_category(tmp_path: Path) -> None:
    paths = write_metrics(
        run_dir=tmp_path,
        predictions_by_modality={
            "video": [
                prediction("a-real", 0, 0.55, sample_type="FakeVideo-FakeAudio"),
                prediction("a-fake", 1, 0.65, sample_type="FakeVideo-FakeAudio"),
                prediction("b-real", 0, 0.35, sample_type="RealVideo-FakeAudio"),
                prediction("b-fake", 1, 0.45, sample_type="RealVideo-FakeAudio"),
            ],
        },
        overfit_thresholds=True,
    )

    fake_video_fake_audio = json.loads(paths["FakeVideo-FakeAudio"].read_text(encoding="utf-8"))
    real_video_fake_audio = json.loads(paths["RealVideo-FakeAudio"].read_text(encoding="utf-8"))

    assert fake_video_fake_audio["calibration"]["video"]["threshold"] == 0.65
    assert real_video_fake_audio["calibration"]["video"]["threshold"] == 0.45
    assert fake_video_fake_audio["results"]["video"]["accuracy"] == 1.0
    assert real_video_fake_audio["results"]["video"]["accuracy"] == 1.0


def test_compute_metrics_adds_compact_review_summary_without_replacing_accuracy() -> None:
    metrics = compute_metrics(
        [
            prediction("review-real", 0, 0.9, triage_label="needs_review"),
            prediction("auto-fake", 1, 0.8, triage_label="fake"),
            prediction("auto-missed-fake", 1, 0.4, triage_label="real"),
            prediction("auto-real", 0, 0.2, triage_label="real"),
        ]
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["review_summary"] == "needs_review=1/4, auto_accuracy=0.6667"


def test_write_metrics_adds_compact_review_summary_only(tmp_path: Path) -> None:
    paths = write_metrics(
        run_dir=tmp_path,
        predictions_by_modality={
            "fusion": [
                prediction(
                    "review-real",
                    0,
                    0.9,
                    triage_label="needs_review",
                )
            ]
        },
    )

    metrics = json.loads(paths["FakeVideo-FakeAudio"].read_text(encoding="utf-8"))

    assert metrics["results"]["fusion"]["review_summary"] == "needs_review=1/1, auto_accuracy=null"
    assert not (tmp_path / "errors" / "FakeVideo-FakeAudio_needs_review.jsonl").exists()


def test_write_metrics_creates_one_file_per_type_when_category_differs(tmp_path: Path) -> None:
    paths = write_metrics(
        run_dir=tmp_path,
        predictions_by_modality={
            "video": [
                prediction("fake-a", 1, 0.9, category="A", sample_type="FakeVideo-FakeAudio"),
                prediction("real-a", 0, 0.1, category="D", sample_type="RealVideo-RealAudio"),
            ],
            "audio": [prediction("fake-a", 1, 0.8, category="A", sample_type="FakeVideo-FakeAudio")],
            "fusion": [prediction("fake-a", 1, 0.85, category="A", sample_type="FakeVideo-FakeAudio")],
        },
    )

    assert {"FakeVideo-FakeAudio", "RealVideo-RealAudio"}.issubset(paths)
    fake_metrics = json.loads(paths["FakeVideo-FakeAudio"].read_text(encoding="utf-8"))
    real_metrics = json.loads(paths["RealVideo-RealAudio"].read_text(encoding="utf-8"))

    assert fake_metrics["dataset_name"] == "FakeVideo-FakeAudio"
    assert real_metrics["dataset_name"] == "RealVideo-RealAudio"
    assert fake_metrics["results"]["video"]["total_count"] == 1
    assert real_metrics["results"]["video"]["total_count"] == 1


def test_write_metrics_emits_expected_types_with_zero_counts(tmp_path: Path) -> None:
    paths = write_metrics(
        run_dir=tmp_path,
        predictions_by_modality={
            "video": [prediction("fake-a", 1, 0.9, category="A", sample_type="FakeVideo-FakeAudio")],
        },
        expected_dataset_names=["FakeVideo-FakeAudio", "FakeVideo-RealAudio"],
    )

    missing_metrics = json.loads(paths["FakeVideo-RealAudio"].read_text(encoding="utf-8"))
    assert missing_metrics["dataset_name"] == "FakeVideo-RealAudio"
    assert missing_metrics["results"]["video"] == {
        "total_count": 0,
        "correct_count": 0,
        "accuracy": None,
        "auc_roc": None,
        "mean_fake_score": None,
        "auc_roc_reason": "no predictions",
    }
    assert missing_metrics["results"]["audio"]["total_count"] == 0
    assert missing_metrics["results"]["fusion"]["total_count"] == 0


def test_overfit_thresholds_are_marked_diagnostic_and_keep_reference_tie_break(tmp_path: Path) -> None:
    paths = write_metrics(
        run_dir=tmp_path,
        predictions_by_modality={
            "video": [
                prediction("real", 0, 0.2),
                prediction("fake", 1, 0.9),
            ],
        },
        overfit_thresholds=True,
    )

    metrics = json.loads(paths["FakeVideo-FakeAudio"].read_text(encoding="utf-8"))
    calibration = metrics["calibration"]["video"]

    assert calibration["threshold"] == 0.5
    assert calibration["accuracy"] == 1.0
    assert calibration["reporting_scope"] == "diagnostic_only"
    assert calibration["benchmark_reporting_valid"] is False
    assert "not valid for benchmark reporting" in calibration["warning"]


def test_overfit_thresholds_keep_inclusive_score_boundary(tmp_path: Path) -> None:
    paths = write_metrics(
        run_dir=tmp_path,
        predictions_by_modality={
            "video": [
                prediction("real", 0, 0.49),
                prediction("fake-at-threshold", 1, 0.5),
                prediction("fake-above-threshold", 1, 0.8),
            ],
        },
        overfit_thresholds=True,
    )

    metrics = json.loads(paths["FakeVideo-FakeAudio"].read_text(encoding="utf-8"))

    assert metrics["calibration"]["video"]["threshold"] == 0.5
    assert metrics["results"]["video"]["accuracy"] == 1.0


def test_invalid_thresholds_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="threshold must be between"):
        write_metrics(
            run_dir=tmp_path,
            predictions_by_modality={"video": [prediction("sample", 1, 0.9)]},
            thresholds={"video": 1.1},
        )
