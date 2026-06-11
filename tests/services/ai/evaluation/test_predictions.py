from __future__ import annotations

import csv
import json
from pathlib import Path

from services.ai.evaluation.manifest import ManifestSample
from services.ai.evaluation.predictions import (
    PredictionWriter,
    build_prediction_records,
    classify_fusion_triage,
    extract_audio_fake_score,
    extract_video_fake_score,
    load_existing_sample_ids,
)


def sample(sample_id: str = "sample-1") -> ManifestSample:
    return ManifestSample(
        sample_id=sample_id,
        dataset_name="FakeAVCeleb",
        source="src",
        target1="t1",
        target2="t2",
        method="method",
        category="A",
        type="FakeVideo-FakeAudio",
        race="Asian",
        gender="Female",
        filename="a.mp4",
        folder_path="FakeVideo-FakeAudio/Asian/Female",
        video_path="/dataset/a.mp4",
        audio_path="/dataset/a.mp4",
        labels={"video_label": 1, "audio_label": 1, "fusion_label": 1},
        label_name="FakeVideo-FakeAudio",
        split="test",
    )


def test_extract_scores_from_existing_pipeline_schemas() -> None:
    assert extract_video_fake_score({"detection": {"video_score": {"final_fake_score": 0.75}}}) == 0.75
    assert extract_video_fake_score({"video_score": {"final_fake_score": 0.65}}) == 0.65
    assert extract_audio_fake_score({"audio_fake_score": 0.2, "audio_fake_prob_like": 0.8}) == 0.2
    assert extract_audio_fake_score({"audio_fake_prob_like": 0.8}) == 0.8
    assert extract_audio_fake_score({"audio_fake_prob_like": 0.0, "scored_window_count": 0}) is None
    assert extract_audio_fake_score({"audio_inference": {"scored_window_count": 0}, "audio_fake_prob_like": 0.0}) is None


def test_build_prediction_records_maps_scores_and_fusion_average() -> None:
    records = build_prediction_records(
        sample(),
        video_result={"detection": {"video_score": {"final_fake_score": 0.9}}},
        audio_result={"audio_fake_prob_like": 0.3},
    )

    assert records.video is not None
    assert records.audio is not None
    assert records.fusion is not None
    assert records.video["fake_score"] == 0.9
    assert records.video["category"] == "A"
    assert records.video["type"] == "FakeVideo-FakeAudio"
    assert records.video["audio_path"] == "/dataset/a.mp4"
    assert records.audio["fake_score"] == 0.3
    assert records.fusion["fake_score"] == 0.6
    assert records.fusion["label"] == 1
    assert records.fusion["triage_label"] == "fake"
    assert records.combined["fusion_triage_label"] == "fake"


def test_fusion_triage_marks_video_audio_conflict_for_review() -> None:
    records = build_prediction_records(
        sample(),
        video_result={"detection": {"video_score": {"final_fake_score": 0.9}}},
        audio_result={"audio_fake_prob_like": 0.01},
    )

    assert classify_fusion_triage(0.9, 0.01) == "needs_review"
    assert classify_fusion_triage(0.4, 0.01) == "needs_review"
    assert records.fusion is not None
    assert records.fusion["fake_score"] == 0.455
    assert records.fusion["triage_label"] == "needs_review"
    assert records.combined["fusion_triage_label"] == "needs_review"


def test_build_prediction_records_excludes_unscored_audio_from_metrics() -> None:
    records = build_prediction_records(
        sample(),
        video_result={"detection": {"video_score": {"final_fake_score": 0.9}}},
        audio_result={
            "audio_fake_prob_like": 0.0,
            "scored_window_count": 0,
            "failed_window_count": 2,
            "skipped_window_count": 0,
            "audio_model_error": "AttributeError: module 'numpy' has no attribute 'float'",
        },
    )

    assert records.audio is None
    assert records.fusion is None
    assert records.combined["audio_fake_score"] is None
    assert records.combined["audio_scored_window_count"] == 0
    assert records.combined["audio_failed_window_count"] == 2
    assert records.combined["fusion_triage_label"] is None
    assert "numpy" in records.combined["audio_model_error"]


def test_prediction_writer_writes_jsonl_csv_failures_and_reads_resume_ids(tmp_path: Path) -> None:
    writer = PredictionWriter(tmp_path)
    records = build_prediction_records(
        sample(),
        video_result={"detection": {"video_score": {"final_fake_score": 0.9}}},
        audio_result={"audio_fake_prob_like": 0.3},
    )

    writer.write_records(records)
    writer.write_failure(sample(), modality="video", error="boom")

    video_rows = [json.loads(line) for line in writer.paths.video_jsonl.read_text(encoding="utf-8").splitlines()]
    failure_rows = [json.loads(line) for line in writer.paths.failed_jsonl.read_text(encoding="utf-8").splitlines()]
    with writer.paths.combined_csv.open("r", encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))

    assert video_rows[0]["sample_id"] == "sample-1"
    assert csv_rows[0]["video_fake_score"] == "0.9"
    assert failure_rows[0]["error"] == "boom"
    assert load_existing_sample_ids([writer.paths.video_jsonl]) == {"sample-1"}


def test_fusion_triage_covers_priority_and_threshold_boundaries() -> None:
    assert classify_fusion_triage(0.9, 0.6) == "fake"
    assert classify_fusion_triage(0.9, 0.2) == "fake"
    assert classify_fusion_triage(0.2, 0.01) == "real"
    assert classify_fusion_triage(0.3, 0.1) == "needs_review"
    assert classify_fusion_triage(0.29, 0.1) == "real"
    assert classify_fusion_triage(0.9, 0.11) == "fake"
