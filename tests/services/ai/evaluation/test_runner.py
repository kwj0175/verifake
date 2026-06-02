from __future__ import annotations

import json
from pathlib import Path

from services.ai.evaluation.config import EvalConfig
from services.ai.evaluation.manifest import ManifestSample
from services.ai.evaluation.runner import _run_audio_pipeline, run_dataset_evaluation


def config(tmp_path: Path) -> EvalConfig:
    return EvalConfig(
        dataset_root=(tmp_path / "dataset").resolve(),
        metadata_csv=(tmp_path / "dataset" / "meta_data.csv").resolve(),
        output_root=(tmp_path / "external-output").resolve(),
        video_enabled=True,
        audio_enabled=True,
    )


def manifest_sample(
    sample_id: str,
    video_path: Path,
    *,
    category: str = "A",
    sample_type: str = "FakeVideo-FakeAudio",
) -> ManifestSample:
    return ManifestSample(
        sample_id=sample_id,
        dataset_name="FakeAVCeleb",
        source="src",
        target1="t1",
        target2="t2",
        method="method",
        category=category,
        type=sample_type,
        race="Asian",
        gender="Female",
        filename=video_path.name,
        folder_path="FakeVideo-FakeAudio/Asian/Female",
        video_path=str(video_path),
        audio_path=str(video_path),
        labels={"video_label": 1, "audio_label": 1, "fusion_label": 1},
        label_name=sample_type,
        split="test",
    )


def test_run_audio_pipeline_uses_env_python_and_device_when_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample = manifest_sample("sample-1", tmp_path / "sample.mp4")
    run_dir = tmp_path / "run"
    captured: dict[str, object] = {}

    def fake_run_audio_stage1(**kwargs):
        captured.update(kwargs)
        return {"audio_fake_prob_like": 0.6}

    monkeypatch.setenv("VERIFAKE_AUDIO_PYTHON", "C:/Python39/python.exe")
    monkeypatch.setenv("VERIFAKE_AUDIO_DEVICE", "cpu")
    monkeypatch.setattr("services.ai.audio_pipeline.audio_stage1.run_audio_stage1", fake_run_audio_stage1)

    _run_audio_pipeline(sample, run_dir)

    assert captured["input_path"] == sample.audio_path
    assert captured["output_dir"] == run_dir / "artifacts" / sample.sample_id / "audio"
    assert captured["request_id"] == sample.sample_id
    assert captured["python_executable"] == "C:/Python39/python.exe"
    assert captured["device"] == "cpu"


def test_run_audio_pipeline_leaves_optional_kwargs_unset_when_env_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample = manifest_sample("sample-2", tmp_path / "sample.mp4")
    run_dir = tmp_path / "run"
    captured: dict[str, object] = {}

    def fake_run_audio_stage1(**kwargs):
        captured.update(kwargs)
        return {"audio_fake_prob_like": 0.6}

    monkeypatch.delenv("VERIFAKE_AUDIO_PYTHON", raising=False)
    monkeypatch.delenv("VERIFAKE_AUDIO_DEVICE", raising=False)
    monkeypatch.setattr("services.ai.audio_pipeline.audio_stage1.run_audio_stage1", fake_run_audio_stage1)

    _run_audio_pipeline(sample, run_dir)

    assert "python_executable" not in captured
    assert "device" not in captured


def test_run_dataset_evaluation_writes_manifest_predictions_metrics_and_artifact_dirs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = config(tmp_path)
    video_path = cfg.dataset_root / "sample.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video")
    samples = [manifest_sample("sample-1", video_path)]
    captured: dict[str, Path] = {}

    monkeypatch.setattr("services.ai.evaluation.runner.build_manifest", lambda **kwargs: samples)

    def fake_video(sample: ManifestSample, run_dir: Path) -> dict[str, object]:
        captured["video_storage_root"] = run_dir / "artifacts" / sample.sample_id / "video" / "jobs"
        return {"video_score": {"final_fake_score": 0.8}}

    def fake_audio(sample: ManifestSample, run_dir: Path) -> dict[str, object]:
        captured["audio_output_dir"] = run_dir / "artifacts" / sample.sample_id / "audio"
        return {"audio_fake_prob_like": 0.6}

    monkeypatch.setattr("services.ai.evaluation.runner._run_video_pipeline", fake_video)
    monkeypatch.setattr("services.ai.evaluation.runner._run_audio_pipeline", fake_audio)

    result = run_dataset_evaluation(cfg, limit=1)

    run_dir = Path(result["run_dir"])
    manifest_rows = (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    video_rows = (run_dir / "predictions_video.jsonl").read_text(encoding="utf-8").splitlines()
    metrics = json.loads((run_dir / "metrics" / "FakeVideo-FakeAudio_metrics.json").read_text(encoding="utf-8"))

    assert run_dir.is_relative_to(cfg.output_root)
    assert captured["video_storage_root"] == run_dir / "artifacts" / "sample-1" / "video" / "jobs"
    assert captured["audio_output_dir"] == run_dir / "artifacts" / "sample-1" / "audio"
    assert len(manifest_rows) == 1
    assert json.loads(video_rows[0])["fake_score"] == 0.8
    assert metrics["results"]["fusion"]["accuracy"] == 1.0
    assert result["predictions"]["video_jsonl"].endswith("predictions_video.jsonl")


def test_run_dataset_evaluation_records_failures_and_continues(tmp_path: Path, monkeypatch) -> None:
    cfg = config(tmp_path)
    samples = [
        manifest_sample("bad", cfg.dataset_root / "bad.mp4", category="B", sample_type="FakeVideo-RealAudio"),
        manifest_sample("good", cfg.dataset_root / "good.mp4", category="A", sample_type="FakeVideo-FakeAudio"),
    ]
    monkeypatch.setattr("services.ai.evaluation.runner.build_manifest", lambda **kwargs: samples)

    def fake_video(sample: ManifestSample, run_dir: Path) -> dict[str, object]:
        if sample.sample_id == "bad":
            raise RuntimeError("broken video")
        return {"video_score": {"final_fake_score": 0.8}}

    monkeypatch.setattr("services.ai.evaluation.runner._run_video_pipeline", fake_video)
    monkeypatch.setattr("services.ai.evaluation.runner._run_audio_pipeline", lambda sample, run_dir: {"audio_fake_prob_like": 0.6})

    result = run_dataset_evaluation(cfg)

    failed = Path(result["predictions"]["failed_jsonl"]).read_text(encoding="utf-8")
    video_rows = Path(result["predictions"]["video_jsonl"]).read_text(encoding="utf-8").splitlines()
    failed_type_metrics = json.loads(Path(result["metrics"]["FakeVideo-RealAudio"]).read_text(encoding="utf-8"))
    assert "broken video" in failed
    assert len(video_rows) == 1
    assert failed_type_metrics["results"]["video"]["total_count"] == 0


def test_run_dataset_evaluation_reuses_run_dir_and_skips_existing_predictions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = config(tmp_path)
    run_dir = cfg.output_root / "existing-run"
    run_dir.mkdir(parents=True)
    (run_dir / "predictions_video.jsonl").write_text(
        '{"sample_id": "partial-only", "dataset_name": "FakeAVCeleb", "category": "A", "type": "FakeVideo-FakeAudio", "label": 1, "fake_score": 0.9}\n',
        encoding="utf-8",
    )
    (run_dir / "predictions_combined.jsonl").write_text(
        '{"sample_id": "already-done", "dataset_name": "FakeAVCeleb"}\n',
        encoding="utf-8",
    )
    samples = [
        manifest_sample("already-done", cfg.dataset_root / "already.mp4"),
        manifest_sample("partial-only", cfg.dataset_root / "partial.mp4"),
        manifest_sample("new-sample", cfg.dataset_root / "new.mp4"),
    ]
    calls: list[str] = []
    monkeypatch.setattr("services.ai.evaluation.runner.build_manifest", lambda **kwargs: samples)

    def fake_video(sample: ManifestSample, run_dir: Path) -> dict[str, object]:
        calls.append(sample.sample_id)
        return {"video_score": {"final_fake_score": 0.8}}

    monkeypatch.setattr("services.ai.evaluation.runner._run_video_pipeline", fake_video)
    monkeypatch.setattr("services.ai.evaluation.runner._run_audio_pipeline", lambda sample, run_dir: {"audio_fake_prob_like": 0.6})

    result = run_dataset_evaluation(cfg, run_dir=run_dir)

    video_rows = Path(result["predictions"]["video_jsonl"]).read_text(encoding="utf-8").splitlines()
    assert Path(result["run_dir"]) == run_dir
    assert calls == ["partial-only", "new-sample"]
    assert len(video_rows) == 3


def test_run_dataset_evaluation_filters_to_one_type_and_exports_final_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = config(tmp_path)
    results_root = tmp_path / "final-results"
    samples = [
        manifest_sample(
            "target",
            cfg.dataset_root / "target.mp4",
            category="D",
            sample_type="RealVideo-RealAudio",
        ),
        manifest_sample(
            "skipped",
            cfg.dataset_root / "skipped.mp4",
            category="A",
            sample_type="FakeVideo-FakeAudio",
        ),
    ]
    calls: list[str] = []
    monkeypatch.setattr("services.ai.evaluation.runner.build_manifest", lambda **kwargs: samples)

    def fake_video(sample: ManifestSample, run_dir: Path) -> dict[str, object]:
        calls.append(sample.sample_id)
        return {"video_score": {"final_fake_score": 0.1}}

    monkeypatch.setattr("services.ai.evaluation.runner._run_video_pipeline", fake_video)
    monkeypatch.setattr(
        "services.ai.evaluation.runner._run_audio_pipeline",
        lambda sample, run_dir: {"audio_fake_prob_like": 0.2},
    )

    result = run_dataset_evaluation(
        cfg,
        sample_type="RealVideo-RealAudio",
        results_root=results_root,
    )

    run_dir = Path(result["run_dir"])
    manifest_rows = (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    final_path = results_root / "RealVideo-RealAudio" / "RealVideo-RealAudio.json"
    final_metrics = json.loads(final_path.read_text(encoding="utf-8"))

    assert calls == ["target"]
    assert len(manifest_rows) == 1
    assert json.loads(manifest_rows[0])["type"] == "RealVideo-RealAudio"
    assert set(result["metrics"]) == {"RealVideo-RealAudio"}
    assert result["final_metrics"] == {"RealVideo-RealAudio": str(final_path)}
    assert final_metrics["dataset_name"] == "RealVideo-RealAudio"
    assert set(final_metrics["results"]) == {"video", "audio", "fusion"}
    assert final_metrics["results"]["video"]["total_count"] == 1
    assert not (results_root / "FakeVideo-FakeAudio").exists()
