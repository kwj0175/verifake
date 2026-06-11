from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import aggregate_audio_rerun_results
from services.ai.evaluation import dataset_eval
from services.ai.evaluation.config import EvalConfig


def test_cli_main_loads_config_runs_evaluation_and_prints_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "eval.ini"
    output_root = tmp_path / "out"
    captured: dict[str, object] = {}

    def fake_load_eval_config(config_path_arg: Path, output_root_override: Path | None = None) -> EvalConfig:
        captured["config_path"] = config_path_arg
        captured["override"] = output_root_override
        return EvalConfig(
            dataset_root=tmp_path / "dataset",
            metadata_csv=tmp_path / "dataset" / "meta_data.csv",
            output_root=output_root,
            video_enabled=True,
            audio_enabled=False,
        )

    run_dir = output_root / "existing-run"

    def fake_run_dataset_evaluation(
        config: EvalConfig,
        limit: int | None = None,
        run_dir: Path | None = None,
        sample_type: str | None = None,
        results_root: Path | None = None,
        shard_index: int | None = None,
        num_shards: int | None = None,
        overfit_thresholds: bool = False,
    ) -> dict[str, object]:
        captured["limit"] = limit
        captured["run_dir"] = run_dir
        captured["sample_type"] = sample_type
        captured["results_root"] = results_root
        captured["shard_index"] = shard_index
        captured["num_shards"] = num_shards
        captured["overfit_thresholds"] = overfit_thresholds
        assert run_dir is not None
        return {
            "run_dir": str(run_dir),
            "manifest_jsonl": str(run_dir / "manifest.jsonl"),
            "predictions": {"video_jsonl": str(output_root / "run" / "predictions_video.jsonl")},
            "metrics": {"FakeAVCeleb": str(output_root / "run" / "metrics" / "FakeAVCeleb_metrics.json")},
        }

    monkeypatch.setattr(dataset_eval, "load_eval_config", fake_load_eval_config)
    monkeypatch.setattr(dataset_eval, "run_dataset_evaluation", fake_run_dataset_evaluation)

    exit_code = dataset_eval.main(
        [
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--limit",
            "7",
            "--run-dir",
            str(run_dir),
            "--type",
            "RealVideo-RealAudio",
            "--results-root",
            str(output_root / "final"),
            "--overfit-thresholds",
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert captured == {
        "config_path": config_path,
        "override": output_root,
        "limit": 7,
        "run_dir": run_dir,
        "sample_type": "RealVideo-RealAudio",
        "results_root": output_root / "final",
        "shard_index": None,
        "num_shards": None,
        "overfit_thresholds": True,
    }
    assert printed["run_dir"].endswith("existing-run")


def test_audio_rerun_aggregate_passes_overfit_thresholds_to_write_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_merge_shard(
        run_dir: Path,
        *,
        shard_index: int,
        num_shards: int,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
        return [], [], [], []

    def fake_write_metrics(**kwargs: object) -> dict[str, Path]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(aggregate_audio_rerun_results, "_merge_shard", fake_merge_shard)
    monkeypatch.setattr(aggregate_audio_rerun_results, "write_metrics", fake_write_metrics)

    aggregate_audio_rerun_results.aggregate(
        tmp_path / "run",
        tmp_path / "results",
        num_shards=1,
        overfit_thresholds=True,
    )

    assert captured["overfit_thresholds"] is True


def test_overfit_threshold_help_marks_diagnostic_only(monkeypatch, capsys) -> None:
    dataset_help = dataset_eval._build_parser().format_help()
    assert "diagnostic in-sample threshold calibration" in dataset_help
    assert "not valid for benchmark reporting" in dataset_help
    assert "reported metrics" not in dataset_help

    monkeypatch.setattr(
        sys,
        "argv",
        ["aggregate_audio_rerun_results.py", "--help"],
    )
    with pytest.raises(SystemExit) as exc_info:
        aggregate_audio_rerun_results.main()

    assert exc_info.value.code == 0
    aggregate_help = capsys.readouterr().out
    assert "diagnostic in-sample threshold calibration" in aggregate_help
    assert "not valid for benchmark reporting" in aggregate_help
    assert "reported metrics" not in aggregate_help
