from __future__ import annotations

from pathlib import Path

import pytest

from services.ai.evaluation.config import (
    DEFAULT_OUTPUT_ROOT,
    EvalConfig,
    load_eval_config,
)


def write_ini(path: Path, output_root: Path | None = None) -> Path:
    output_line = f"root = {output_root}\n" if output_root is not None else ""
    path.write_text(
        "\n".join(
            [
                "[dataset]",
                f"root = {path.parent / 'FakeAVCeleb_v1.2'}",
                "metadata_csv = meta_data.csv",
                "",
                "[output]",
                output_line.rstrip(),
                "",
                "[inference]",
                "video_enabled = true",
                "audio_enabled = false",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_load_eval_config_uses_default_output_root_when_ini_output_missing(tmp_path: Path) -> None:
    config_path = write_ini(tmp_path / "eval.ini")

    config = load_eval_config(config_path)

    assert isinstance(config, EvalConfig)
    assert config.dataset_root == (tmp_path / "FakeAVCeleb_v1.2").resolve()
    assert config.metadata_csv == (tmp_path / "FakeAVCeleb_v1.2" / "meta_data.csv").resolve()
    assert config.output_root == DEFAULT_OUTPUT_ROOT
    assert DEFAULT_OUTPUT_ROOT == (Path.home() / "Downloads" / "FakeAVCeleb_v1.2_eval_outputs").resolve()
    assert config.video_enabled is True
    assert config.audio_enabled is False


def test_load_eval_config_output_root_override_wins(tmp_path: Path) -> None:
    ini_output = tmp_path / "ini-output"
    override = tmp_path / "override-output"
    config_path = write_ini(tmp_path / "eval.ini", output_root=ini_output)

    config = load_eval_config(config_path, output_root_override=override)

    assert config.output_root == override.resolve()


def test_load_eval_config_rejects_repo_local_output_root(tmp_path: Path) -> None:
    config_path = write_ini(tmp_path / "eval.ini")
    repo_local_output = Path("/Users/wison/Documents/working/project/verifake/outputs/eval")

    with pytest.raises(ValueError, match="outside the repository"):
        load_eval_config(config_path, output_root_override=repo_local_output)
