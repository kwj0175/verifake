# Evaluation Config

`services/ai/evaluation/config.py` is the configuration loader for the FakeAVCeleb dataset evaluation runner.

## Purpose

This module decides:

- which FakeAVCeleb dataset root to read from
- which metadata CSV to parse
- where evaluation outputs should be written
- whether video inference is enabled
- whether audio inference is enabled

The evaluation runner imports this module before building the manifest or running any pipeline stages.

## Main Constants

### `DEFAULT_OUTPUT_ROOT`

Default output directory when the INI file does not provide an `[output] root` value and the CLI does not pass `--output-root`.

```text
~/Downloads/FakeAVCeleb_v1.2_eval_outputs
```

### `REPO_ROOT`

Resolved repository root, calculated from the location of `config.py`.

It is used to prevent evaluation outputs from being written inside the git repository.

## `EvalConfig`

`EvalConfig` is a frozen dataclass with these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `dataset_root` | `Path` | Root directory of the FakeAVCeleb dataset |
| `metadata_csv` | `Path` | Path to the FakeAVCeleb metadata CSV |
| `output_root` | `Path` | External directory where evaluation runs are stored |
| `video_enabled` | `bool` | Whether to run the video pipeline |
| `audio_enabled` | `bool` | Whether to run the audio pipeline |

## Path Resolution

`_resolve_path()` expands `~`, resolves absolute paths, and resolves relative paths against a provided base directory.

Current behavior:

- `[dataset] root` is resolved relative to the INI file directory.
- `[dataset] metadata_csv` is resolved relative to `dataset_root`.
- CLI `--output-root` is resolved from the current filesystem context.
- `[output] root` is resolved directly if no CLI override is provided.

## Output Safety Check

`_validate_output_root()` rejects output paths inside the repository.

This is intentional because evaluation runs can create large artifacts, predictions, metrics, and failure logs. Those outputs should stay outside git-managed source directories.

If the output root is inside the repo, `load_eval_config()` raises:

```text
ValueError: Evaluation output root must be outside the repository
```

## `load_eval_config()`

Signature:

```python
load_eval_config(
    config_path: str | Path,
    output_root_override: str | Path | None = None,
) -> EvalConfig
```

Behavior:

1. Reads the INI file using `configparser`.
2. Raises `FileNotFoundError` if the config file cannot be read.
3. Reads dataset paths from `[dataset]`.
4. Chooses `output_root` in this priority order:
   - `output_root_override`
   - `[output] root`
   - `DEFAULT_OUTPUT_ROOT`
5. Rejects repo-local output directories.
6. Reads inference toggles from `[inference]`.
7. Returns an `EvalConfig` instance.

## Expected INI Shape

Example:

```ini
[dataset]
root = /path/to/FakeAVCeleb_v1.2
metadata_csv = meta_data.csv

[output]
root = /Users/example/Downloads/FakeAVCeleb_v1.2_eval_outputs

[inference]
video_enabled = true
audio_enabled = false
```

## Git Note

The repository currently ignores files matching `**/config.py`, so `services/ai/evaluation/config.py` is ignored by git unless it is force-added or renamed.

Because other evaluation modules import `services.ai.evaluation.config`, the file must be included somehow before committing this feature. Otherwise a fresh checkout will fail with `ModuleNotFoundError`.
