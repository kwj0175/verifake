from pathlib import Path

from services.ai.common.job_paths import create_job_dirs


def test_create_job_dirs_creates_expected_structure(tmp_path: Path) -> None:
    paths = create_job_dirs("job_test_001", storage_root=tmp_path / "jobs")

    assert paths["job_root"].exists()
    assert paths["input_dir"].exists()
    assert paths["frames_dir"].exists()
    assert paths["faces_dir"].exists()
    assert paths["metadata_dir"].exists()
    assert paths["output_dir"].exists()
    assert paths["logs_dir"].exists()
    assert paths["normalized_video_path"].name == "normalized.mp4"
    assert paths["preprocessing_json_path"].as_posix().endswith("metadata/preprocessing.json")
