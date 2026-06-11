# pyright: reportMissingImports=false

from __future__ import annotations

from pathlib import Path

import pytest

from services.ai.inference.video_stage1.transforms import preprocess_face_crop
from services.ai.pipelines.video_stage1.exceptions import Stage1UnavailableError


def test_preprocess_face_crop_raises_for_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        preprocess_face_crop(
            "/tmp/does-not-exist.jpg",
            resolution=256,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )


def test_preprocess_face_crop_raises_when_runtime_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    crop_path = tmp_path / "face.jpg"
    crop_path.write_bytes(b"face")

    monkeypatch.setattr(
        "services.ai.inference.video_stage1.transforms._load_transform_runtime",
        lambda: (_ for _ in ()).throw(Stage1UnavailableError("missing runtime")),
    )

    with pytest.raises(Stage1UnavailableError, match="missing runtime"):
        preprocess_face_crop(
            crop_path,
            resolution=256,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
