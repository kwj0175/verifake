from __future__ import annotations

from pathlib import Path

from services.ai.inference.video_stage1.deepfakebench_efficientnet_b4 import (
    predict_face_crops,
)


def test_predict_face_crops_returns_empty_list_for_empty_input() -> None:
    assert predict_face_crops([]) == []


def test_predict_face_crops_returns_mock_scores_with_original_metadata() -> None:
    face_items = [
        {
            "face_id": "face_000001_00",
            "frame_index": 1,
            "timestamp_sec": 0.333,
            "crop_path": "storage/jobs/mock_job_001/faces/frame_000001_face_00.jpg",
        }
    ]

    result = predict_face_crops(face_items)

    assert result == [
        {
            "face_id": "face_000001_00",
            "frame_index": 1,
            "timestamp_sec": 0.333,
            "crop_path": "storage/jobs/mock_job_001/faces/frame_000001_face_00.jpg",
            "raw_fake_score": 0.5,
            "inference_success": True,
        }
    ]


def test_predict_face_crops_real_mode_maps_model_scores(
    monkeypatch,
    tmp_path: Path,
) -> None:
    crop_path_1 = tmp_path / "face_1.jpg"
    crop_path_1.write_bytes(b"face-1")
    crop_path_2 = tmp_path / "face_2.jpg"
    crop_path_2.write_bytes(b"face-2")

    face_items = [
        {
            "face_id": "face_000001_00",
            "frame_index": 1,
            "timestamp_sec": 0.333,
            "crop_path": str(crop_path_1),
        },
        {
            "face_id": "face_000002_00",
            "frame_index": 2,
            "timestamp_sec": 0.666,
            "crop_path": str(crop_path_2),
        },
    ]

    class FakeBatch:
        def to(self, device: object) -> "FakeBatch":
            return self

    class FakeProbabilities:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def detach(self) -> "FakeProbabilities":
            return self

        def cpu(self) -> "FakeProbabilities":
            return self

        def tolist(self) -> list[float]:
            return self._values

    class FakeModel:
        def __call__(self, data_dict: dict[str, object], inference: bool = False) -> dict[str, FakeProbabilities]:
            assert inference is True
            assert "image" in data_dict
            return {"prob": FakeProbabilities([0.2, 0.9])}

    monkeypatch.setattr(
        "services.ai.inference.video_stage1.deepfakebench_efficientnet_b4.load_efficientnet_b4_detector",
        lambda **kwargs: (FakeModel(), "cpu", {"resolution": 256, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}),
    )
    monkeypatch.setattr(
        "services.ai.inference.video_stage1.deepfakebench_efficientnet_b4.preprocess_face_crop",
        lambda crop_path, resolution, mean, std: {"crop_path": crop_path, "resolution": resolution, "mean": mean, "std": std},
    )
    monkeypatch.setattr(
        "services.ai.inference.video_stage1.deepfakebench_efficientnet_b4.stack_face_tensors",
        lambda tensors: FakeBatch(),
    )

    result = predict_face_crops(
        face_items,
        batch_size=2,
        use_mock=False,
        weights_path="/tmp/fake-model.pth",
    )

    assert result == [
        {
            "face_id": "face_000001_00",
            "frame_index": 1,
            "timestamp_sec": 0.333,
            "crop_path": str(crop_path_1),
            "raw_fake_score": 0.2,
            "inference_success": True,
        },
        {
            "face_id": "face_000002_00",
            "frame_index": 2,
            "timestamp_sec": 0.666,
            "crop_path": str(crop_path_2),
            "raw_fake_score": 0.9,
            "inference_success": True,
        },
    ]
