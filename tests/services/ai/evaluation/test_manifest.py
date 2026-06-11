from __future__ import annotations

import json
from pathlib import Path

from services.ai.evaluation.manifest import build_manifest, summarize_manifest, write_manifest_jsonl


def write_metadata(path: Path) -> Path:
    path.write_text(
        "source,target1,target2,method,category,type,race,gender,filename,path,\n"
        "srcA,t1,t2,faceswap,A,FakeVideo-FakeAudio,Asian,Female,a.mp4,FakeAVCeleb/FakeVideo-FakeAudio/Asian/Female,\n"
        "srcA,t1,t2,wav2lip,B,FakeVideo-RealAudio,Asian,Female,a.mp4,FakeAVCeleb/FakeVideo-RealAudio/Asian/Female,\n"
        "srcB,t1,t2,faceswap,C,RealVideo-FakeAudio,Black,Male,b.mp4,RealVideo-FakeAudio/Black/Male,\n"
        "srcC,t1,t2,none,D,RealVideo-RealAudio,White,Female,c.mp4,FakeAVCeleb/RealVideo-RealAudio/White/Female,\n",
        encoding="utf-8",
    )
    return path


def test_build_manifest_parses_fakeavceleb_rows_and_labels(tmp_path: Path) -> None:
    metadata_csv = write_metadata(tmp_path / "meta_data.csv")

    samples = build_manifest(dataset_root=tmp_path, metadata_csv=metadata_csv)

    assert len(samples) == 4
    assert samples[0].video_path == str((tmp_path / "FakeVideo-FakeAudio/Asian/Female/a.mp4").resolve())
    assert samples[0].audio_path == samples[0].video_path
    assert samples[0].category == "A"
    assert samples[0].type == "FakeVideo-FakeAudio"
    assert samples[0].labels == {"video_label": 1, "audio_label": 1, "fusion_label": 1}
    assert samples[1].labels == {"video_label": 1, "audio_label": 0, "fusion_label": 1}
    assert samples[2].labels == {"video_label": 0, "audio_label": 1, "fusion_label": 1}
    assert samples[3].labels == {"video_label": 0, "audio_label": 0, "fusion_label": 0}
    assert len({sample.sample_id for sample in samples}) == 4
    assert samples[0].split == "test"


def test_manifest_summary_and_jsonl_writer(tmp_path: Path) -> None:
    metadata_csv = write_metadata(tmp_path / "meta_data.csv")
    samples = build_manifest(dataset_root=tmp_path, metadata_csv=metadata_csv)
    output_path = tmp_path / "run" / "manifest.jsonl"

    summary = summarize_manifest(samples)
    write_manifest_jsonl(samples, output_path)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary["dataset_count"] == 4
    assert summary["sample_count"] == 4
    assert summary["category_counts"]["A"] == 1
    assert summary["type_counts"]["FakeVideo-FakeAudio"] == 1
    assert rows[0]["dataset_name"] == "FakeAVCeleb"
    assert rows[0]["label_name"] == "FakeVideo-FakeAudio"
    assert rows[0]["audio_path"] == rows[0]["video_path"]
