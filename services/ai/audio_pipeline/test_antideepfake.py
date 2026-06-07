from __future__ import annotations

import csv
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.ai.audio_pipeline.antideepfake import (
    AudioFileMetadata,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_HPARAMS_PATH,
    _parse_score_row,
    _read_audio_metadata,
    _write_protocol_csv,
    run_antideepfake_inference,
)


def _write_test_wav(path: Path, *, sample_rate: int = 16000) -> None:
    import wave

    frame_count = sample_rate
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frame_count)


class AntiDeepfakeWrapperTests(unittest.TestCase):
    def test_read_audio_metadata_uses_stdlib_for_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.wav"
            _write_test_wav(audio_path, sample_rate=16000)

            with patch("importlib.import_module", side_effect=AssertionError("torchaudio should not be imported")):
                metadata = _read_audio_metadata(audio_path)

            self.assertEqual(metadata.duration_seconds, 1.0)
            self.assertEqual(metadata.sample_rate, 16000)
            self.assertEqual(metadata.channels, 1)
            self.assertEqual(metadata.encoding, "PCM_S")
            self.assertEqual(metadata.bits_per_sample, 16)

    def test_write_protocol_csv_creates_dual_probe_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            protocol_path = Path(temp_dir) / "protocol.csv"
            audio_path = Path(temp_dir) / "sample.wav"
            _write_test_wav(audio_path)

            metadata = AudioFileMetadata(
                duration_seconds=1.0,
                sample_rate=16000,
                channels=1,
                encoding="PCM_S",
                bits_per_sample=16,
            )
            _write_protocol_csv(protocol_path, audio_path, metadata, "req-1")

            with protocol_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertEqual({row["Label"] for row in rows}, {"fake", "real"})
            self.assertTrue(all(Path(row["Path"]).is_absolute() for row in rows))

    def test_parse_score_row_reads_expected_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            score_path = Path(temp_dir) / "evaluation_score.csv"
            with score_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ID", "Score", "Label"])
                writer.writeheader()
                writer.writerow(
                    {
                        "ID": "req-2-fake-probe",
                        "Score": "[3.0, 1.0]",
                        "Label": "0",
                    }
                )
                writer.writerow(
                    {
                        "ID": "req-2-real-probe",
                        "Score": "[3.0, 1.0]",
                        "Label": "1",
                    }
                )

            result = _parse_score_row(score_path, "req-2")

            self.assertGreater(result.fake_probability, result.real_probability)
            self.assertEqual(result.predicted_label, "fake")

    def test_run_inference_raises_for_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.wav"
            hparams_path = Path(temp_dir) / "config.yaml"
            _write_test_wav(audio_path)
            hparams_path.write_text("placeholder: true\n", encoding="utf-8")

            with self.assertRaises(FileNotFoundError) as context:
                run_antideepfake_inference(
                    audio_path,
                    checkpoint_path=Path(temp_dir) / "missing.ckpt",
                    hparams_path=hparams_path,
                )

            self.assertIn(str(DEFAULT_CHECKPOINT_PATH), str(context.exception))

    def test_default_hparams_path_exists_in_repo(self) -> None:
        self.assertTrue(DEFAULT_HPARAMS_PATH.exists())
        self.assertEqual(DEFAULT_HPARAMS_PATH.suffix, ".yaml")
        self.assertIn("services", str(DEFAULT_HPARAMS_PATH))

    def test_run_inference_invokes_vendored_script_and_parses_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "sample.wav"
            hparams_path = temp_path / "config.yaml"
            checkpoint_path = temp_path / "mms_300m.ckpt"
            _write_test_wav(audio_path)
            hparams_path.write_text("placeholder: true\n", encoding="utf-8")
            checkpoint_path.write_bytes(b"checkpoint")

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                score_path = Path(command[command.index("--score_path") + 1])
                score_path.parent.mkdir(parents=True, exist_ok=True)
                with score_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["ID", "Score", "Label"])
                    writer.writeheader()
                    writer.writerow(
                        {
                            "ID": "req-3-fake-probe",
                            "Score": "[0.25, 1.25]",
                            "Label": "0",
                        }
                    )
                    writer.writerow(
                        {
                            "ID": "req-3-real-probe",
                            "Score": "[0.25, 1.25]",
                            "Label": "1",
                        }
                    )
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with patch(
                "services.ai.audio_pipeline.antideepfake._read_audio_metadata",
                return_value=AudioFileMetadata(
                    duration_seconds=1.0,
                    sample_rate=16000,
                    channels=1,
                    encoding="PCM_S",
                    bits_per_sample=16,
                ),
            ), patch(
                "services.ai.audio_pipeline.antideepfake.subprocess.run",
                side_effect=fake_run,
            ) as mock_run:
                result = run_antideepfake_inference(
                    audio_path,
                    request_id="req-3",
                    checkpoint_path=checkpoint_path,
                    hparams_path=hparams_path,
                    artifacts_dir=temp_path / "artifacts",
                    device="cpu",
                )

            self.assertEqual(result.file_path, str(audio_path.resolve()))
            self.assertEqual(result.predicted_label, "real")
            self.assertIsNotNone(result.score_csv_path)
            score_csv_path = result.score_csv_path
            self.assertIsNotNone(score_csv_path)
            if score_csv_path is None:
                raise AssertionError("score_csv_path should be set when artifacts_dir is provided")
            self.assertTrue(score_csv_path.replace("\\", "/").endswith("req-3/output/evaluation_score.csv"))
            self.assertEqual(mock_run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
