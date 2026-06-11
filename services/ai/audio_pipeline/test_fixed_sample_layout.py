from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
NEW_SAMPLE_PATH = REPO_ROOT / "storage/audio/fixed_sample_audio.wav"
OLD_SAMPLE_PATH = REPO_ROOT / "services/ai/audio_pipeline/test_samples/LJ001-0001.wav"


class FixedSampleLayoutTests(unittest.TestCase):
    def test_fixed_sample_uses_backend_aligned_location(self) -> None:
        self.assertTrue(NEW_SAMPLE_PATH.exists())
        self.assertFalse(OLD_SAMPLE_PATH.exists())


if __name__ == "__main__":
    unittest.main()
