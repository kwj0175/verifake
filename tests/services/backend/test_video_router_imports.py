# pyright: reportMissingImports=false

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_importing_backend_main_does_not_load_stage1_ai_runtime_modules() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = """
import importlib
import json
import sys

before = set(sys.modules)
importlib.import_module('services.backend.main')
after = set(sys.modules)

heavy_modules = sorted(
    name
    for name in (after - before)
    if name == 'cv2'
    or name.startswith('retinaface')
    or name.startswith('tensorflow')
    or name.startswith('tf_keras')
)

print(json.dumps(heavy_modules))
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(repo_root),
    }

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []
