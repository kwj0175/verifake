"""VeriFake runtime diagnostic helper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.ai.common.runtime_probe import collect_runtime_snapshot, summarize_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CPU/GPU runtime status")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    snapshot = collect_runtime_snapshot()

    if args.json:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    else:
        print(summarize_runtime())
        print(
            "\nTorch CUDA enabled:",
            "OK" if snapshot["torch"].get("cuda_available") else "NO",
        )
        if snapshot["torch"].get("cuda_devices"):
            print("Torch devices:", ", ".join(snapshot["torch"]["cuda_devices"]))

    has_torch_cuda = bool(snapshot["torch"].get("cuda_available"))
    has_tf_cuda = bool(snapshot["tensorflow"].get("has_gpu"))

    return 0 if (has_torch_cuda or has_tf_cuda) else 1


if __name__ == "__main__":
    raise SystemExit(main())
