from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).with_name("config.stage1.json")


def load_stage1_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_stage1_storage_root() -> str:
    config = load_stage1_config()
    return str(config["storage_root"])
