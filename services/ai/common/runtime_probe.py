"""Runtime diagnostics for CPU/GPU inference backends."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run_nvidia_smi() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None


def collect_torch_runtime_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional import
        return {
            "installed": False,
            "error": str(exc),
        }

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    devices: list[str] = []
    if cuda_available:
        for index in range(device_count):
            try:
                devices.append(torch.cuda.get_device_name(index))
            except Exception:
                devices.append(f"cuda:{index}")

    return {
        "installed": True,
        "version": torch.__version__,
        "version_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_devices": devices,
        "default_device": str(torch.device("cuda" if cuda_available else "cpu")),
        "nvidia_smi": _run_nvidia_smi(),
    }


def collect_tf_runtime_info() -> dict[str, Any]:
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - optional import
        return {
            "installed": False,
            "error": str(exc),
        }

    try:
        gpu_devices = tf.config.list_physical_devices("GPU")
    except Exception:
        gpu_devices = []

    return {
        "installed": True,
        "version": tf.__version__,
        "gpu_devices": [str(item) for item in gpu_devices],
        "has_gpu": bool(gpu_devices),
    }


def collect_runtime_snapshot() -> dict[str, Any]:
    return {
        "python": {
            "executable": Path(sys.executable).as_posix(),
            "version": sys.version,
            "platform": platform.platform(),
        },
        "torch": collect_torch_runtime_info(),
        "tensorflow": collect_tf_runtime_info(),
    }


def resolve_torch_device(*, default: str = "cpu", cuda_index_env_var: str = "VERIFAKE_CUDA_DEVICE") -> str:
    """Return the runtime device string with automatic CUDA fallback.

    If CUDA is available in the current Python runtime, this returns ``cuda``
    (or ``cuda:<index>`` when an index is configured). Otherwise returns
    the provided default (typically ``cpu``).
    """
    runtime = collect_torch_runtime_info()
    if not runtime.get("installed") or not runtime.get("cuda_available"):
        return default

    configured = os.getenv(cuda_index_env_var, "").strip()
    if configured:
        if configured.isdigit():
            requested = int(configured)
            device_count = int(runtime.get("cuda_device_count") or 0)
            if requested < 0 or requested >= device_count:
                return "cuda"
            return f"cuda:{requested}"
        return configured

    return "cuda"


def summarize_runtime() -> str:
    snapshot = collect_runtime_snapshot()
    torch_info = snapshot["torch"]
    tf_info = snapshot["tensorflow"]

    lines = [
        f"python={snapshot['python']['version'].split()[0]}",
        f"platform={snapshot['python']['platform']}",
    ]
    if torch_info.get("installed"):
        lines.append(
            f"torch={torch_info['version']} cuda={torch_info['version_cuda']} "
            f"available={torch_info['cuda_available']} devices={torch_info['cuda_device_count']}"
        )
        if torch_info.get("cuda_devices"):
            lines.append("torch_gpus=" + ", ".join(torch_info["cuda_devices"]))
    else:
        lines.append(f"torch=not_installed ({torch_info.get('error')})")

    if tf_info.get("installed"):
        lines.append(f"tensorflow={tf_info['version']} devices={len(tf_info['gpu_devices'])}")
    else:
        lines.append(f"tensorflow=not_installed ({tf_info.get('error')})")

    if torch_info.get("nvidia_smi"):
        lines.append(f"nvidia_smi={torch_info['nvidia_smi']}")

    return "\n".join(lines)


def print_runtime_snapshot() -> None:
    print(summarize_runtime())
