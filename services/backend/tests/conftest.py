"""최상위 conftest: AI 무거운 패키지를 stub으로 대체하여 테스트 환경 구성."""
from __future__ import annotations
import os
import sys
import types

# ── 1. DB 환경변수 (모듈 import 전 설정) ─────────────────────────────────────
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

# ── 2. AI 무거운 모듈 stub ────────────────────────────────────────────────────
_SIMPLE_STUBS = [
    "cv2",
    "retinaface",
    "retinaface.pre_ask",
    "tensorflow",
    "tf_keras",
    "torch",
    "torchvision",
    "torchvision.transforms",
    "efficientnet_pytorch",
    "instaloader",
]
for _name in _SIMPLE_STUBS:
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)

# videohash: VideoHash 클래스 포함 stub
if "videohash" not in sys.modules:
    _vh = types.ModuleType("videohash")
    class _VideoHash:
        def __init__(self, path: object = None, **kwargs: object) -> None:
            self.hash_hex = "0" * 16
    _vh.VideoHash = _VideoHash  # type: ignore[attr-defined]
    sys.modules["videohash"] = _vh

# static_ffmpeg stub
if "static_ffmpeg" not in sys.modules:
    _sfmpeg = types.ModuleType("static_ffmpeg")
    _sfmpeg.add_paths = lambda: None  # type: ignore[attr-defined]
    sys.modules["static_ffmpeg"] = _sfmpeg

# ── 3. PIL 호환성 ─────────────────────────────────────────────────────────────
try:
    import PIL.Image
    if not hasattr(PIL.Image, "ANTIALIAS"):
        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
except ImportError:
    pass
