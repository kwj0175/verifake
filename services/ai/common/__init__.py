from services.ai.common.job_paths import build_job_paths, create_job_dirs
from services.ai.common.json_io import read_json, write_json


def probe_video(*args, **kwargs):
    from services.ai.common.video_probe import probe_video as _probe_video

    return _probe_video(*args, **kwargs)

__all__ = [
    "build_job_paths",
    "create_job_dirs",
    "read_json",
    "write_json",
    "probe_video",
]
