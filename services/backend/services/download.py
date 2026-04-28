import asyncio
import re
import instaloader
from pathlib import Path

from services.backend.services.processor import separate_streams, TMP_DIR

from videohash import VideoHash

def _extract_shortcode(url: str) -> str:
    match = re.search(r'/(?:p|reel)/([A-Za-z0-9_-]+)', url)
    if not match:
        raise ValueError("인스타그램 shortcode를 추출할 수 없습니다.")
    return match.group(1)


def _download_instagram(url: str, dest_dir: Path):
    shortcode = _extract_shortcode(url)
    loader = instaloader.Instaloader(
        dirname_pattern=str(dest_dir),
        download_pictures=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        post_metadata_txt_pattern="",
    )
    post = instaloader.Post.from_shortcode(loader.context, shortcode)
    loader.download_post(post, target=shortcode)


async def run_download(task_id: str, url: str, tasks_db: dict):
    dest_dir = TMP_DIR / task_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    tasks_db[task_id]["status"] = "PROCESSING"
    try:
        await asyncio.to_thread(_download_instagram, url, dest_dir)

        video_files = list(dest_dir.glob("**/*.mp4"))
        if not video_files:
            raise FileNotFoundError("다운로드된 영상 파일을 찾을 수 없습니다.")

        video_path, audio_path = separate_streams(video_files[0], task_id)
        tasks_db[task_id]["status"] = "DONE"
        tasks_db[task_id]["download_dir"] = str(dest_dir)
        tasks_db[task_id]["video_path"] = video_path
        tasks_db[task_id]["audio_path"] = audio_path
        tasks_db[task_id]["pHash"] = VideoHash(path=video_path).hash_hex
    except Exception as e:
        tasks_db[task_id]["status"] = "FAILED"
        tasks_db[task_id]["error"] = str(e)
