from __future__ import annotations

import asyncio
import os
import re
import uuid
from pathlib import Path

import instaloader
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from videohash import VideoHash

from services.backend import database, models
from services.backend.services.processor import separate_streams, TMP_DIR

load_dotenv()


def _extract_shortcode(url: str) -> str:
    match = re.search(r"/(?:p|reel)/([A-Za-z0-9_-]+)", url)
    if not match:
        raise ValueError("인스타그램 shortcode를 추출할 수 없습니다.")
    return match.group(1)


def _download_instagram(url: str, dest_dir: Path) -> None:
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

    # 환경변수에서 인스타그램 계정 정보 로드
    username = os.getenv("INSTAGRAM_USERNAME")
    password = os.getenv("INSTAGRAM_PASSWORD")
    if username and password:
        try:
            loader.login(username, password)
            print(f"[instagram] 로그인 성공: {username}")
        except Exception as exc:
            print(f"[instagram] 로그인 실패 (비로그인으로 계속): {exc}")

    post = instaloader.Post.from_shortcode(loader.context, shortcode)
    loader.download_post(post, target=shortcode)


def _ensure_user_id(task: models.VideoMetadata) -> None:
    if not task.user_id:
        task.user_id = str(uuid.uuid4())
        print(f"Generated new user_id: {task.user_id} for task: {task.task_id}")


def _verify_downloaded_files(dest_dir: Path) -> Path:
    video_files = list(dest_dir.glob("**/*.mp4"))
    if not video_files:
        raise FileNotFoundError("다운로드된 영상 파일을 찾을 수 없습니다.")
    return video_files[0]


async def _process_video_file(
    task_id: str,
    video_file: Path,
) -> tuple[str, str, str]:
    video_path, audio_path = separate_streams(video_file, task_id)
    phash_value = await asyncio.to_thread(
        lambda: VideoHash(path=video_path).hash_hex
    )
    return video_path, audio_path, phash_value


def _update_task_success(
    task: models.VideoMetadata,
    video_path: str,
    audio_path: str,
    phash_value: str,
) -> None:
    task.storage_path = video_path
    task.audio_path = audio_path
    task.phash_value = phash_value
    task.status = "COMPLETED"


def _update_task_failure(task: models.VideoMetadata, error: str) -> None:
    task.status = "FAILED"
    print(f"Task {task.task_id} failed: {error}")


async def run_download(task_id: str, url: str) -> None:
    db = database.SessionLocal()

    try:
        task = db.query(models.VideoMetadata).filter(
            models.VideoMetadata.task_id == task_id
        ).first()

        if not task:
            print(f"Task {task_id} not found in database")
            return

        _ensure_user_id(task)

        dest_dir = TMP_DIR / task_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        task.status = "PROCESSING"
        task.download_dir = str(dest_dir)
        db.commit()

        print(f"Starting download for task {task_id}...")
        await asyncio.to_thread(_download_instagram, url, dest_dir)

        video_file = _verify_downloaded_files(dest_dir)

        video_path, audio_path, phash_value = await _process_video_file(
            task_id,
            video_file,
        )

        _update_task_success(task, video_path, audio_path, phash_value)
        db.commit()

    except Exception as exc:
        db.rollback()
        task = db.query(models.VideoMetadata).filter(
            models.VideoMetadata.task_id == task_id
        ).first()
        if task:
            _update_task_failure(task, str(exc))
            db.commit()

    finally:
        db.close()
