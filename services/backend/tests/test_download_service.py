from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi import BackgroundTasks, HTTPException


class DownloadServiceTests(unittest.IsolatedAsyncioTestCase):
    # 버그 수정: instagram.py가 DB 기반으로 리팩토링됨
    # 구형 tasks_db 딕셔너리 인터페이스 → 새 DB 기반(mock_db) 인터페이스로 재작성

    async def test_receive_instagram_rejects_non_instagram_link(self) -> None:
        from services.backend.routers.instagram import receive_instagram

        mock_db = MagicMock()

        with self.assertRaises(HTTPException) as context:
            await receive_instagram(
                background_tasks=BackgroundTasks(),
                title="sample",
                link="https://example.com/video",
                db=mock_db,
            )

        self.assertEqual(context.exception.status_code, 400)

    async def test_get_status_returns_existing_task_shape(self) -> None:
        from services.backend import crud
        from services.backend.routers.instagram import get_status
        from services.backend import models
        from datetime import datetime

        mock_task = models.VideoMetadata(
            task_id="upload-1",
            status="PENDING",
            verdict=None,
            origin_url=None,
            storage_path=None,
            audio_path=None,
            phash_value=None,
            deepfake_score=None,
            user_id=None,
            created_at=datetime.now(),
        )
        mock_db = MagicMock()

        with patch.object(crud, "get_task_by_id", return_value=mock_task):
            result = await get_status("upload-1", db=mock_db)

        self.assertEqual(result["task_id"], "upload-1")
        self.assertEqual(result["status"], "PENDING")
        self.assertIn("verdict", result)

    async def test_get_status_raises_for_missing_upload_task(self) -> None:
        from services.backend import crud
        from services.backend.routers.instagram import get_status

        mock_db = MagicMock()

        with patch.object(crud, "get_task_by_id", return_value=None):
            with self.assertRaises(HTTPException) as context:
                await get_status("missing", db=mock_db)

        self.assertEqual(context.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
