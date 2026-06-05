# pyright: reportMissingImports=false
from __future__ import annotations

import concurrent.futures
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from services.backend import crud, models
from services.backend.database import SessionLocal, get_db
from services.backend.services.download import run_download
from services.backend.services.processor import save_and_split
from services.backend.services.video_analyzer import parse_result_json, run_video_detect_job
from services.backend.tasks import create_video_detect_job

router = APIRouter()


# ---------------------------------------------------------------------------
# 내부 헬퍼: 비디오 분석 (preprocess → detect)
# ---------------------------------------------------------------------------

def _run_video_analysis(task_id: str, video_path: str) -> tuple[str, float] | None:
    """Stage1 전처리 → 탐지 실행. (verdict, score) 또는 None 반환."""
    from services.ai.pipelines.video_stage1.config import get_stage1_storage_root
    from services.ai.common.job_paths import build_job_paths
    from services.backend.routers.video import run_video_stage1_preprocess_job

    print(f"[video] ① 전처리 시작 ({task_id})")
    try:
        preprocess_result = run_video_stage1_preprocess_job(
            Path(video_path), job_id=task_id
        )
    except Exception as exc:
        print(f"[video] ① 전처리 실패 ({task_id}): {exc}")
        return None

    print(f"[video] ② 전처리 완료 → 탐지 시작 ({task_id})")
    storage_root = Path(get_stage1_storage_root())
    if not storage_root.is_absolute():
        storage_root = (Path(__file__).resolve().parents[3] / storage_root).resolve()

    job_paths = build_job_paths(preprocess_result["job_id"], storage_root=storage_root)
    preprocessing_json: Path = job_paths["preprocessing_json_path"]

    if not preprocessing_json.exists():
        print(f"[video] ② preprocessing.json 없음 ({task_id})")
        return None

    artifacts_dir = str(Path("storage/jobs") / task_id / "output")
    create_video_detect_job(task_id, str(preprocessing_json.resolve()), artifacts_dir)
    print(f"[video] ③ 딥페이크 탐지 모델 실행 중... ({task_id})")
    run_video_detect_job(task_id, preprocessing_json)
    print(f"[video] ③ 딥페이크 탐지 완료 ({task_id})")

    result_path = Path(artifacts_dir) / "result.json"
    if not result_path.exists():
        from services.backend.tasks import get_video_detect_job
        job = get_video_detect_job(task_id)
        if job and job.get("result_path"):
            result_path = Path(job["result_path"])

    if result_path.exists():
        try:
            verdict, score = parse_result_json(result_path)
            print(f"[video] ④ 결과 파싱 완료 → verdict={verdict}, score={score} ({task_id})")
            return verdict, score
        except Exception as exc:
            print(f"[video] ④ 결과 파싱 실패 ({task_id}): {exc}")
            return None

    print(f"[video] ④ result.json 없음 ({task_id})")
    return None


# ---------------------------------------------------------------------------
# 내부 헬퍼: 오디오 분석
# ---------------------------------------------------------------------------

def _run_audio_analysis(task_id: str, audio_path: str) -> bool:
    """오디오 분석 실행. 성공 여부 반환."""
    from services.backend.tasks import create_audio_job
    from services.backend.services.audio_analyzer import run_audio_job

    try:
        create_audio_job(task_id, audio_path, str(Path("storage/jobs") / task_id / "audio"))
        run_audio_job(task_id, Path(audio_path))
        print(f"[audio] analysis completed for {task_id}")
        return True
    except Exception as exc:
        print(f"[audio] analysis failed for {task_id}: {exc}")
        return False


# ---------------------------------------------------------------------------
# 내부 헬퍼: 비디오 + 오디오 병렬 파이프라인
# ---------------------------------------------------------------------------

def _run_video_pipeline(task_id: str, video_path: str) -> None:
    """비디오 분석 + 오디오 분석을 병렬로 실행 후 DB 저장."""
    db: Session = SessionLocal()
    try:
        task = db.query(models.VideoMetadata).filter(
            models.VideoMetadata.task_id == task_id
        ).first()
        if not task:
            return

        # ── 상태: 전처리 중 ──────────────────────────────────────────────
        task.status = "PREPROCESSING"
        db.commit()

        audio_path = task.audio_path

        # ── 상태: AI 분석 중 ─────────────────────────────────────────────
        task.status = "ANALYZING"
        db.commit()

        # 비디오 + 오디오 병렬 실행
        video_result = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            video_future = executor.submit(_run_video_analysis, task_id, video_path)

            # 오디오 파일이 있을 때만 오디오 분석 실행
            if audio_path and Path(audio_path).exists():
                audio_future = executor.submit(_run_audio_analysis, task_id, audio_path)
            else:
                audio_future = None

            # 비디오 결과 수집
            try:
                video_result = video_future.result()
            except Exception as exc:
                print(f"[pipeline] video thread error for {task_id}: {exc}")

            # 오디오 결과 수집 (실패해도 전체 파이프라인은 계속)
            if audio_future:
                try:
                    audio_future.result()
                except Exception as exc:
                    print(f"[pipeline] audio thread error for {task_id}: {exc}")

        # ── 결과 DB 저장 ─────────────────────────────────────────────────
        if video_result is not None:
            verdict, score = video_result
            task.verdict = verdict
            task.deepfake_score = score
            task.status = "COMPLETED"
        else:
            task.status = "FAILED"

        db.commit()

    except Exception as exc:
        db.rollback()
        try:
            task = db.query(models.VideoMetadata).filter(
                models.VideoMetadata.task_id == task_id
            ).first()
            if task:
                task.status = "FAILED"
                db.commit()
        except Exception:
            pass
        print(f"[pipeline] unexpected error for {task_id}: {exc}")
    finally:
        db.close()


def _run_download_then_pipeline(task_id: str, url: str) -> None:
    """인스타그램 다운로드 완료 후 비디오 파이프라인 자동 트리거."""
    import asyncio

    try:
        asyncio.run(run_download(task_id, url))
    except Exception as exc:
        print(f"[download] failed for {task_id}: {exc}")
        _set_task_status(task_id, "FAILED")
        return

    db: Session = SessionLocal()
    try:
        task = db.query(models.VideoMetadata).filter(
            models.VideoMetadata.task_id == task_id
        ).first()
        if not task or not task.storage_path:
            _set_task_status(task_id, "FAILED")
            return
        video_path = task.storage_path
    finally:
        db.close()

    _run_video_pipeline(task_id, video_path)


def _set_task_status(task_id: str, status: str) -> None:
    db: Session = SessionLocal()
    try:
        task = db.query(models.VideoMetadata).filter(
            models.VideoMetadata.task_id == task_id
        ).first()
        if task:
            task.status = status
            db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

@router.post("/instagram", summary="인스타그램 영상 수집", tags=["Upload"])
async def receive_instagram(
    background_tasks: BackgroundTasks,
    title: str = Form(..., description="영상 제목"),
    link: str = Form(..., description="인스타그램 영상 링크"),
    db: Session = Depends(get_db),
) -> dict:
    if "instagram.com" not in link:
        raise HTTPException(status_code=400, detail="유효한 인스타그램 링크가 아닙니다.")

    task_id = str(uuid4())
    new_task = models.VideoMetadata(task_id=task_id, origin_url=link, status="PENDING")
    db.add(new_task)
    db.commit()
    db.refresh(new_task)

    background_tasks.add_task(_run_download_then_pipeline, task_id, link)

    return {
        "task_id": task_id,
        "status": "PENDING",
        "timestamp": datetime.now().isoformat(),
        "message": "수집 요청이 완료되었습니다.",
    }


@router.post("/video", summary="영상 파일 수집", tags=["Upload"])
async def receive_video(
    background_tasks: BackgroundTasks,
    title: str = Form(..., description="영상 제목"),
    videoFile: UploadFile = File(..., description="업로드할 영상 파일"),
    db: Session = Depends(get_db),
) -> dict:
    task_id = str(uuid4())
    content = await videoFile.read()
    download_dir, video_path, audio_path = save_and_split(task_id, videoFile.filename, content)

    new_task = models.VideoMetadata(
        task_id=task_id,
        download_dir=download_dir,
        storage_path=video_path,
        audio_path=audio_path,
        status="PENDING",
    )
    db.add(new_task)
    db.commit()

    background_tasks.add_task(_run_video_pipeline, task_id, video_path)

    return {
        "task_id": task_id,
        "status": "PENDING",
        "timestamp": datetime.now().isoformat(),
        "message": "수집 요청이 완료되었습니다. 분석이 백그라운드에서 시작됩니다.",
    }


@router.get("/status/{task_id}", summary="분석 상태 조회", tags=["Status"])
async def get_status(task_id: str, db: Session = Depends(get_db)) -> dict:
    task = crud.get_task_by_id(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="해당 task_id를 DB에서 찾을 수 없습니다.")

    return {
        "task_id": task.task_id,
        "user_id": task.user_id,
        "status": task.status,
        "origin_url": task.origin_url,
        "video_path": task.storage_path,
        "audio_path": task.audio_path,
        "phash_value": task.phash_value,
        "verdict": task.verdict,
        "deepfake_score": task.deepfake_score,
        "created_at": task.created_at.isoformat() if task.created_at else None,
    }
