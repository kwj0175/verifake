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

    if not result_path.exists():
        print(f"[video] ④ result.json 없음 ({task_id})")
        return None

    # LLM 설명 생성
    audio_result_path = Path(__file__).resolve().parents[3] / "storage" / "jobs" / task_id / "audio" / "audio_stage1_result.json"
    if audio_result_path.exists():
        try:
            print(f"[video] ④ LLM 설명 생성 중... ({task_id})")
            from services.backend.processor import run_video_stage1_result_explainer_job
            run_video_stage1_result_explainer_job(result_path, audio_result_path)
            print(f"[video] ④ LLM 설명 생성 완료 ({task_id})")
        except Exception as exc:
            print(f"[video] ④ LLM 설명 생성 실패 (계속 진행): {exc}")
    else:
        print(f"[video] ④ 오디오 결과 없어 LLM 설명 건너뜀 ({task_id})")

    try:
        verdict, score = parse_result_json(result_path)
        print(f"[video] ⑤ 결과 파싱 완료 → verdict={verdict}, score={score} ({task_id})")
        return verdict, score
    except Exception as exc:
        print(f"[video] ⑤ 결과 파싱 실패 ({task_id}): {exc}")
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

    # result.json에서 LLM 설명, top_segments, face_detect_ratio, video_fake_score 읽기
    llm_explanations = None
    top_segments = []
    face_detect_ratio = None
    video_fake_score = None
    audio_score = None
    audio_suspicious_segments = []

    if task.task_id:
        result_path = Path(__file__).resolve().parents[3] / "storage" / "jobs" / task.task_id / "output" / "result.json"
        if result_path.exists():
            try:
                import json as _json
                result_data = _json.loads(result_path.read_text(encoding="utf-8"))
                llm_explanations = result_data.get("llm_explanations")
                raw_top_segments = result_data.get("detection", {}).get("top_segments", [])
                _reason_map = {
                    "high consecutive face artifact scores": "얼굴 경계 부자연스러움",
                    "high fake score": "딥페이크 가능성 높음",
                    "face artifact detected": "얼굴 아티팩트 감지",
                    "eye blink pattern anomaly": "눈 깜빡임 패턴 이상",
                    "facial boundary irregularity": "얼굴 경계 불규칙",
                    "unnatural facial movement": "부자연스러운 얼굴 움직임",
                    "low quality face region": "얼굴 영역 품질 낮음",
                }
                top_segments = []
                for seg in raw_top_segments:
                    reason_en = seg.get("reason", "")
                    reason_ko = _reason_map.get(reason_en, reason_en)
                    top_segments.append({**seg, "reason": reason_ko})
                face_detect_ratio = result_data.get("quality_metrics", {}).get("face_detect_ratio")
                video_fake_score = result_data.get("detection", {}).get("video_score", {}).get("final_fake_score")
            except Exception:
                pass

        audio_result_path = Path(__file__).resolve().parents[3] / "storage" / "jobs" / task.task_id / "audio" / "audio_stage1_result.json"
        if audio_result_path.exists():
            try:
                import json as _json
                audio_data = _json.loads(audio_result_path.read_text(encoding="utf-8"))
                # audio_fake_prob_like: 0~1 범위 → 퍼센트로 변환
                prob = audio_data.get("audio_fake_prob_like")
                if prob is not None:
                    audio_score = round(float(prob) * 100, 1)
                # 의심 구간 (시간 + 이유)
                raw_segs = audio_data.get("top_suspicious_audio_segments", [])
                audio_suspicious_segments = []
                for s in raw_segs:
                    start = s.get('start_sec', 0)
                    end = s.get('end_sec', 0)
                    prob = s.get('fake_prob_like', None)
                    m_start = int(start // 60)
                    s_start = int(start % 60)
                    m_end = int(end // 60)
                    s_end = int(end % 60)
                    time_str = f"{m_start}:{s_start:02d}~{m_end}:{s_end:02d}"
                    # fake_prob_like 값 기반으로 이유 생성
                    if prob is not None:
                        p = float(prob)
                        if p >= 0.9:
                            reason = "음성 끊김 감지"
                        elif p >= 0.7:
                            reason = "발화 속도 불규칙"
                        elif p >= 0.5:
                            reason = "음성 패턴 이상"
                        else:
                            reason = "경미한 변조 의심"
                        audio_suspicious_segments.append(f"{time_str} - {reason}")
                    else:
                        audio_suspicious_segments.append(time_str)
            except Exception:
                pass

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
        "llm_explanations": llm_explanations,
        "top_segments": top_segments,
        "face_detect_ratio": face_detect_ratio,
        "video_fake_score": video_fake_score,
        "audio_score": audio_score,
        "audio_suspicious_segments": audio_suspicious_segments,
    }
