import subprocess
import uuid
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.ai.common.job_paths import build_job_paths
from services.backend.processor import (
    run_video_stage1_preprocess_job,
    run_video_stage1_result_explainer_job,
    separate_streams,
)

router = APIRouter()


class SplitRequest(BaseModel):
    file_path: str


class VideoStage1PreprocessRequest(BaseModel):
    file_path: str
    job_id: str | None = None


class VideoStage1ExplainRequest(BaseModel):
    video_result_json: str
    audio_result_json: str


def _to_percent(score: object) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("result.json contains a non-numeric score value.")
    return round(max(0.0, min(1.0, float(score))) * 100.0, 1)


def _to_fake_status(score_percent: float) -> str:
    return "FAKE" if score_percent >= 50.0 else "REAL"


def _build_frontend_result(result: dict[str, Any]) -> dict[str, Any]:
    detection = result.get("detection")
    detection_obj = detection if isinstance(detection, dict) else {}
    video_score = detection_obj.get("video_score")
    video_score_obj = video_score if isinstance(video_score, dict) else {}
    top_segments = detection_obj.get("top_segments")
    top_segment_items = top_segments if isinstance(top_segments, list) else []
    quality_metrics = result.get("quality_metrics")
    quality_obj = quality_metrics if isinstance(quality_metrics, dict) else {}
    llm_explanations = result.get("llm_explanations")
    llm_obj = llm_explanations if isinstance(llm_explanations, dict) else {}

    score = _to_percent(video_score_obj.get("final_fake_score"))
    max_score = _to_percent(video_score_obj.get("max_fake_score"))

    segments: list[dict[str, Any]] = []
    for item in top_segment_items:
        if not isinstance(item, dict):
            continue
        segments.append({
            "startSec": item.get("start_sec"),
            "endSec": item.get("end_sec"),
            "score": _to_percent(item.get("segment_score")),
            "reason": item.get("reason"),
            "framePath": item.get("representative_frame_path"),
        })
    segments.sort(key=lambda item: item["score"], reverse=True)

    thumbnail_url = None
    thumbnail_path = None
    if segments:
        frame_path = segments[0].get("framePath")
        if isinstance(frame_path, str) and frame_path:
            thumbnail_path = frame_path
            if frame_path.startswith(("http://", "https://", "file://")):
                thumbnail_url = frame_path

    llm_summary = llm_obj.get("summary_text")
    llm_detail = llm_obj.get("detail_text")
    llm_reason = llm_detail if isinstance(llm_detail, str) and llm_detail else llm_summary
    status = _to_fake_status(score)

    return {
        "status": status,
        "score": score,
        "maxScore": max_score,
        "aggregationMethod": video_score_obj.get("aggregation_method"),
        "llmSummary": llm_summary,
        "llmDetail": llm_detail,
        "llmReason": llm_reason,
        "thumbnailUrl": thumbnail_url,
        "thumbnailPath": thumbnail_path,
        "videoDetection": "감지됨" if status == "FAKE" or segments else "없음",
        "audioDetection": "미분석",
        "segments": segments,
        "quality": {
            "faceDetectRatio": quality_obj.get("face_detect_ratio"),
            "faceVisibilityRatio": quality_obj.get("face_visibility_ratio"),
            "blurScore": quality_obj.get("blur_score"),
            "darkFrameRatio": quality_obj.get("dark_frame_ratio"),
        },
    }


@router.post("/split")
def split_media(req: SplitRequest):
    try:
        job_id = str(uuid.uuid4())
        input_path = Path(req.file_path)

        if not input_path.exists():
            raise HTTPException(status_code=400, detail="파일이 존재하지 않습니다.")

        video, audio = separate_streams(input_path, job_id)
        return {"job_id": job_id, "video": video, "audio": audio}

    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="ffmpeg 실행 실패")

    # 버그 수정: HTTPException이 except Exception에 잡혀 500으로 변환되던 문제 수정
    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video-stage1/preprocess")
def preprocess_video_stage1(req: VideoStage1PreprocessRequest):
    try:
        input_path = Path(req.file_path)

        if not input_path.exists():
            raise HTTPException(status_code=400, detail="파일이 존재하지 않습니다.")

        result = run_video_stage1_preprocess_job(input_path, job_id=req.job_id)
        job_id = result.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise HTTPException(status_code=500, detail="job_id가 누락되었습니다.")
        job_paths = build_job_paths(job_id)

        return {
            "job_id": job_id,
            "status": result["status"],
            "preprocessing_json": job_paths["preprocessing_json_path"].as_posix(),
        }

    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="ffmpeg 실행 실패")

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video-stage1/explain")
def explain_video_stage1(req: VideoStage1ExplainRequest):
    """LLM 설명 생성 API."""
    import json as _json

    video_result_json_path = Path(req.video_result_json)
    audio_result_json_path = Path(req.audio_result_json)

    if not video_result_json_path.exists() or not video_result_json_path.is_file():
        raise HTTPException(status_code=400, detail="video_result.json 파일이 존재하지 않습니다.")

    if not audio_result_json_path.exists() or not audio_result_json_path.is_file():
        raise HTTPException(status_code=400, detail="audio_result.json 파일이 존재하지 않습니다.")

    try:
        result = run_video_stage1_result_explainer_job(
            video_result_json_path,
            audio_result_json_path,
        )
        frontend_result = _build_frontend_result(result)

        try:
            video_result_json_path.write_text(
                _json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as write_exc:
            print(f"[explain] result.json write failed: {write_exc}")

    except (ValueError, JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "job_id": result.get("job_id"),
        "source_status": result.get("status"),
        "explain_status": "success",
        "video_result_json": video_result_json_path.as_posix(),
        "audio_result_json": audio_result_json_path.as_posix(),
        "llm_explanations": result.get("llm_explanations"),
        "result": frontend_result,
    }
