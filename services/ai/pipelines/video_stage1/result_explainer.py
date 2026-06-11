from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from services.ai.common.json_io import read_json, write_json
from services.ai.pipelines.video_stage1.exceptions import (
    Stage1ExplanationError,
    Stage1UnavailableError,
)

ENV_FILE_PATH = Path(__file__).resolve().parents[4] / ".env"
PROMPTS_DIR = Path(__file__).with_name("prompts")
SUMMARY_PROMPT_PATH = PROMPTS_DIR / "result_summary_prompt.txt"
DETAIL_PROMPT_PATH = PROMPTS_DIR / "result_detail_prompt.txt"
GEMINI_MODEL = "gemini-2.5-flash-lite"


def _load_prompt_template(path: Path) -> str:
    if not path.exists():
        raise Stage1ExplanationError(f"Prompt file not found: {path}")

    template = path.read_text(encoding="utf-8").strip()
    if "{{VIDEO_RESULT_JSON}}" not in template or "{{AUDIO_RESULT_JSON}}" not in template:
        raise Stage1ExplanationError(
            f"Prompt template must include both {{VIDEO_RESULT_JSON}} and {{AUDIO_RESULT_JSON}} placeholders: {path}"
        )
    return template


def _build_prompt(
    template: str,
    video_result_payload: dict[str, Any],
    audio_result_payload: dict[str, Any],
) -> str:
    video_result_json = json.dumps(video_result_payload, ensure_ascii=False, indent=2)
    audio_result_json = json.dumps(audio_result_payload, ensure_ascii=False, indent=2)
    return (
        template
        .replace("{{VIDEO_RESULT_JSON}}", video_result_json)
        .replace("{{AUDIO_RESULT_JSON}}", audio_result_json)
    )


def _validate_result_payload(payload: dict[str, Any]) -> None:
    job_id = payload.get("job_id")
    status = payload.get("status")
    detection = payload.get("detection")

    if not isinstance(job_id, str) or not job_id:
        raise ValueError("result.json must include a non-empty string job_id.")
    if not isinstance(status, str) or not status:
        raise ValueError("result.json must include a non-empty string status.")
    if not isinstance(detection, dict):
        raise ValueError("result.json must include a detection object.")

    video_score = detection.get("video_score")
    if not isinstance(video_score, dict):
        raise ValueError("result.json must include detection.video_score.")

    for required_key in ("final_fake_score", "max_fake_score"):
        if required_key not in video_score:
            raise ValueError(
                f"result.json must include detection.video_score.{required_key}."
            )


def _validate_audio_result_payload(payload: dict[str, Any]) -> None:
    required_keys = (
        "audio_fake_prob_like",
        "audio_uncertainty",
        "evidence_level",
    )
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise ValueError(
            "audio result json must include "
            + ", ".join(missing_keys)
            + "."
        )


def _load_dotenv_file() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise Stage1UnavailableError(
            "Stage1 result explanation requires python-dotenv for .env auto-loading. Install services/backend/requirements.txt before calling this pipeline."
        ) from exc

    load_dotenv(dotenv_path=ENV_FILE_PATH, override=False)


def _load_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key

    _load_dotenv_file()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise Stage1UnavailableError(
            "Gemini API key is missing. Set GOOGLE_API_KEY or GEMINI_API_KEY before calling the Stage1 result explainer."
        )
    return api_key


def _create_genai_client(api_key: str):
    try:
        from google import genai
    except ImportError as exc:
        raise Stage1UnavailableError(
            "Stage1 result explanation requires the google-genai package. Install services/backend/requirements.txt before calling this pipeline."
        ) from exc

    return genai.Client(api_key=api_key)


def _generate_text(client: Any, prompt: str) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    text = getattr(response, "text", "")
    if not isinstance(text, str) or not text.strip():
        raise Stage1ExplanationError("Gemini response did not contain plain text output.")
    return text.strip()


def run_video_stage1_result_explainer(
    video_result_json_path: str,
    audio_result_json_path: str,
) -> dict[str, Any]:
    video_result_path = Path(video_result_json_path)
    if not video_result_path.exists():
        raise FileNotFoundError(f"video_result.json file not found: {video_result_path}")
    if not video_result_path.is_file():
        raise ValueError(f"video_result.json path must point to a file: {video_result_path}")

    audio_result_path = Path(audio_result_json_path)
    if not audio_result_path.exists():
        raise FileNotFoundError(f"audio_result.json file not found: {audio_result_path}")
    if not audio_result_path.is_file():
        raise ValueError(f"audio_result.json path must point to a file: {audio_result_path}")

    video_payload_obj = read_json(video_result_path)
    if not isinstance(video_payload_obj, dict):
        raise ValueError("video result json must contain a JSON object.")
    video_payload: dict[str, Any] = dict(video_payload_obj)
    _validate_result_payload(video_payload)

    audio_payload_obj = read_json(audio_result_path)
    if not isinstance(audio_payload_obj, dict):
        raise ValueError("audio result json must contain a JSON object.")
    audio_payload: dict[str, Any] = dict(audio_payload_obj)
    _validate_audio_result_payload(audio_payload)

    video_prompt_source: dict[str, Any] = {
        str(key): value for key, value in video_payload.items() if key != "llm_explanations"
    }
    summary_prompt = _build_prompt(
        _load_prompt_template(SUMMARY_PROMPT_PATH),
        video_prompt_source,
        audio_payload,
    )
    detail_prompt = _build_prompt(
        _load_prompt_template(DETAIL_PROMPT_PATH),
        video_prompt_source,
        audio_payload,
    )

    api_key = _load_api_key()
    client = _create_genai_client(api_key)
    summary_text = _generate_text(client, summary_prompt)
    detail_text = _generate_text(client, detail_prompt)

    video_payload["llm_explanations"] = {
        "model": GEMINI_MODEL,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary_text": summary_text,
        "detail_text": detail_text,
    }

    write_json(video_result_path, video_payload)
    return video_payload
