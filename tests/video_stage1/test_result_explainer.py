from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


result_explainer = importlib.import_module(
    "services.ai.pipelines.video_stage1.result_explainer"
)
exceptions = importlib.import_module("services.ai.pipelines.video_stage1.exceptions")


def test_result_explainer_updates_result_json(tmp_path: Path, monkeypatch) -> None:
    video_result_path = tmp_path / "video_result.json"
    audio_result_path = tmp_path / "audio_result.json"
    result_payload = {
        "job_id": "job_test_101",
        "status": "success",
        "detection": {
            "video_score": {
                "final_fake_score": 0.72,
                "max_fake_score": 0.91,
            },
            "top_segments": [
                {"start_sec": 3.0, "end_sec": 5.0, "reason": "high score"}
            ],
        },
    }
    audio_result_payload = {
        "request_id": "audio_req_101",
        "audio_fake_prob_like": 0.86,
        "audio_uncertainty": 0.12,
        "evidence_level": "sufficient",
    }

    video_result_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audio_result_path.write_text(
        json.dumps(audio_result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "result_summary_prompt.txt").write_text(
        "summary\n{{VIDEO_RESULT_JSON}}\n---\n{{AUDIO_RESULT_JSON}}\n",
        encoding="utf-8",
    )
    (prompts_dir / "result_detail_prompt.txt").write_text(
        "detail\n{{VIDEO_RESULT_JSON}}\n---\n{{AUDIO_RESULT_JSON}}\n",
        encoding="utf-8",
    )

    calls: list[str] = []

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeModels:
        def generate_content(self, *, model: str, contents: str) -> FakeResponse:
            calls.append(contents)
            assert model == "gemini-2.5-flash-lite"
            if contents.startswith("summary"):
                return FakeResponse("요약 응답")
            return FakeResponse("상세 응답")

    class FakeClient:
        def __init__(self) -> None:
            self.models = FakeModels()

    monkeypatch.setattr(result_explainer, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(
        result_explainer,
        "SUMMARY_PROMPT_PATH",
        prompts_dir / "result_summary_prompt.txt",
    )
    monkeypatch.setattr(
        result_explainer,
        "DETAIL_PROMPT_PATH",
        prompts_dir / "result_detail_prompt.txt",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(
        result_explainer,
        "_create_genai_client",
        lambda api_key: FakeClient(),
    )

    updated = result_explainer.run_video_stage1_result_explainer(
        str(video_result_path),
        str(audio_result_path),
    )

    assert updated["llm_explanations"]["model"] == "gemini-2.5-flash-lite"
    assert updated["llm_explanations"]["summary_text"] == "요약 응답"
    assert updated["llm_explanations"]["detail_text"] == "상세 응답"
    assert len(calls) == 2
    assert '"final_fake_score": 0.72' in calls[0]
    assert '"audio_fake_prob_like": 0.86' in calls[0]

    persisted = json.loads(video_result_path.read_text(encoding="utf-8"))
    assert persisted["llm_explanations"]["summary_text"] == "요약 응답"
    assert persisted["llm_explanations"]["detail_text"] == "상세 응답"


def test_result_explainer_requires_api_key(tmp_path: Path, monkeypatch) -> None:
    video_result_path = tmp_path / "video_result.json"
    audio_result_path = tmp_path / "audio_result.json"
    video_result_path.write_text(
        json.dumps(
            {
                "job_id": "job_test_103",
                "status": "success",
                "detection": {
                    "video_score": {
                        "final_fake_score": 0.5,
                        "max_fake_score": 0.6,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    audio_result_path.write_text(
        json.dumps(
            {
                "request_id": "audio_req_103",
                "audio_fake_prob_like": 0.45,
                "audio_uncertainty": 0.2,
                "evidence_level": "sufficient",
            }
        ),
        encoding="utf-8",
    )

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "result_summary_prompt.txt").write_text(
        "summary\n{{VIDEO_RESULT_JSON}}\n{{AUDIO_RESULT_JSON}}\n",
        encoding="utf-8",
    )
    (prompts_dir / "result_detail_prompt.txt").write_text(
        "detail\n{{VIDEO_RESULT_JSON}}\n{{AUDIO_RESULT_JSON}}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(result_explainer, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(
        result_explainer,
        "SUMMARY_PROMPT_PATH",
        prompts_dir / "result_summary_prompt.txt",
    )
    monkeypatch.setattr(
        result_explainer,
        "DETAIL_PROMPT_PATH",
        prompts_dir / "result_detail_prompt.txt",
    )
    monkeypatch.setattr(result_explainer, "ENV_FILE_PATH", tmp_path / ".missing.env")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(exceptions.Stage1UnavailableError):
        result_explainer.run_video_stage1_result_explainer(
            str(video_result_path),
            str(audio_result_path),
        )


def test_result_explainer_rejects_invalid_result_shape(tmp_path: Path, monkeypatch) -> None:
    video_result_path = tmp_path / "video_result.json"
    audio_result_path = tmp_path / "audio_result.json"
    video_result_path.write_text(
        json.dumps({"job_id": "job_test_102", "status": "success"}),
        encoding="utf-8",
    )
    audio_result_path.write_text(
        json.dumps(
            {
                "request_id": "audio_req_102",
                "audio_fake_prob_like": 0.45,
                "audio_uncertainty": 0.2,
                "evidence_level": "sufficient",
            }
        ),
        encoding="utf-8",
    )

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "result_summary_prompt.txt").write_text(
        "summary\n{{VIDEO_RESULT_JSON}}\n{{AUDIO_RESULT_JSON}}\n",
        encoding="utf-8",
    )
    (prompts_dir / "result_detail_prompt.txt").write_text(
        "detail\n{{VIDEO_RESULT_JSON}}\n{{AUDIO_RESULT_JSON}}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(result_explainer, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(
        result_explainer,
        "SUMMARY_PROMPT_PATH",
        prompts_dir / "result_summary_prompt.txt",
    )
    monkeypatch.setattr(
        result_explainer,
        "DETAIL_PROMPT_PATH",
        prompts_dir / "result_detail_prompt.txt",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    with pytest.raises(ValueError):
        result_explainer.run_video_stage1_result_explainer(
            str(video_result_path),
            str(audio_result_path),
        )


def test_result_explainer_loads_api_key_from_dotenv(tmp_path: Path, monkeypatch) -> None:
    video_result_path = tmp_path / "video_result.json"
    audio_result_path = tmp_path / "audio_result.json"
    result_payload = {
        "job_id": "job_test_104",
        "status": "success",
        "detection": {
            "video_score": {
                "final_fake_score": 0.65,
                "max_fake_score": 0.8,
            },
            "top_segments": [],
        },
    }
    audio_result_payload = {
        "request_id": "audio_req_104",
        "audio_fake_prob_like": 0.32,
        "audio_uncertainty": 0.61,
        "evidence_level": "low_evidence",
    }

    video_result_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audio_result_path.write_text(
        json.dumps(audio_result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "result_summary_prompt.txt").write_text(
        "summary\n{{VIDEO_RESULT_JSON}}\n{{AUDIO_RESULT_JSON}}\n",
        encoding="utf-8",
    )
    (prompts_dir / "result_detail_prompt.txt").write_text(
        "detail\n{{VIDEO_RESULT_JSON}}\n{{AUDIO_RESULT_JSON}}\n",
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    env_path.write_text("GOOGLE_API_KEY=dotenv-test-key\n", encoding="utf-8")

    seen_api_keys: list[str] = []

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeModels:
        def generate_content(self, *, model: str, contents: str) -> FakeResponse:
            return FakeResponse("응답")

    class FakeClient:
        def __init__(self) -> None:
            self.models = FakeModels()

    monkeypatch.setattr(result_explainer, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(
        result_explainer,
        "SUMMARY_PROMPT_PATH",
        prompts_dir / "result_summary_prompt.txt",
    )
    monkeypatch.setattr(
        result_explainer,
        "DETAIL_PROMPT_PATH",
        prompts_dir / "result_detail_prompt.txt",
    )
    monkeypatch.setattr(result_explainer, "ENV_FILE_PATH", env_path)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        result_explainer,
        "_create_genai_client",
        lambda api_key: seen_api_keys.append(api_key) or FakeClient(),
    )

    updated = result_explainer.run_video_stage1_result_explainer(
        str(video_result_path),
        str(audio_result_path),
    )

    assert seen_api_keys == ["dotenv-test-key"]
    assert updated["llm_explanations"]["summary_text"] == "응답"
    assert updated["llm_explanations"]["detail_text"] == "응답"


def test_result_explainer_rejects_invalid_audio_result_shape(tmp_path: Path, monkeypatch) -> None:
    video_result_path = tmp_path / "video_result.json"
    audio_result_path = tmp_path / "audio_result.json"
    video_result_path.write_text(
        json.dumps(
            {
                "job_id": "job_test_105",
                "status": "success",
                "detection": {
                    "video_score": {
                        "final_fake_score": 0.41,
                        "max_fake_score": 0.6,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    audio_result_path.write_text(
        json.dumps({"request_id": "audio_req_105"}),
        encoding="utf-8",
    )

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "result_summary_prompt.txt").write_text(
        "summary\n{{VIDEO_RESULT_JSON}}\n{{AUDIO_RESULT_JSON}}\n",
        encoding="utf-8",
    )
    (prompts_dir / "result_detail_prompt.txt").write_text(
        "detail\n{{VIDEO_RESULT_JSON}}\n{{AUDIO_RESULT_JSON}}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(result_explainer, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(
        result_explainer,
        "SUMMARY_PROMPT_PATH",
        prompts_dir / "result_summary_prompt.txt",
    )
    monkeypatch.setattr(
        result_explainer,
        "DETAIL_PROMPT_PATH",
        prompts_dir / "result_detail_prompt.txt",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    with pytest.raises(ValueError):
        result_explainer.run_video_stage1_result_explainer(
            str(video_result_path),
            str(audio_result_path),
        )
