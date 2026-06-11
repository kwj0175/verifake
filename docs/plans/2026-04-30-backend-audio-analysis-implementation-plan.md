# Backend Audio Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** FastAPI backend에서 별도 Python 3.9 AI runtime을 subprocess로 호출해 audio stage1 분석을 수행하고, status/result API로 결과를 제공한다.

**Architecture:** 기존 backend app 구조는 유지하고, shared task store + audio subprocess service + audio router를 추가한다. 기존 `/api/v1/status/{task_id}` 흐름은 보존하고, audio job은 `/api/v1/audio/jobs/...` 아래로 분리한다.

**Tech Stack:** FastAPI, BackgroundTasks, subprocess, pathlib, stdlib unittest, existing `services.ai.audio_pipeline.audio_stage1` CLI.

---

### Task 1: shared tasks store 도입

**Files:**
- Create: `services/backend/tasks.py`
- Modify: `services/backend/download_service.py`
- Test: `services/backend/tests/test_tasks.py`
- Test: `services/backend/tests/test_download_service.py`

**Step 1: Write the failing tests**

- `test_tasks.py`
  - upload task 생성 시 `PENDING`, `verdict=None`
  - audio job 생성 시 `status`, `file_path`, `artifacts_dir`, `result`, `error` 초기화
- `test_download_service.py`
  - 기존 `/api/v1/share` 흐름이 shared store를 써도 응답 shape가 안 깨지는지
  - 기존 `/api/v1/status/{task_id}`가 그대로 동작하는지

**Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest discover -s services/backend/tests -p "test_tasks.py"
python -m unittest discover -s services/backend/tests -p "test_download_service.py"
```

Expected: `tasks.py` 부재 또는 import failure.

**Step 3: Write minimal implementation**

- `tasks.py` 생성
- `upload_tasks_db`, `audio_jobs_db` 및 최소 helper 함수 추가
- `download_service.py`의 module-local `tasks_db` 제거 후 shared store import로 변경

**Step 4: Run tests to verify they pass**

```powershell
python -m unittest discover -s services/backend/tests -p "test_tasks.py"
python -m unittest discover -s services/backend/tests -p "test_download_service.py"
```

**Step 5: Commit**

```bash
git add services/backend/tasks.py services/backend/download_service.py services/backend/tests/test_tasks.py services/backend/tests/test_download_service.py
git commit -m "refactor: move backend task state into shared module"
```

### Task 2: audio subprocess service 추가

**Files:**
- Create: `services/backend/services/audio_analyzer.py`
- Test: `services/backend/tests/test_audio_analyzer.py`

**Step 1: Write the failing tests**

테스트 케이스:
- `VERIFAKE_AI_PYTHON` 미설정 시 실패
- 잘못된 interpreter path 시 실패
- command가 `-m services.ai.audio_pipeline.audio_stage1`와 expected args를 포함하는지
- subprocess 성공 시 `SUCCEEDED` + parsed result 저장
- timeout 시 `TIMED_OUT`
- non-zero exit 시 `FAILED`
- 결과 JSON 누락 시 `FAILED`

**Step 2: Run test to verify it fails**

```powershell
python -m unittest discover -s services/backend/tests -p "test_audio_analyzer.py"
```

Expected: module not found / import failure.

**Step 3: Write minimal implementation**

구현 요소:
- `get_audio_python()`
- `build_audio_stage1_command()`
- `run_audio_job(job_id, input_path)`
- timeout constant
- stdout/stderr capture
- `storage/jobs/{job_id}/audio` artifact dir 생성

**Step 4: Run test to verify it passes**

```powershell
python -m unittest discover -s services/backend/tests -p "test_audio_analyzer.py"
```

**Step 5: Commit**

```bash
git add services/backend/services/audio_analyzer.py services/backend/tests/test_audio_analyzer.py
git commit -m "feat: add audio analyzer subprocess service"
```

### Task 3: audio router와 app wiring 추가

**Files:**
- Create: `services/backend/routers/audio.py`
- Modify: `services/backend/main.py`
- Test: `services/backend/tests/test_audio_router.py`

**Step 1: Write the failing tests**

테스트 케이스:
- `POST /api/v1/audio/jobs`가 없는 상태에서 실패
- source file 없으면 `400`
- env var 검증 실패면 `500` 또는 `503`
- 성공 시 `202`
- `GET /api/v1/audio/jobs/{task_id}` returns summary
- `GET /api/v1/audio/jobs/{task_id}/result` returns `409` before success
- unknown task id returns `404`
- route registration 후 `/api/v1/status/{task_id}`와 충돌하지 않는지

**Step 2: Run test to verify it fails**

```powershell
python -m unittest discover -s services/backend/tests -p "test_audio_router.py"
```

Expected: router or route missing.

**Step 3: Write minimal implementation**

- `audio.py`에 `POST /jobs`, `GET /jobs/{task_id}`, `GET /jobs/{task_id}/result`
- `BackgroundTasks` 사용
- `main.py`에 `app.include_router(audio.router, prefix="/api/v1/audio")`

**Step 4: Run test to verify it passes**

```powershell
python -m unittest discover -s services/backend/tests -p "test_audio_router.py"
```

**Step 5: Commit**

```bash
git add services/backend/routers/audio.py services/backend/main.py services/backend/tests/test_audio_router.py
git commit -m "feat: add backend audio analysis job endpoints"
```

### Task 4: full backend verification

**Files:**
- Verify only

**Step 1: Run full backend tests**

```powershell
python -m unittest discover -s services/backend/tests -p "test_*.py"
```

**Step 2: Run route smoke check**

```powershell
python -c "from services.backend.main import app; print(sorted(route.path for route in app.routes if route.path.startswith('/api/v1/audio') or route.path.startswith('/api/v1/status')))"
```

Expected paths:
- `/api/v1/status/{task_id}`
- `/api/v1/audio/jobs`
- `/api/v1/audio/jobs/{task_id}`
- `/api/v1/audio/jobs/{task_id}/result`

**Step 3: Manual runtime smoke**

```powershell
$env:VERIFAKE_AI_PYTHON="C:\venvs\verifake-antideepfake-py39\Scripts\python.exe"
python -m uvicorn services.backend.main:app --reload
```

이후:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/audio/jobs" -ContentType "application/json" -Body '{"file_path":"C:\project\verifake\storage\audio\fixed_sample_audio.wav"}'
```

**Step 4: Verify artifact path**

Expected result path:

```text
storage/jobs/{task_id}/audio/audio_stage1_result.json
```

**Step 5: Commit**

```bash
git add .
git commit -m "test: verify backend audio analysis integration"
```

### Notes

- `VERIFAKE_AI_PYTHON`은 필수 환경변수다. fallback 없음.
- 정상 status 응답에는 `stdout`/`stderr`를 그대로 노출하지 않는다.
- 상세 로그는 in-memory task store에만 남긴다.
- result endpoint는 성공 전에는 `409`, 모르는 task는 `404`를 반환한다.
- 기존 `/api/v1/status/{task_id}` 흐름은 그대로 유지한다.
