# Backend Audio Analysis Integration Design

**Goal:** 기존 FastAPI backend 구조를 유지한 채, 별도 Python 3.9 AI 런타임에서 `audio_stage1`을 subprocess로 실행하고 결과를 backend API로 노출한다.

## Current Reality

- `services/backend/main.py`는 현재 `media.router`와 `download_service.router`만 등록한다.
- `POST /media/split`는 `storage/video/{job_id}_video.mp4`, `storage/audio/{job_id}_audio.wav`를 생성하는 분리 작업만 한다.
- `services/ai/audio_pipeline/audio_stage1.py`는 이미 end-to-end CLI로 동작하며, 최종 결과를 `audio_stage1_result.json`으로 저장한다.
- 현재 backend는 audio pipeline을 전혀 호출하지 않는다.

## Design Choice

### 1. AI runtime separation

- backend Python과 AI Python은 분리한다.
- backend는 FastAPI 실행용 runtime 그대로 유지한다.
- AI 추론은 `VERIFAKE_AI_PYTHON` 환경변수에 지정된 Python 3.9 runtime으로 subprocess 호출한다.

예시:

```text
VERIFAKE_AI_PYTHON=C:\venvs\verifake-antideepfake-py39\Scripts\python.exe
```

이 값은 **필수**이며, 없거나 파일이 없거나 실행 불가일 경우 즉시 실패해야 한다.

### 2. API namespace separation

기존 `GET /api/v1/status/{task_id}`는 유지한다.

audio는 별도 namespace로 분리한다.

- `POST /api/v1/audio/jobs`
- `GET /api/v1/audio/jobs/{task_id}`
- `GET /api/v1/audio/jobs/{task_id}/result`

이렇게 해야 기존 upload/video status 라우트와 충돌하지 않는다.

### 3. Shared task store

현재 `tasks_db`는 `services/backend/download_service.py` 안에만 있다.

이를 `services/backend/tasks.py`로 이동해 공유한다.

단, 기존 upload/video 흐름과 새 audio job 흐름은 서로 구분해서 저장한다.

권장 구조:

```python
upload_tasks_db: dict[str, dict]
audio_jobs_db: dict[str, dict]
```

이렇게 하면 기존 `/api/v1/status/{task_id}` 응답 계약을 깨지 않고 audio job 상태만 별도 관리할 수 있다.

## Runtime Flow

### POST /api/v1/audio/jobs

1. `file_path`를 받는다.
2. 입력 파일 존재 여부를 확인한다.
3. `VERIFAKE_AI_PYTHON` 환경변수와 실행 파일 경로를 검증한다.
4. `task_id`를 생성한다.
5. `audio_jobs_db[task_id]`를 `PENDING` 상태로 만든다.
6. FastAPI `BackgroundTasks`로 실제 작업을 enqueue 한다.
7. `202 Accepted`를 반환한다.

### Background task

1. 상태를 `SPLITTING`으로 바꾼다.
2. `processor.separate_streams()`를 호출해 `storage/audio/{job_id}_audio.wav`를 만든다.
3. 상태를 `ANALYZING`으로 바꾼다.
4. output dir: `storage/jobs/{job_id}/audio`
5. subprocess로 실행:

```text
VERIFAKE_AI_PYTHON -m services.ai.audio_pipeline.audio_stage1 \
  --input storage/audio/{job_id}_audio.wav \
  --output-dir storage/jobs/{job_id}/audio \
  --request-id {job_id}
```

6. timeout, stdout/stderr capture를 적용한다.
7. 성공 시 `storage/jobs/{job_id}/audio/audio_stage1_result.json`을 읽고 상태를 `SUCCEEDED`로 갱신한다.
8. 실패 시 `FAILED` 또는 `TIMED_OUT`으로 갱신하고 error summary를 저장한다.

### GET /api/v1/audio/jobs/{task_id}

일반 status 응답이다.

반환 필드 예시:

```json
{
  "task_id": "...",
  "status": "ANALYZING",
  "stage": "audio_stage1",
  "result_path": null,
  "error": null,
  "timestamp": "..."
}
```

여기서는 `stdout` / `stderr` 전체를 노출하지 않고, 요약된 `error`만 반환한다.

### GET /api/v1/audio/jobs/{task_id}/result

- 모르는 task id면 `404`
- 아직 완료 전이면 `409`
- 성공이면 `audio_stage1_result.json`의 parsed JSON을 반환한다.

## Error Handling

다음은 전부 명확한 실패 상태로 저장해야 한다.

- `VERIFAKE_AI_PYTHON` 미설정
- 환경변수 경로가 존재하지 않음
- 실행 권한/호출 불가
- media split 실패
- subprocess non-zero exit
- subprocess timeout
- result JSON 누락 또는 파싱 실패

`audio_jobs_db` 내부에는 아래 정보를 남긴다.

- `status`
- `stage`
- `audio_path`
- `artifacts_dir`
- `result_path`
- `error`
- `stdout`
- `stderr`
- `returncode`
- `started_at`
- `finished_at`

다만 일반 status 응답에는 `stdout`/`stderr` 전체를 포함하지 않는다.

## Testing Scope

필수 테스트 범위:

1. shared task store 생성/조회/업데이트
2. 기존 `download_service` 흐름이 그대로 유지되는지
3. audio analyzer가 env var 미설정/잘못된 경로/timeout/non-zero exit를 올바르게 처리하는지
4. audio router가 `202/404/409`를 올바르게 반환하는지
5. `main.py`에 새 router가 등록되더라도 기존 `/api/v1/status/{task_id}`가 유지되는지

## Non-Goals

- DB 영속화
- Celery/Redis/queue 도입
- video AI inference 연동
- audio pipeline 내부 로직 수정

이 설계는 **현재 구조를 최소 변경으로 연결하는 v1 설계**다.
