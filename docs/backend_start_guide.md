## 빌드 과정

### 1. 파이썬 가상환경 생성 및 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 패키지 다운

#### 2-1. backend 기본 실행 패키지

```bash
pip install -r services/backend/requirements.txt
```

#### 2-2. Stage1 AI 런타임 추가 설치

Stage1 A/B 엔드포인트까지 실제로 실행하려면 아래 AI 런타임을 추가로 설치해야 합니다.

```bash
pip install -r services/backend/requirements-ai-stage1.txt
```

RetinaFace / TensorFlow 런타임을 쓸 때는 환경 변수도 함께 설정합니다.

```bash
# macOS / Linux
export TF_USE_LEGACY_KERAS=1

# Windows PowerShell
$env:TF_USE_LEGACY_KERAS=1
```

### 3. 서버 실행 명령어
**실행할 떄 항상 가상환경 안에서 실행할 것**
```bash
# 환경에 따라 접속 안될 수도 있음.
# wsl은 끝에 --host 0.0.0.0 추가
uvicorn services.backend.main:app --reload
```

### 4. Stage1 backend 동작 계약

- backend 기본 기동 자체는 Stage1 AI 런타임 없이도 가능해야 합니다.
- `/media/video-stage1/preprocess` 와 `/api/v1/video-stage1/preprocess` 는 프로젝트 내부 실제 파일 경로만 허용합니다.
- `job_id` 는 `../escape` 같은 경로 이탈 문자열을 허용하지 않습니다.
- Stage1 A AI 런타임이 없거나 깨졌으면 HTTP 500과 고정 메시지로 응답합니다.
- 그 외 예상 못한 예외는 내부 상세 에러를 그대로 노출하지 않고 일반적인 500 메시지로 응답합니다.

**Swagger: http://localhost:8000/docs**
