# VeriFake Video Pipeline Stage 1 - A 작업 문서

## 0. 문서 목적

이 문서는 `Wison05/verifake` 레포 기준으로 영상 파이프라인 1단계 중 **A 담당자 작업**을 정리한 문서입니다.

A 담당자는 **영상 입력, mp4 정규화, 프레임 샘플링, 얼굴 검출, 얼굴 crop 저장, 품질 메트릭 계산, `preprocessing.json` 생성**까지만 담당합니다.

B 담당자는 이 문서에서 정의한 `preprocessing.json`을 입력으로 받아 DeepfakeBench + EfficientNet-B4 추론, 구간 병합, 결과 요약을 담당합니다.

---

## 1. 이번 작업 범위

### 포함 범위

A 담당자가 해야 하는 일은 다음과 같습니다.

1. 입력 영상 경로 확인
2. `job_id` 기준 작업 폴더 생성
3. 원본 영상 저장 또는 참조
4. 내부 처리용 `normalized.mp4` 생성
5. 영상 메타데이터 추출
6. 3fps 기준 프레임 샘플링
7. 샘플 프레임 저장
8. 사람 얼굴 검출
9. 얼굴 crop 저장
10. 프레임별 얼굴 메타데이터 저장
11. 품질 메트릭 계산
12. `metadata/preprocessing.json` 생성

### 제외 범위

이번 1단계 A 작업에서는 아래 기능을 만들지 않습니다.

1. DeepfakeBench 추론
2. EfficientNet-B4 점수 계산
3. fake score 생성
4. 의심 구간 병합
5. ViT 재검사
6. MINTIME 실행
7. attention map 생성
8. LLM 설명 생성
9. unsupported-content / low-evidence 라우팅 고도화

---

## 2. 현재 레포 구조 기준 정리

현재 레포는 아래 구조를 기준으로 작업하는 것이 적합합니다.

```txt
verifake/
  docs/
  infra/
  apps/
    mobile/
  services/
    backend/
      main.py
      processor.py
      routers/
        media.py
      requirements.txt
    ai/
      README.md
      deepfakebench/
      antideepfake/
      audio_pipeline/
    security/
```

현재 `services/backend/processor.py`에는 영상/음성 분리용 `separate_streams()`가 있고, `storage/video`, `storage/audio`를 사용합니다.

이번 영상 1단계에서는 기존 구조를 완전히 갈아엎지 말고, 아래처럼 확장하는 것이 안전합니다.

```txt
verifake/
  services/
    ai/
      common/
        __init__.py
        job_paths.py
        json_io.py
        video_probe.py
      pipelines/
        __init__.py
        video_stage1/
          __init__.py
          preprocess.py
          face_detect.py
          quality.py
          frame_sampler.py
          schema.py
      inference/
        __init__.py
        README.md
```

A 담당자는 주로 아래 위치에 코드를 추가합니다.

```txt
services/ai/common/
services/ai/pipelines/video_stage1/
```

백엔드 API와 연결할 때만 아래 파일을 최소 수정합니다.

```txt
services/backend/processor.py
services/backend/routers/media.py
```

---

## 3. 병렬 작업 전에 A/B가 먼저 정해야 하는 공통 계약

아래 내용은 A와 B가 반드시 동일하게 사용해야 합니다.

### 3.1 작업 단위

| 항목 | 고정값 |
|---|---|
| 작업 ID 필드명 | `job_id` |
| 스키마 버전 | `video-stage1-v1` |
| 내부 영상 포맷 | `mp4` |
| 전처리 결과 파일 | `metadata/preprocessing.json` |
| 탐지 결과 파일 | `output/detection.json` |
| 최종 결과 파일 | `output/result.json` |
| 상태값 | `success`, `partial_success`, `failed` |

---

### 3.2 저장 폴더 구조

기존 `storage/video`, `storage/audio` 구조가 있으므로, 새 Stage 1 작업물은 `storage/jobs/{job_id}` 아래에 모읍니다.

```txt
storage/
  jobs/
    {job_id}/
      input/
        original.mp4
        normalized.mp4
      frames/
        frame_000000.jpg
        frame_000001.jpg
      faces/
        frame_000000_face_00.jpg
        frame_000001_face_00.jpg
      metadata/
        preprocessing.json
      output/
        detection.json
        result.json
      logs/
        preprocess.log
        detection.log
```

A 담당자는 아래 파일과 폴더까지만 생성합니다.

```txt
storage/jobs/{job_id}/input/original.*
storage/jobs/{job_id}/input/normalized.mp4
storage/jobs/{job_id}/frames/*.jpg
storage/jobs/{job_id}/faces/*.jpg
storage/jobs/{job_id}/metadata/preprocessing.json
storage/jobs/{job_id}/logs/preprocess.log
```

B 담당자는 A가 만든 `preprocessing.json`과 `faces/*.jpg`를 읽습니다.

---

### 3.3 공통 config 파일

추천 파일 위치:

```txt
services/ai/pipelines/video_stage1/config.stage1.json
```

```json
{
  "schema_version": "video-stage1-v1",
  "pipeline_stage": 1,
  "internal_video_format": "mp4",
  "sample_fps": 3.0,
  "max_video_duration_sec": 300,
  "frame_image_format": "jpg",
  "face_crop_format": "jpg",
  "bbox_format": "xyxy",
  "timestamp_unit": "seconds",
  "score_range": [0.0, 1.0],
  "storage_root": "storage/jobs",
  "paths": {
    "job_root": "storage/jobs/{job_id}",
    "input_dir": "storage/jobs/{job_id}/input",
    "frames_dir": "storage/jobs/{job_id}/frames",
    "faces_dir": "storage/jobs/{job_id}/faces",
    "metadata_dir": "storage/jobs/{job_id}/metadata",
    "output_dir": "storage/jobs/{job_id}/output",
    "logs_dir": "storage/jobs/{job_id}/logs"
  },
  "preprocessing_output": "storage/jobs/{job_id}/metadata/preprocessing.json",
  "detection_output": "storage/jobs/{job_id}/output/detection.json",
  "final_output": "storage/jobs/{job_id}/output/result.json"
}
```

---

## 4. A가 B에게 넘겨야 하는 정보

B가 추론을 바로 돌리려면 A는 최소한 아래 정보를 반드시 만들어야 합니다.

| 구분 | 필드명 | 설명 |
|---|---|---|
| 작업 ID | `job_id` | 영상 1개 처리 단위 |
| 영상 경로 | `input.normalized_video_path` | 내부 처리용 mp4 경로 |
| 영상 메타 | `video_metadata` | width, height, fps, duration 등 |
| 품질 지표 | `quality_metrics` | 신뢰도 판단용 값 |
| 얼굴 요약 | `face_summary` | 얼굴 검출 여부, 비율, 안정성 |
| 프레임 목록 | `frames[]` | 샘플링된 프레임 정보 |
| 얼굴 목록 | `frames[].faces[]` | 얼굴 bbox, crop 경로 |
| crop 경로 | `crop_path` | B가 추론할 이미지 경로 |
| bbox | `bbox` | `[x1, y1, x2, y2]` |
| timestamp | `timestamp_sec` | 초 단위 float |

---

## 5. A 작업 결과 JSON 스키마

파일 위치:

```txt
storage/jobs/{job_id}/metadata/preprocessing.json
```

예시:

```json
{
  "schema_version": "video-stage1-v1",
  "job_id": "job_20260427_0001",
  "pipeline_stage": 1,
  "status": "success",
  "created_at": "2026-04-27T23:55:00+09:00",
  "input": {
    "original_video_path": "storage/jobs/job_20260427_0001/input/original.mov",
    "normalized_video_path": "storage/jobs/job_20260427_0001/input/normalized.mp4",
    "original_extension": "mov",
    "internal_format": "mp4"
  },
  "video_metadata": {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "duration_sec": 18.4,
    "total_frame_count": 552,
    "sample_fps": 3.0,
    "sampled_frame_count": 55
  },
  "quality_metrics": {
    "face_detect_ratio": 0.82,
    "face_visibility_ratio": 0.75,
    "avg_face_size_ratio": 0.18,
    "min_face_size_ratio": 0.09,
    "max_face_size_ratio": 0.28,
    "blur_score": 0.21,
    "motion_blur_ratio": 0.08,
    "dark_frame_ratio": 0.03,
    "compression_artifact_score": 0.17
  },
  "face_summary": {
    "human_face_detected": true,
    "face_detect_failed_frame_count": 10,
    "max_face_count_per_frame": 1,
    "avg_face_count_per_frame": 0.82,
    "multi_face_flag": false,
    "face_track_stability": 0.81
  },
  "frames": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "frame_path": "storage/jobs/job_20260427_0001/frames/frame_000000.jpg",
      "face_count": 1,
      "faces": [
        {
          "face_id": "face_000000_00",
          "face_index": 0,
          "detected": true,
          "bbox": [120, 80, 300, 280],
          "bbox_area_ratio": 0.0174,
          "detection_confidence": 0.96,
          "crop_path": "storage/jobs/job_20260427_0001/faces/frame_000000_face_00.jpg"
        }
      ]
    }
  ],
  "errors": []
}
```

---

## 6. A 추천 파일 구성

아래 구조로 만들면 B와 충돌이 적습니다.

```txt
services/ai/pipelines/video_stage1/
  __init__.py
  preprocess.py
  frame_sampler.py
  face_detect.py
  quality.py
  schema.py
  config.stage1.json

services/ai/common/
  __init__.py
  job_paths.py
  json_io.py
  video_probe.py
```

### 파일별 역할

| 파일 | 역할 |
|---|---|
| `preprocess.py` | A 전체 파이프라인 실행 진입점 |
| `frame_sampler.py` | 3fps 프레임 샘플링 |
| `face_detect.py` | 얼굴 검출 및 crop 저장 |
| `quality.py` | 품질 메트릭 계산 |
| `schema.py` | `preprocessing.json` 구조 정의 |
| `job_paths.py` | `storage/jobs/{job_id}` 경로 생성 |
| `json_io.py` | JSON 저장/로드 유틸 |
| `video_probe.py` | ffprobe 기반 영상 메타 추출 |

---

## 7. A 구현 함수명 추천

```python
# services/ai/pipelines/video_stage1/preprocess.py

def run_video_stage1_preprocess(input_path: str, job_id: str | None = None) -> dict:
    """
    입력 영상을 받아 Stage 1 전처리를 수행하고 preprocessing.json dict를 반환한다.
    """
```

```python
# services/ai/common/job_paths.py

def create_job_dirs(job_id: str) -> dict:
    """
    storage/jobs/{job_id} 하위 폴더를 생성하고 경로 dict를 반환한다.
    """
```

```python
# services/ai/common/video_probe.py

def probe_video(video_path: str) -> dict:
    """
    ffprobe 또는 OpenCV로 width, height, fps, duration, frame_count를 추출한다.
    """
```

```python
# services/ai/pipelines/video_stage1/frame_sampler.py

def sample_frames(video_path: str, frames_dir: str, sample_fps: float = 3.0) -> list[dict]:
    """
    normalized.mp4에서 프레임을 샘플링하고 frame_path, frame_index, timestamp_sec를 반환한다.
    """
```

```python
# services/ai/pipelines/video_stage1/face_detect.py

def detect_and_crop_faces(frames: list[dict], faces_dir: str) -> list[dict]:
    """
    프레임별 얼굴을 검출하고 crop 이미지를 저장한 뒤 frames 구조에 faces 배열을 채운다.
    """
```

```python
# services/ai/pipelines/video_stage1/quality.py

def calculate_quality_metrics(video_metadata: dict, frames: list[dict]) -> dict:
    """
    품질 지표를 계산한다. 이 값은 fake 판정 근거가 아니라 신뢰도 판단용이다.
    """
```

---

## 8. mp4 정규화 규칙

이번 기준은 “입력은 다양할 수 있지만 내부 처리용 영상은 무조건 mp4”입니다.

### 입력 허용

```txt
.mp4
.mov
.mkv
.avi
.webm
```

### 내부 출력

```txt
storage/jobs/{job_id}/input/normalized.mp4
```

### ffmpeg 추천 규칙

처음에는 안정성을 위해 재인코딩을 추천합니다.

```bash
ffmpeg -y \
  -i input_video \
  -an \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  storage/jobs/{job_id}/input/normalized.mp4
```

기존 `processor.py`는 `-c:v copy` 방식으로 영상만 분리하지만, Stage 1 분석용 working copy는 코덱 호환성을 위해 `libx264`로 맞추는 편이 안전합니다.

---

## 9. 품질 메트릭 기준

`quality_metrics`는 가짜 판정 점수가 아닙니다. B가 결과 신뢰도를 해석할 때 참고하는 값입니다.

| 필드명 | 설명 | 추천 계산 방식 |
|---|---|---|
| `face_detect_ratio` | 얼굴이 검출된 샘플 프레임 비율 | 얼굴 검출 프레임 수 / 샘플 프레임 수 |
| `face_visibility_ratio` | 얼굴이 충분히 보이는 비율 | bbox 크기와 confidence 기준 |
| `avg_face_size_ratio` | 평균 얼굴 bbox 면적 비율 | bbox_area / frame_area 평균 |
| `min_face_size_ratio` | 최소 얼굴 크기 비율 | bbox_area_ratio 최소값 |
| `max_face_size_ratio` | 최대 얼굴 크기 비율 | bbox_area_ratio 최대값 |
| `blur_score` | 흐림 정도 | Laplacian variance 기반 정규화 |
| `motion_blur_ratio` | 움직임 흐림 프레임 비율 | 프레임 간 차이 또는 blur score 기준 |
| `dark_frame_ratio` | 너무 어두운 프레임 비율 | 평균 밝기 threshold 이하 비율 |
| `compression_artifact_score` | 압축 아티팩트 근사값 | 블록 경계 차이 또는 간단한 휴리스틱 |

---

## 10. face summary 계산 기준

```json
{
  "human_face_detected": true,
  "face_detect_failed_frame_count": 10,
  "max_face_count_per_frame": 1,
  "avg_face_count_per_frame": 0.82,
  "multi_face_flag": false,
  "face_track_stability": 0.81
}
```

| 필드명 | 기준 |
|---|---|
| `human_face_detected` | 하나 이상의 얼굴 crop이 있으면 `true` |
| `face_detect_failed_frame_count` | 얼굴이 0개인 샘플 프레임 수 |
| `max_face_count_per_frame` | 한 프레임에서 검출된 최대 얼굴 수 |
| `avg_face_count_per_frame` | 샘플 프레임 기준 평균 얼굴 수 |
| `multi_face_flag` | 한 프레임이라도 얼굴 2개 이상이면 `true` |
| `face_track_stability` | 인접 프레임 bbox 중심 이동량/크기 변화량 기반 0~1 값 |

---

## 11. A 실행 명령 추천

직접 실행용 CLI는 아래처럼 잡으면 좋습니다.

```bash
python -m services.ai.pipelines.video_stage1.preprocess \
  --input ./samples/test.mov \
  --job-id job_20260427_0001
```

성공 시 생성되어야 하는 파일:

```txt
storage/jobs/job_20260427_0001/metadata/preprocessing.json
```

---

## 12. 백엔드 연결 방식

현재 `services/backend/main.py`는 FastAPI 앱을 만들고 `/media` 라우터를 연결하고 있습니다.

A 작업을 API로 연결하려면 기존 `/media/split`은 유지하고, 새 엔드포인트를 추가하는 방식이 안전합니다.

추천 추가 엔드포인트:

```txt
POST /media/video-stage1/preprocess
```

요청 예시:

```json
{
  "file_path": "samples/test.mov"
}
```

응답 예시:

```json
{
  "job_id": "job_20260427_0001",
  "status": "success",
  "preprocessing_json": "storage/jobs/job_20260427_0001/metadata/preprocessing.json"
}
```

---

## 13. A 완료 조건

아래가 모두 되면 A 작업은 완료입니다.

```txt
[ ] 입력 파일 존재 확인 가능
[ ] job_id 생성 또는 외부 입력 가능
[ ] storage/jobs/{job_id} 구조 생성 가능
[ ] original 파일 저장 또는 복사 가능
[ ] normalized.mp4 생성 가능
[ ] video_metadata 계산 가능
[ ] 3fps 샘플링 가능
[ ] frames/*.jpg 저장 가능
[ ] 얼굴 검출 가능
[ ] faces/*.jpg 저장 가능
[ ] frames[].faces[] 구조 생성 가능
[ ] quality_metrics 계산 가능
[ ] face_summary 계산 가능
[ ] preprocessing.json 저장 가능
[ ] B가 preprocessing.json만 보고 crop_path를 읽을 수 있음
```

---

## 14. A가 절대 하지 말아야 할 것

```txt
[ ] raw_fake_score 생성 금지
[ ] frame_scores 생성 금지
[ ] segment_scores 생성 금지
[ ] video_score 생성 금지
[ ] DeepfakeBench 모델 추론 금지
[ ] EfficientNet-B4 wrapper 구현 금지
[ ] ViT, MINTIME, attention map 구현 금지
[ ] LLM 요약 구현 금지
```

---

## 15. B와 맞춰야 하는 최종 체크

A 작업 완료 후 B에게 아래 3개만 전달하면 됩니다.

```txt
1. job_id
2. storage/jobs/{job_id}/metadata/preprocessing.json
3. storage/jobs/{job_id}/faces/*.jpg
```

B는 이 3개만 있으면 추론을 진행할 수 있어야 합니다.

---

## 16. 커밋 메시지 추천

```txt
feat(ai): add video stage1 preprocessing pipeline
```

또는 작업을 더 작게 나누면:

```txt
feat(ai): add stage1 job path utilities
feat(ai): add video frame sampler
feat(ai): add face crop preprocessing output
feat(ai): add preprocessing json schema
```
