# VeriFake Video Pipeline Stage 1 - B 작업 문서

## 0. 문서 목적

이 문서는 `Wison05/verifake` 레포 기준으로 영상 파이프라인 1단계 중 **B 담당자 작업**을 정리한 문서입니다.

B 담당자는 A가 만든 `metadata/preprocessing.json`을 입력으로 받아 **DeepfakeBench + EfficientNet-B4 추론, 얼굴/프레임 점수 계산, 의심 구간 병합, 최종 결과 요약 JSON 생성**을 담당합니다.

A 담당자는 영상 전처리와 얼굴 crop 저장까지만 담당합니다.

---

## 1. 이번 작업 범위

### 포함 범위

B 담당자가 해야 하는 일은 다음과 같습니다.

1. `preprocessing.json` 로드
2. 얼굴 crop 목록 추출
3. DeepfakeBench + EfficientNet-B4 inference wrapper 작성
4. 얼굴 crop 단위 `raw_fake_score` 계산
5. 프레임 단위 `frame_scores` 생성
6. 의심 프레임 병합으로 `segment_scores` 생성
7. 상위 의심 구간 `top_segments` 3~5개 추출
8. 영상 단위 `video_score` 계산
9. `output/detection.json` 생성
10. `output/result.json` 생성

### 제외 범위

이번 1단계 B 작업에서는 아래 기능을 만들지 않습니다.

1. mp4 변환
2. 프레임 샘플링
3. 얼굴 검출
4. 얼굴 crop 생성
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
        analysis/
        preprocessing/
        training/
      antideepfake/
      audio_pipeline/
    security/
```

기존 문서 기준으로 `services/ai/deepfakebench/`는 외부 기반 벤치마크 엔진 위치입니다. 따라서 B 담당자는 DeepfakeBench 원본 폴더 안을 직접 크게 수정하기보다, 서비스용 wrapper를 별도로 둡니다.

추천 위치:

```txt
services/ai/inference/video_stage1/
```

그리고 점수 병합/결과 요약 로직은 아래 위치에 둡니다.

```txt
services/ai/pipelines/video_stage1/
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
| 점수 범위 | `0.0 ~ 1.0` |
| fake score 필드명 | `raw_fake_score` |

---

### 3.2 저장 폴더 구조

B는 A가 만든 아래 구조를 그대로 읽습니다.

```txt
storage/
  jobs/
    {job_id}/
      input/
        original.mp4
        normalized.mp4
      frames/
        frame_000000.jpg
      faces/
        frame_000000_face_00.jpg
      metadata/
        preprocessing.json
      output/
        detection.json
        result.json
      logs/
        detection.log
```

B가 읽는 파일:

```txt
storage/jobs/{job_id}/metadata/preprocessing.json
storage/jobs/{job_id}/faces/*.jpg
storage/jobs/{job_id}/frames/*.jpg
```

B가 생성하는 파일:

```txt
storage/jobs/{job_id}/output/detection.json
storage/jobs/{job_id}/output/result.json
storage/jobs/{job_id}/logs/detection.log
```

---

### 3.3 공통 config 파일

추천 파일 위치:

```txt
services/ai/pipelines/video_stage1/config.stage1.json
```

B는 이 config에서 구간 병합 규칙을 읽습니다.

```json
{
  "schema_version": "video-stage1-v1",
  "pipeline_stage": 1,
  "score_range": [0.0, 1.0],
  "storage_root": "storage/jobs",
  "preprocessing_output": "storage/jobs/{job_id}/metadata/preprocessing.json",
  "detection_output": "storage/jobs/{job_id}/output/detection.json",
  "final_output": "storage/jobs/{job_id}/output/result.json",
  "detector": {
    "framework": "DeepfakeBench",
    "model_name": "EfficientNet-B4",
    "score_field": "raw_fake_score",
    "batch_size": 16
  },
  "segment_merge": {
    "score_threshold": 0.6,
    "max_gap_sec": 1.0,
    "min_segment_duration_sec": 0.5,
    "top_k": 5
  },
  "video_score": {
    "aggregation_method": "topk_mean",
    "topk_frame_count": 10
  }
}
```

---

## 4. B가 A로부터 받는 입력 JSON

파일 위치:

```txt
storage/jobs/{job_id}/metadata/preprocessing.json
```

B는 최소한 아래 필드를 사용합니다.

| 입력 필드 | 사용 목적 |
|---|---|
| `job_id` | 결과 파일 저장 위치 결정 |
| `input.normalized_video_path` | 원본 영상 참조용 |
| `video_metadata.duration_sec` | 구간 계산 검증 |
| `video_metadata.sample_fps` | 프레임 간격 판단 |
| `quality_metrics` | 최종 result에 포함 |
| `face_summary` | 최종 result에 포함 |
| `frames[].frame_index` | 프레임 점수 매핑 |
| `frames[].timestamp_sec` | 구간 시작/종료 계산 |
| `frames[].frame_path` | 대표 프레임 경로 저장 |
| `frames[].faces[].face_id` | 얼굴 점수 매핑 |
| `frames[].faces[].crop_path` | 모델 입력 이미지 |
| `frames[].faces[].bbox` | 결과 디버깅용 |

---

## 5. B 작업 결과 JSON 스키마

파일 위치:

```txt
storage/jobs/{job_id}/output/detection.json
```

예시:

```json
{
  "schema_version": "video-stage1-v1",
  "job_id": "job_20260427_0001",
  "pipeline_stage": 1,
  "status": "success",
  "created_at": "2026-04-27T23:58:00+09:00",
  "detector": {
    "framework": "DeepfakeBench",
    "model_name": "EfficientNet-B4",
    "detector_type": "face_artifact_detector",
    "score_type": "fake_raw_score",
    "score_range": [0.0, 1.0]
  },
  "inference_summary": {
    "analyzed_face_crop_count": 45,
    "analyzed_frame_count": 45,
    "skipped_frame_count": 10,
    "inference_status": "success"
  },
  "face_scores": [
    {
      "face_id": "face_000000_00",
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "crop_path": "storage/jobs/job_20260427_0001/faces/frame_000000_face_00.jpg",
      "raw_fake_score": 0.21,
      "inference_success": true
    }
  ],
  "frame_scores": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "face_count": 1,
      "max_fake_score": 0.21,
      "avg_fake_score": 0.21,
      "score_source": "face_scores"
    }
  ],
  "segment_scores": [
    {
      "segment_id": "seg_0001",
      "start_sec": 4.333,
      "end_sec": 6.0,
      "duration_sec": 1.667,
      "frame_count": 6,
      "max_fake_score": 0.86,
      "avg_fake_score": 0.74,
      "representative_frame_index": 16,
      "representative_frame_path": "storage/jobs/job_20260427_0001/frames/frame_000016.jpg"
    }
  ],
  "top_segments": [
    {
      "rank": 1,
      "segment_id": "seg_0001",
      "start_sec": 4.333,
      "end_sec": 6.0,
      "segment_score": 0.86,
      "reason": "high consecutive face artifact scores"
    }
  ],
  "video_score": {
    "max_fake_score": 0.86,
    "topk_mean_fake_score": 0.74,
    "avg_fake_score": 0.38,
    "final_fake_score": 0.74,
    "aggregation_method": "topk_mean"
  },
  "errors": []
}
```

---

## 6. 최종 통합 result JSON

파일 위치:

```txt
storage/jobs/{job_id}/output/result.json
```

B는 `preprocessing.json`과 `detection.json`을 합쳐서 화면/API 응답용 결과를 만듭니다.

```json
{
  "schema_version": "video-stage1-v1",
  "job_id": "job_20260427_0001",
  "pipeline_stage": 1,
  "status": "success",
  "input": {
    "normalized_video_path": "storage/jobs/job_20260427_0001/input/normalized.mp4"
  },
  "video_metadata": {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "duration_sec": 18.4,
    "sample_fps": 3.0
  },
  "quality_metrics": {
    "face_detect_ratio": 0.82,
    "face_visibility_ratio": 0.75,
    "avg_face_size_ratio": 0.18,
    "blur_score": 0.21,
    "motion_blur_ratio": 0.08,
    "dark_frame_ratio": 0.03,
    "compression_artifact_score": 0.17
  },
  "face_summary": {
    "human_face_detected": true,
    "multi_face_flag": false,
    "face_track_stability": 0.81
  },
  "detection": {
    "detector": "DeepfakeBench + EfficientNet-B4",
    "video_score": {
      "final_fake_score": 0.74,
      "max_fake_score": 0.86,
      "aggregation_method": "topk_mean"
    },
    "top_segments": [
      {
        "rank": 1,
        "start_sec": 4.333,
        "end_sec": 6.0,
        "segment_score": 0.86,
        "representative_frame_path": "storage/jobs/job_20260427_0001/frames/frame_000016.jpg"
      }
    ]
  },
  "stage1_note": "quality_metrics are used for reliability/context, not as direct fake evidence."
}
```

---

## 7. B 추천 파일 구성

```txt
services/ai/inference/
  __init__.py
  video_stage1/
    __init__.py
    deepfakebench_efficientnet_b4.py
    model_loader.py
    transforms.py

services/ai/pipelines/video_stage1/
  scoring.py
  segment_merge.py
  summarize.py
  detect.py
```

### 파일별 역할

| 파일 | 역할 |
|---|---|
| `deepfakebench_efficientnet_b4.py` | EfficientNet-B4 추론 wrapper |
| `model_loader.py` | 모델 config/weight 로딩 |
| `transforms.py` | 얼굴 crop 전처리 |
| `scoring.py` | face score -> frame score 변환 |
| `segment_merge.py` | 고점수 프레임 구간 병합 |
| `summarize.py` | video_score, result.json 생성 |
| `detect.py` | B 전체 파이프라인 실행 진입점 |

---

## 8. B 구현 함수명 추천

```python
# services/ai/pipelines/video_stage1/detect.py

def run_video_stage1_detection(preprocessing_json_path: str) -> dict:
    """
    preprocessing.json을 읽고 detection.json과 result.json을 생성한다.
    """
```

```python
# services/ai/inference/video_stage1/deepfakebench_efficientnet_b4.py

def predict_face_crops(face_items: list[dict], batch_size: int = 16) -> list[dict]:
    """
    crop_path 목록을 받아 face_id별 raw_fake_score를 반환한다.
    """
```

```python
# services/ai/pipelines/video_stage1/scoring.py

def aggregate_face_scores_to_frame_scores(face_scores: list[dict]) -> list[dict]:
    """
    얼굴 단위 점수를 프레임 단위 점수로 집계한다.
    """
```

```python
# services/ai/pipelines/video_stage1/segment_merge.py

def merge_suspicious_frames(frame_scores: list[dict], threshold: float = 0.6, max_gap_sec: float = 1.0) -> list[dict]:
    """
    threshold 이상 프레임을 인접 구간으로 병합한다.
    """
```

```python
# services/ai/pipelines/video_stage1/summarize.py

def build_video_score(frame_scores: list[dict], segment_scores: list[dict]) -> dict:
    """
    전체 영상 점수를 계산한다.
    """
```

```python
# services/ai/pipelines/video_stage1/summarize.py

def build_final_result(preprocessing: dict, detection: dict) -> dict:
    """
    preprocessing.json과 detection.json을 합쳐 result.json 구조를 만든다.
    """
```

---

## 9. DeepfakeBench 연결 원칙

`services/ai/deepfakebench/`는 외부 기반 엔진 폴더로 유지합니다.

B 담당자는 가능하면 DeepfakeBench 원본 코드를 직접 수정하지 말고, 아래 wrapper에서 호출합니다.

```txt
services/ai/inference/video_stage1/deepfakebench_efficientnet_b4.py
```

### 먼저 확인할 것

```txt
[ ] EfficientNet-B4 config 위치
[ ] detector class 위치
[ ] checkpoint/weight 로딩 방식
[ ] 입력 이미지 크기
[ ] normalize 방식
[ ] output score 의미
[ ] fake score가 높은 쪽인지 낮은 쪽인지
```

### 주의점

`raw_fake_score`는 반드시 “높을수록 fake 가능성이 높음”으로 통일해야 합니다.

만약 모델 출력이 real probability라면 다음처럼 변환해야 합니다.

```python
raw_fake_score = 1.0 - real_probability
```

---

## 10. 얼굴 점수 -> 프레임 점수 규칙

한 프레임에 얼굴이 여러 개 있을 수 있으므로 아래 규칙을 사용합니다.

```txt
frame max_fake_score = 해당 프레임 내 face raw_fake_score 최댓값
frame avg_fake_score = 해당 프레임 내 face raw_fake_score 평균값
```

1단계에서는 최종 판단에 `max_fake_score`를 우선 사용합니다.

이유:

```txt
다중 인물 영상에서 한 명만 조작되어도 해당 프레임은 의심 구간으로 봐야 하기 때문
```

---

## 11. 구간 병합 규칙

기본값:

```json
{
  "score_threshold": 0.6,
  "max_gap_sec": 1.0,
  "min_segment_duration_sec": 0.5,
  "top_k": 5
}
```

처리 방식:

```txt
1. frame_scores에서 max_fake_score >= 0.6인 프레임만 고른다.
2. timestamp 차이가 1.0초 이하이면 같은 구간으로 병합한다.
3. 구간 길이가 0.5초 미만이면 isolated spike로 보고 제외할 수 있다.
4. 구간별 max_fake_score, avg_fake_score를 계산한다.
5. max_fake_score 기준으로 정렬한다.
6. 상위 3~5개만 top_segments로 저장한다.
```

---

## 12. 영상 단위 점수 규칙

1단계에서는 단순 평균보다 `topk_mean`을 추천합니다.

```txt
final_fake_score = 상위 의심 프레임 N개의 평균 점수
```

추천 기본값:

```txt
topk_frame_count = 10
```

이유:

```txt
전체 영상 중 일부 구간만 조작된 경우 단순 평균을 쓰면 의심도가 희석될 수 있기 때문
```

---

## 13. B 실행 명령 추천

```bash
python -m services.ai.pipelines.video_stage1.detect \
  --preprocessing-json storage/jobs/job_20260427_0001/metadata/preprocessing.json
```

성공 시 생성되어야 하는 파일:

```txt
storage/jobs/job_20260427_0001/output/detection.json
storage/jobs/job_20260427_0001/output/result.json
```

---

## 14. A가 완성되기 전 B가 사용할 mock JSON

B는 A 구현을 기다리지 말고 아래 mock 파일로 먼저 개발합니다.

추천 파일 위치:

```txt
tests/fixtures/video_stage1/preprocessing.mock.json
```

```json
{
  "schema_version": "video-stage1-v1",
  "job_id": "mock_job_001",
  "pipeline_stage": 1,
  "status": "success",
  "input": {
    "normalized_video_path": "tests/fixtures/video_stage1/mock_job_001/input/normalized.mp4"
  },
  "video_metadata": {
    "width": 1280,
    "height": 720,
    "fps": 30.0,
    "duration_sec": 10.0,
    "total_frame_count": 300,
    "sample_fps": 3.0,
    "sampled_frame_count": 30
  },
  "quality_metrics": {
    "face_detect_ratio": 0.9,
    "face_visibility_ratio": 0.88,
    "avg_face_size_ratio": 0.16,
    "min_face_size_ratio": 0.1,
    "max_face_size_ratio": 0.22,
    "blur_score": 0.18,
    "motion_blur_ratio": 0.05,
    "dark_frame_ratio": 0.02,
    "compression_artifact_score": 0.14
  },
  "face_summary": {
    "human_face_detected": true,
    "face_detect_failed_frame_count": 3,
    "max_face_count_per_frame": 1,
    "avg_face_count_per_frame": 0.9,
    "multi_face_flag": false,
    "face_track_stability": 0.84
  },
  "frames": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "frame_path": "tests/fixtures/video_stage1/mock_job_001/frames/frame_000000.jpg",
      "face_count": 1,
      "faces": [
        {
          "face_id": "face_000000_00",
          "face_index": 0,
          "detected": true,
          "bbox": [300, 120, 520, 380],
          "bbox_area_ratio": 0.062,
          "detection_confidence": 0.97,
          "crop_path": "tests/fixtures/video_stage1/mock_job_001/faces/frame_000000_face_00.jpg"
        }
      ]
    }
  ],
  "errors": []
}
```

---

## 15. 모델이 아직 연결되지 않았을 때 임시 mock score 규칙

DeepfakeBench 연결이 늦어질 수 있으므로, B는 먼저 mock scorer로 전체 JSON 흐름을 완성해도 됩니다.

```python
def predict_face_crops_mock(face_items: list[dict]) -> list[dict]:
    results = []
    for item in face_items:
        results.append({
            "face_id": item["face_id"],
            "frame_index": item["frame_index"],
            "timestamp_sec": item["timestamp_sec"],
            "crop_path": item["crop_path"],
            "raw_fake_score": 0.5,
            "inference_success": True,
        })
    return results
```

이 방식으로 먼저 아래 기능을 검증합니다.

```txt
[ ] preprocessing.json 로드
[ ] face_items 추출
[ ] face_scores 생성
[ ] frame_scores 생성
[ ] segment_scores 생성
[ ] detection.json 저장
[ ] result.json 저장
```

그 다음 실제 EfficientNet-B4 wrapper를 연결합니다.

---

## 16. 백엔드 연결 방식

A가 `POST /media/video-stage1/preprocess`를 만든다면, B는 아래 엔드포인트를 추가하는 것이 자연스럽습니다.

추천 추가 엔드포인트:

```txt
POST /media/video-stage1/detect
```

요청 예시:

```json
{
  "preprocessing_json": "storage/jobs/job_20260427_0001/metadata/preprocessing.json"
}
```

응답 예시:

```json
{
  "job_id": "job_20260427_0001",
  "status": "success",
  "detection_json": "storage/jobs/job_20260427_0001/output/detection.json",
  "result_json": "storage/jobs/job_20260427_0001/output/result.json"
}
```

---

## 17. B 완료 조건

아래가 모두 되면 B 작업은 완료입니다.

```txt
[ ] preprocessing.json 로드 가능
[ ] frames[].faces[].crop_path 추출 가능
[ ] face crop batch 추론 가능
[ ] raw_fake_score 생성 가능
[ ] face_scores 생성 가능
[ ] frame_scores 생성 가능
[ ] threshold 기반 의심 프레임 추출 가능
[ ] 의심 프레임을 segment_scores로 병합 가능
[ ] top_segments 3~5개 추출 가능
[ ] video_score 계산 가능
[ ] detection.json 저장 가능
[ ] result.json 저장 가능
[ ] A가 만든 실제 preprocessing.json으로 실행 가능
```

---

## 18. B가 절대 하지 말아야 할 것

```txt
[ ] normalized.mp4 생성 금지
[ ] 프레임 샘플링 구현 금지
[ ] 얼굴 검출 구현 금지
[ ] 얼굴 crop 생성 금지
[ ] quality_metrics 직접 계산 금지
[ ] preprocessing.json 구조 임의 변경 금지
[ ] DeepfakeBench 원본 폴더 대규모 수정 금지
[ ] ViT, MINTIME, attention map 구현 금지
[ ] LLM 요약 구현 금지
```

---

## 19. A와 맞춰야 하는 최종 체크

B는 A에게 아래 3개만 받으면 됩니다.

```txt
1. job_id
2. storage/jobs/{job_id}/metadata/preprocessing.json
3. storage/jobs/{job_id}/faces/*.jpg
```

B는 A의 내부 구현 방식에 의존하지 말고, 오직 `preprocessing.json` 스키마에만 의존해야 합니다.

---

## 20. 커밋 메시지 추천

```txt
feat(ai): add video stage1 deepfake detection pipeline
```

또는 작업을 더 작게 나누면:

```txt
feat(ai): add efficientnet b4 inference wrapper
feat(ai): add face score aggregation
feat(ai): add suspicious segment merge logic
feat(ai): add stage1 detection result schema
```
