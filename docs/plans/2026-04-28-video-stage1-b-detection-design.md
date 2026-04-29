# Video Stage1 B Detection Design

**Scope:** Implement Stage 1 B only for `docs/verifake_stage1_B_detection.md`.

## Confirmed boundaries

- Read only `storage/jobs/{job_id}/metadata/preprocessing.json` and referenced `faces/*.jpg`, `frames/*.jpg`
- Produce only `output/detection.json`, `output/result.json`, and `logs/detection.log`
- Do not modify Task A behavior or schema
- Do not add the optional backend endpoint in this phase
- Do not edit `services/ai/deepfakebench/**`
- Do not connect real model weights in this phase; use mock scorer first

## Chosen approach

Use a small service-side wrapper plus a pure pipeline layer.

- `services/ai/inference/video_stage1/`
  - scorer interface and mock EfficientNet-B4 wrapper
- `services/ai/pipelines/video_stage1/`
  - config, schemas, scoring, segment merge, summarize, detect entrypoint
- `tests/fixtures/video_stage1/`
  - mock preprocessing input fixture
- `tests/services/ai/...`
  - pytest coverage for each pure module plus end-to-end detect flow

## Design decisions

### 1. Mock-first inference

The detection wrapper returns face-level items with:

- `face_id`
- `frame_index`
- `timestamp_sec`
- `crop_path`
- `raw_fake_score`
- `inference_success`

In this phase, `raw_fake_score` is fixed to `0.5` as documented by the B spec.

### 2. Model semantics held in wrapper only

Local DeepfakeBench evidence shows EfficientNet-B4 inference uses `softmax(pred)[:, 1]` and class `1` means fake. That mapping is preserved as a future integration rule inside the inference wrapper, not leaked into pipeline logic.

### 3. Pure pipeline modules

- `scoring.py`: face scores to frame scores
- `segment_merge.py`: suspicious frame filtering and segment merge
- `summarize.py`: `video_score` and final `result.json` projection
- `detect.py`: orchestrates file I/O and module calls only

### 4. Schema-first validation

Use Pydantic models, matching the style already used in `services/ai/audio_pipeline/schemas.py`, so JSON contracts are explicit and validated.

### 5. Expected mock-first behavior

Because the mock scorer returns `0.5` and segment threshold is `0.6`:

- `detection.json` should still be produced
- `result.json` should still be produced
- `segment_scores` should be empty by default
- `top_segments` should be empty by default

Segment behavior above threshold will be covered by synthetic unit tests rather than the default end-to-end fixture.

## File layout

```txt
services/ai/inference/
  __init__.py
  video_stage1/
    __init__.py
    deepfakebench_efficientnet_b4.py

services/ai/pipelines/
  __init__.py
  video_stage1/
    __init__.py
    config.stage1.json
    schemas.py
    scoring.py
    segment_merge.py
    summarize.py
    detect.py

tests/
  fixtures/video_stage1/preprocessing.mock.json
  services/ai/inference/video_stage1/test_deepfakebench_efficientnet_b4.py
  services/ai/pipelines/video_stage1/test_schemas.py
  services/ai/pipelines/video_stage1/test_scoring.py
  services/ai/pipelines/video_stage1/test_segment_merge.py
  services/ai/pipelines/video_stage1/test_summarize.py
  services/ai/pipelines/video_stage1/test_detect.py
```

## Non-goals

- Task A preprocessing implementation
- MP4 normalization, frame sampling, face detection, crop generation
- Real EfficientNet-B4 checkpoint loading
- Backend API integration
- ViT/MINTIME/attention/LLM extensions
