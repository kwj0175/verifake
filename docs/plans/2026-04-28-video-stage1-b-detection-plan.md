# Video Stage1 B Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the Stage 1 B mock-first video detection pipeline that reads `preprocessing.json`, generates `detection.json` and `result.json`, and does not touch Task A.

**Architecture:** Keep inference concerns behind a service wrapper and keep pipeline modules pure and testable. Use schema validation and pytest-first development so file I/O and aggregation rules are grounded in the written spec rather than guesswork.

**Tech Stack:** Python, Pydantic, pytest, pathlib, JSON file I/O.

---

### Task 1: Add schema/config foundation

**Files:**
- Create: `tests/services/ai/pipelines/video_stage1/test_schemas.py`
- Create: `services/ai/pipelines/__init__.py`
- Create: `services/ai/pipelines/video_stage1/__init__.py`
- Create: `services/ai/pipelines/video_stage1/config.stage1.json`
- Create: `services/ai/pipelines/video_stage1/schemas.py`

**Step 1: Write the failing test**

Add tests that verify:
- config contains B-spec keys and values
- score bounds are enforced
- minimal detection/result objects validate

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/services/ai/pipelines/video_stage1/test_schemas.py -q`

**Step 3: Write minimal implementation**

Add the package files, config JSON, and Pydantic models required by the tests.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/services/ai/pipelines/video_stage1/test_schemas.py -q`

---

### Task 2: Add mock inference wrapper

**Files:**
- Create: `tests/services/ai/inference/video_stage1/test_deepfakebench_efficientnet_b4.py`
- Create: `services/ai/inference/__init__.py`
- Create: `services/ai/inference/video_stage1/__init__.py`
- Create: `services/ai/inference/video_stage1/deepfakebench_efficientnet_b4.py`

**Step 1: Write the failing test**

Add tests that verify:
- empty inputs return empty output
- returned items preserve face metadata
- mock scorer uses `raw_fake_score=0.5` and `inference_success=True`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/services/ai/inference/video_stage1/test_deepfakebench_efficientnet_b4.py -q`

**Step 3: Write minimal implementation**

Implement the mock scorer wrapper only.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/services/ai/inference/video_stage1/test_deepfakebench_efficientnet_b4.py -q`

---

### Task 3: Add face-to-frame aggregation

**Files:**
- Create: `tests/services/ai/pipelines/video_stage1/test_scoring.py`
- Create: `services/ai/pipelines/video_stage1/scoring.py`

**Step 1: Write the failing test**

Add tests for:
- per-frame grouping
- `face_count`
- `max_fake_score`
- `avg_fake_score`
- stable ordering by `frame_index`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/services/ai/pipelines/video_stage1/test_scoring.py -q`

**Step 3: Write minimal implementation**

Implement `aggregate_face_scores_to_frame_scores`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/services/ai/pipelines/video_stage1/test_scoring.py -q`

---

### Task 4: Add suspicious segment merge logic

**Files:**
- Create: `tests/services/ai/pipelines/video_stage1/test_segment_merge.py`
- Create: `services/ai/pipelines/video_stage1/segment_merge.py`

**Step 1: Write the failing test**

Add tests for:
- threshold filtering on `max_fake_score`
- merge when timestamp gap is within `max_gap_sec`
- split when gap exceeds threshold
- drop too-short segments
- produce ranked `top_segments`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/services/ai/pipelines/video_stage1/test_segment_merge.py -q`

**Step 3: Write minimal implementation**

Implement pure segment merge helpers.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/services/ai/pipelines/video_stage1/test_segment_merge.py -q`

---

### Task 5: Add video summary builders

**Files:**
- Create: `tests/services/ai/pipelines/video_stage1/test_summarize.py`
- Create: `services/ai/pipelines/video_stage1/summarize.py`

**Step 1: Write the failing test**

Add tests for:
- `max_fake_score`, `avg_fake_score`, `topk_mean_fake_score`, `final_fake_score`
- projection from preprocessing + detection into final result
- empty-segment / empty-top-k behavior in mock-first mode

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/services/ai/pipelines/video_stage1/test_summarize.py -q`

**Step 3: Write minimal implementation**

Implement `build_video_score` and `build_final_result`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/services/ai/pipelines/video_stage1/test_summarize.py -q`

---

### Task 6: Add end-to-end detect entrypoint

**Files:**
- Create: `tests/fixtures/video_stage1/preprocessing.mock.json`
- Create: `tests/services/ai/pipelines/video_stage1/test_detect.py`
- Create: `services/ai/pipelines/video_stage1/detect.py`

**Step 1: Write the failing test**

Add an end-to-end test that:
- creates placeholder face/frame files under `tmp_path`
- loads the preprocessing fixture
- runs the detect entrypoint
- verifies `detection.json`, `result.json`, and `logs/detection.log`
- verifies default mock-first output has no suspicious segments

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/services/ai/pipelines/video_stage1/test_detect.py -q`

**Step 3: Write minimal implementation**

Implement orchestration only:
- read preprocessing JSON
- extract face items
- call mock scorer
- aggregate scores
- merge segments
- build summaries
- write output files

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/services/ai/pipelines/video_stage1/test_detect.py -q`

---

### Task 7: Final verification

**Files:**
- Verify all files above

**Step 1: Run targeted suite**

Run: `PYTHONPATH=. python -m pytest tests/services/ai -q`

**Step 2: Run diagnostics**

Run LSP diagnostics on all changed Python files.

**Step 3: Smoke-run the CLI**

Run: `PYTHONPATH=. python -m services.ai.pipelines.video_stage1.detect --preprocessing-json <fixture-or-job-path>`

**Step 4: Confirm outputs**

Verify the expected JSON/log files are created and contain the expected mock-first shape.
