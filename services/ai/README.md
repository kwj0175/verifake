# AI Services

이 폴더는 Verifake의 AI 관련 코드들을 두는 영역입니다.

권장 구조:
- `deepfakebench/` : 외부 기반 벤치마크/학습 엔진
- `inference/` : 서비스용 추론 래퍼
- `common/` : 공통 유틸 및 헬퍼

## AntiDeepfake integration

- Vendored upstream source stays in `services/ai/antideepfake/`.
- Audio-pipeline-facing wrapper lives in `services/ai/audio_pipeline/antideepfake.py`.
- Default repo-specific inference config lives in `services/ai/antideepfake/hparams/mms_300m_audio_pipeline.yaml`.
- Default checkpoint path is `services/ai/checkpoints/audio/antideepfake/mms_300m.ckpt`.
- Default runtime entrypoint for the wrapper is:
  - `python -m services.ai.audio_pipeline.antideepfake <audio-file>`
- Direct vendored entrypoint is:
  - `python services/ai/antideepfake/main.py inference services/ai/antideepfake/hparams/mms_300m_audio_pipeline.yaml --base_path <repo-root> --test_csv <protocol.csv> --ckpt_path services/ai/checkpoints/audio/antideepfake/mms_300m.ckpt`

## 오디오 전처리

- 입력 파일을 내부 분석용 `WAV / mono / 16kHz / pcm_s16le` 형식으로 정규화하는 전처리 CLI는 `services/ai/audio_pipeline/audio_preprocess.py`에 둡니다.
- 실행 예시:
  - `python -m services.ai.audio_pipeline.audio_preprocess --input path/to/input.mp4 --output-dir outputs/audio --json-output outputs/audio/audio_preprocess_result.json`
- 이 단계는 전처리 전용입니다. VAD, sliding window, AntiDeepfake 추론, fake score 계산은 포함하지 않습니다.

## 오디오 VAD / 음성 구간 태깅

- stage 2 VAD CLI는 `services/ai/audio_pipeline/audio_vad.py`에 둡니다.
- 실행 예시:
  - `python -m services.ai.audio_pipeline.audio_vad --preprocess-json outputs/audio/audio_preprocess_result.json --json-output outputs/audio/audio_vad_result.json`
  - `python -m services.ai.audio_pipeline.audio_vad --input-wav outputs/audio/normalized/sample_16k_mono.wav --json-output outputs/audio/audio_vad_result.json`
- 이 단계는 normalized wav 전체 타임라인 위에 speech tag와 speech/silence metadata를 남기는 단계입니다.
- `silero_vad` 가 설치되어 있으면 우선 사용하고, 없으면 energy 기반 fallback으로 동작합니다.
- sliding window 생성, AntiDeepfake 모델 추론, fake/real score 계산은 포함하지 않습니다.

## 오디오 sliding window metadata

- stage 3 windowing CLI는 `services/ai/audio_pipeline/audio_windowing.py`에 둡니다.
- 실행 예시:
  - `python -m services.ai.audio_pipeline.audio_windowing --vad-json outputs/audio/audio_vad_result.json --json-output outputs/audio/audio_windows_result.json`
  - `python -m services.ai.audio_pipeline.audio_windowing --input-wav outputs/audio/normalized/sample_16k_mono.wav --window-sec 4.0 --hop-sec 2.0 --json-output outputs/audio/audio_windows_result.json`
- 이 단계는 normalized wav 전체 타임라인 기준 overlapping window를 생성하고, 각 window의 speech overlap metadata만 계산합니다.
- speech_segments는 hard filter가 아니며, window별 `speech_overlap_sec`, `speech_coverage_ratio`, `has_speech` 계산에만 사용합니다.
- AntiDeepfake 모델 추론, fake/real score 계산, suspicious segment merge는 포함하지 않습니다.

## 오디오 AntiDeepfake window inference

- stage 4 inference CLI는 `services/ai/audio_pipeline/audio_inference.py`에 둡니다.
- 실행 예시:
  - `python -m services.ai.audio_pipeline.audio_inference --windows-json outputs/audio/audio_windows_result.json --checkpoint-path services/ai/checkpoints/audio/antideepfake/mms_300m.ckpt --json-output outputs/audio/audio_inference_result.json --device cpu`
- 이 단계는 stage 3의 `audio_windows_result.json`을 읽고, normalized wav에서 window clip을 추출한 뒤 AntiDeepfake를 window 단위로 호출합니다.
- `has_speech=false` 또는 `speech_coverage_ratio < 0.05` 인 window는 score를 계산하지 않고 skip 상태만 남깁니다.
- `audio_fake_prob_like`는 calibrated probability가 아니라 softmax 기반 probability-like score입니다.
- suspicious segment merge, 최종 audio JSON assembly, calibration, LLM 설명은 포함하지 않습니다.

## 오디오 suspicious segment merge / summary

- stage 5 summary CLI는 `services/ai/audio_pipeline/audio_segments.py`에 둡니다.
- 실행 예시:
  - `python -m services.ai.audio_pipeline.audio_segments --inference-json outputs/audio/audio_inference_result.json --json-output outputs/audio/audio_segments_result.json --suspicious-threshold 0.5 --top-k 5 --max-merge-gap-sec 0.5`
- 이 단계는 stage 4의 `audio_inference_result.json`만 읽고, `inference_status == "scored"` 인 window들 중 `audio_fake_prob_like >= suspicious_threshold` 인 window를 suspicious 후보로 사용합니다.
- suspicious window는 시간순으로 정렬한 뒤, 겹치거나 gap이 `max_merge_gap_sec` 이하이면 하나의 suspicious segment로 병합합니다.
- `audio_fake_prob_like`는 calibrated probability가 아니라 probability-like score이며, threshold 0.5는 초기 heuristic 기준입니다.
- 이 단계의 intermediate segment 필드는 `start`, `end`, `duration`을 사용합니다. 최종 fusion / LLM 전달 단계에서 필요하면 `start_sec`, `end_sec` 같은 최종 schema 필드로 변환합니다.
- 이 단계는 top suspicious audio segments와 전체 audio score summary만 생성하며, final overall/video/audio fusion JSON은 생성하지 않습니다.

### Dependency notes

- Use `services/ai/antideepfake/requirements.txt` for the AntiDeepfake runtime.
- The upstream code depends on a pinned `fairseq` commit; do not replace it with the latest PyPI release.
- In the Python 3.9 AntiDeepfake environment, pin `pip==24.0` before installing `services/ai/antideepfake/requirements.txt`.
- This avoids the `fairseq` / `omegaconf 2.0.x` metadata rejection that newer pip versions trigger.

### TODOs

- TODO: map raw AntiDeepfake logits/probabilities into the richer `AudioAnalysisResult` schema once the rest of the audio evidence pipeline is ready.
- TODO: add additional upstream model configs only when this repo actually needs them.
