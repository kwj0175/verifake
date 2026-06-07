# AI 런타임 환경 세팅

이 문서는 VeriFake에서 audio/video AI를 로컬에서 실행하기 위한 환경 세팅 기준입니다. 핵심은 audio AntiDeepfake 환경과 video/evaluation 환경을 분리하는 것입니다.

## 권장 환경 구조

```text
.venv-antideepfake  # audio AntiDeepfake 전용
.venv-eval          # dataset evaluation, video stage1, backend AI 보조 실행
```

두 환경을 분리하는 이유:

- AntiDeepfake audio는 오래된 `fairseq` pinned commit과 `numpy==1.21.2` 조합이 필요합니다.
- video/backend 쪽은 `tensorflow`, `retina-face`, `opencv`, `torchvision` 등 별도 의존성이 필요합니다.
- 한 venv에 모두 설치하면 TensorFlow/Torch/Numpy 조합이 서로 충돌하거나, audio 쪽 `fairseq` import가 깨질 수 있습니다.

## Audio AntiDeepfake 환경

현재 검증된 환경:

```text
Python 3.9.x
pip==24.0
numpy==1.21.2
fairseq==1.0.0a0+862efab
```

중요: `numpy`는 아무 버전이나 쓰면 안 됩니다. `fairseq` 내부 코드가 `np.float`를 참조하기 때문에 `numpy>=1.24`에서는 다음 오류가 날 수 있습니다.

```text
AttributeError: module 'numpy' has no attribute 'float'
```

### 설치

PowerShell 기준:

```powershell
py -3.9 -m venv .venv-antideepfake

.\.venv-antideepfake\Scripts\python.exe -m pip install pip==24.0
.\.venv-antideepfake\Scripts\python.exe -m pip install -r services\ai\antideepfake\requirements.txt
.\.venv-antideepfake\Scripts\python.exe -m pip install -r services\ai\requirements.txt
```

`services\ai\antideepfake\requirements.txt`에는 반드시 아래 조합이 유지되어야 합니다.

```text
numpy==1.21.2
git+https://github.com/pytorch/fairseq.git@862efab86f649c04ea31545ce28d13c59560113d
```

주의:

- `numpy`를 최신 버전으로 업그레이드하지 마세요.
- `fairseq`를 최신 PyPI 버전으로 바꾸지 마세요.
- `services\backend\requirements-ai-stage1.txt`를 `.venv-antideepfake`에 설치하지 마세요. TensorFlow/Numpy 조합 때문에 audio 런타임이 깨질 수 있습니다.
- `pip`이 24.0보다 최신이어도 이미 설치된 패키지가 바로 깨지는 것은 아니지만, 재설치 시 `fairseq`/`omegaconf` metadata 문제를 만들 수 있으므로 audio venv는 `pip==24.0`으로 맞춥니다.

### Audio 환경 검증

```powershell
.\.venv-antideepfake\Scripts\python.exe -c "import sys, numpy; print(sys.version.split()[0]); print(numpy.__version__)"
```

기대값:

```text
3.9.x
1.21.2
```

`fairseq` import까지 확인:

```powershell
.\.venv-antideepfake\Scripts\python.exe -c "import numpy; print(numpy.__version__); import fairseq; print(fairseq.__version__)"
```

기대값:

```text
1.21.2
1.0.0a0+862efab
```

GPU 확인:

```powershell
.\.venv-antideepfake\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## Video / Evaluation 환경

현재 검증된 환경:

```text
Python 3.11.x
.venv-eval
```

설치:

```powershell
py -3.11 -m venv .venv-eval

.\.venv-eval\Scripts\python.exe -m pip install --upgrade pip
.\.venv-eval\Scripts\python.exe -m pip install -r services\backend\requirements-ai-stage1.txt
.\.venv-eval\Scripts\python.exe -m pip install static-ffmpeg
```

video/backend 쪽 주요 패키지:

```text
opencv-python-headless
retina-face
tensorflow
tf-keras
torch
torchvision
torchaudio
efficientnet-pytorch
Pillow
```

검증:

```powershell
.\.venv-eval\Scripts\python.exe -c "import cv2, torch, torchvision, tensorflow, retinaface; print('video env ok')"
```

## CUDA Torch 설정

Torch가 CPU wheel로 설치된 경우 GPU를 못 씁니다. CUDA wheel로 다시 설치해야 합니다.

예시:

```powershell
.\scripts\setup_cuda_torch.ps1 -PythonPath .\.venv-antideepfake\Scripts\python.exe
.\scripts\setup_cuda_torch.ps1 -PythonPath .\.venv-eval\Scripts\python.exe
```

주의: 이 스크립트는 내부에서 `pip`을 upgrade합니다. `.venv-antideepfake`에서 이 스크립트를 실행한 뒤 AntiDeepfake requirements를 다시 설치해야 한다면, 먼저 `pip==24.0`으로 다시 낮춘 뒤 설치하세요.

```powershell
.\.venv-antideepfake\Scripts\python.exe -m pip install pip==24.0
.\.venv-antideepfake\Scripts\python.exe -m pip install -r services\ai\antideepfake\requirements.txt
```

## 필수 모델 파일

Audio AntiDeepfake checkpoint:

```text
services/ai/checkpoints/audio/antideepfake/mms_300m.ckpt
```

Audio hparams:

```text
services/ai/antideepfake/hparams/mms_300m_audio_pipeline.yaml
```

Video EfficientNet-B4 checkpoint:

```text
services/ai/checkpoints/video/effnb4_best.pth
```

모델 가중치 파일은 git에 커밋하지 않고 release asset 또는 별도 저장소에서 받아 배치합니다.

## 환경변수

Backend audio 분석:

```powershell
$env:VERIFAKE_AI_PYTHON = "C:\project\verifake\.venv-antideepfake\Scripts\python.exe"
$env:VERIFAKE_AI_DEVICE = "cuda:0"
$env:VERIFAKE_CUDA_DEVICE = "0"
```

Dataset evaluation audio shard:

```powershell
$env:VERIFAKE_AUDIO_PYTHON = "C:\project\verifake\.venv-antideepfake\Scripts\python.exe"
$env:VERIFAKE_AUDIO_DEVICE = "cuda:0"
$env:VERIFAKE_CUDA_DEVICE = "0"
```

Backend video 분석:

```powershell
$env:VERIFAKE_VIDEO_AI_PYTHON = "C:\project\verifake\.venv-eval\Scripts\python.exe"
$env:VERIFAKE_CUDA_DEVICE = "0"
```

ffmpeg가 PATH에 없으면 `static-ffmpeg` binary 경로를 PATH 앞에 추가합니다.

```powershell
$env:PATH = "C:\project\verifake\.venv-eval\Lib\site-packages\static_ffmpeg\bin\win32;$env:PATH"
```

## Audio-only dataset rerun

기존 video 결과를 덮어쓰지 않고 audio만 다시 돌릴 때는 audio-only config를 사용합니다.

```ini
[inference]
video_enabled = false
audio_enabled = true
```

실행 예시:

```powershell
.\scripts\run_audio_rerun_shards.ps1 `
  -RunDir "C:\Users\alsdl\Downloads\FakeAVCeleb_v1.2_eval_outputs\run_20260508_143312_0147a137_resume" `
  -ResultsRoot "C:\Users\alsdl\Downloads\FakeAVCeleb_v1.2_eval_outputs" `
  -Python "C:\project\verifake\.venv-antideepfake\Scripts\python.exe" `
  -FfmpegBin "C:\project\verifake\.venv-eval\Lib\site-packages\static_ffmpeg\bin\win32" `
  -LogDir "C:\tmp\verifake_audio_rerun" `
  -NumShards 4 `
  -Parallelism 4 `
  -Device "cuda:0"
```

## 정상 동작 확인 기준

Audio 결과가 정상인지 볼 때는 단순히 `audio_fake_prob_like`가 존재하는지만 보면 안 됩니다. 아래 조건을 같이 봅니다.

정상 추론:

```text
scored_window_count > 0
failed_window_count == 0 또는 낮은 값
audio_fake_prob_like가 샘플마다 실제 점수로 기록됨
audio_model_error 없음
```

비정상 추론:

```text
scored_window_count == 0
failed_model_error 발생
audio_model_error 존재
audio_fake_prob_like가 missing/null
```

과거 문제처럼 실패한 audio를 `0.0` 점수로 저장하면 안 됩니다. 점수화된 window가 없으면 audio score는 missing/null로 남겨야 하고, evaluation metric 분모에서도 제외되어야 합니다.

## 빠른 체크리스트

- `.venv-antideepfake`는 Python 3.9.x를 사용한다.
- `.venv-antideepfake`는 `pip==24.0`, `numpy==1.21.2`를 사용한다.
- `.venv-antideepfake`에서 `import fairseq`가 성공한다.
- `.venv-eval`과 `.venv-antideepfake`를 섞지 않는다.
- audio checkpoint와 hparams가 기본 경로에 존재한다.
- video checkpoint가 기본 경로에 존재한다.
- GPU 사용 시 Torch CUDA wheel이 설치되어 있고 `torch.cuda.is_available()`가 `True`다.
- ffmpeg가 PATH에 있거나 `static-ffmpeg` 경로를 PATH에 추가했다.
