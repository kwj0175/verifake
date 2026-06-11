# Release Weights Guide

이 문서는 GitHub Release로 배포하는 가중치 파일을 이 repo에서 어떻게 다루는지 간단히 정리한 문서입니다.

## 원칙

- 가중치 파일(`.ckpt`, `.pt`, `.bin` 등)은 **git에 커밋하지 않습니다**.
- 실제 파일은 **GitHub Release 또는 외부 스토리지에서 다운로드**해서 사용합니다.
- 코드는 각 모델의 **기본 경로만 기대**하고, 파일이 없으면 사용자가 직접 배치해야 합니다.
- 기본 경로를 쓰지 않을 경우 `--checkpoint-path` 같은 인자로 **명시적으로 경로를 넘깁니다**.

## 공통 위치 규칙

이 프로젝트는 체크포인트를 보통 아래 아래에 둡니다.

```text
services/ai/checkpoints/<domain>/<model>/
```

예:

- audio: `services/ai/checkpoints/audio/antideepfake/`
- video: video 모델도 같은 방식으로 `services/ai/checkpoints/video/...` 아래에 두는 패턴을 따릅니다.

## 현재 릴리즈 예시

현재 업로드된 릴리즈:

- tag: `audio-weights-v1`
- URL: `https://github.com/oocij1jl/verifake/releases/tag/audio-weights-v1`

이 릴리즈의 체크포인트는 아래 경로에 두면 됩니다.

```text
services/ai/checkpoints/audio/antideepfake/mms_300m.ckpt
```

함께 쓰는 hparams 경로:

```text
services/ai/antideepfake/hparams/mms_300m_audio_pipeline.yaml
```

## 사용 예시

repo root에서:

```powershell
python -m services.ai.audio_pipeline.audio_stage1 `
  --input path/to/audio.wav `
  --output-dir outputs/audio `
  --checkpoint-path services/ai/checkpoints/audio/antideepfake/mms_300m.ckpt `
  --hparams-path services/ai/antideepfake/hparams/mms_300m_audio_pipeline.yaml `
  --device cpu
```

## 확인 방법

```powershell
python -c "from pathlib import Path; p=Path(r'services/ai/checkpoints/audio/antideepfake/mms_300m.ckpt'); print(p.exists()); print(p.resolve())"
```

`True`가 출력되면 정상입니다.

## 메모

- `.gitignore`에서 `services/ai/checkpoints/**/*.ckpt` 등을 ignore 합니다.
- 따라서 release asset을 받아도 **git tracked 파일이 되지 않는 것이 정상**입니다.
- audio 외 다른 가중치도 같은 원칙으로 관리하면 됩니다.
