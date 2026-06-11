# verifake

## AI/Backend runtime check (Windows)

현재 시스템 GPU가 붙어있어도 `torch+cpu` 환경이면 Stage 1 추론은 CPU로 동작할 수 있습니다. 아래 체크부터 실행하세요.

```powershell
python --version
.\scripts\runtime_check.py
.\scripts\setup_cuda_torch.ps1
```

- `runtime_check.py`는 현재 런타임에서 PyTorch/TensorFlow/CUDA 가용성을 출력합니다.
- `setup_cuda_torch.ps1`는 `.venv-antideepfake\Scripts\python.exe` 기준으로 CUDA 휠을 다시 설치합니다.

## 권장 실행순서

1. Stage 1 API/백엔드 실행 환경의 Python을 `.venv-antideepfake` 또는 해당 스크립트가 가리키는 환경으로 맞춥니다.
2. 필요 시 강제로 AI 서브프로세스 경로를 고정:

```powershell
$env:VERIFAKE_VIDEO_AI_PYTHON = "C:\project\verifake\.venv-antideepfake\Scripts\python.exe"
```

3. 백엔드 실행 후 `/video-stage1/detect` API를 호출하면
   - `services/ai/common/runtime_probe.py`의 로그와 `torch.is_available()` 결과가 장치 선택 의사결정에 반영됩니다.

각 디렉터리에서 파일 추가 했으면 .gitkeep은 지워주세요.
