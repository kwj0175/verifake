param(
    [string]$TorchVersion = "2.6.0",
    [string]$CudaTag = "cu126",
    [string]$TorchvisionVersion = "0.21.0",
    [string]$TorchaudioVersion = "2.6.0",
    [string]$PythonPath = ""
)

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $candidate = Join-Path (Get-Location) ".venv-antideepfake\\Scripts\\python.exe"
    if (Test-Path $candidate) {
        $PythonPath = $candidate
    }
}

if (-not (Test-Path $PythonPath)) {
    throw "Python executable not found: $PythonPath"
}

Write-Host "[1/4] Uninstall existing torch binaries from: $PythonPath"
& $PythonPath -m pip uninstall -y torch torchvision torchaudio

Write-Host "[2/4] Install CUDA wheels (index=$CudaTag, version=$TorchVersion)"
Write-Host "[2/4] Packages: torch==$TorchVersion+$CudaTag, torchvision==$TorchvisionVersion+$CudaTag, torchaudio==$TorchaudioVersion+$CudaTag"
& $PythonPath -m pip install --upgrade pip
& $PythonPath -m pip install `
    --index-url "https://download.pytorch.org/whl/$CudaTag" `
    --extra-index-url "https://pypi.org/simple" `
    "torch==$TorchVersion+$CudaTag" `
    "torchvision==$TorchvisionVersion+$CudaTag" `
    "torchaudio==$TorchaudioVersion+$CudaTag"

Write-Host "[3/4] Verify import"
& $PythonPath -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

Write-Host "[4/4] Runtime check"
& $PythonPath scripts/runtime_check.py

Write-Host "Done. If you use backend subprocess, set VERIFAKE_VIDEO_AI_PYTHON=$PythonPath"
