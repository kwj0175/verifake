param(
  [Parameter(Mandatory = $true)]
  [string]$RunDir,

  [Parameter(Mandatory = $true)]
  [string]$ResultsRoot,

  [Parameter(Mandatory = $true)]
  [string]$Python,

  [Parameter(Mandatory = $true)]
  [string]$FfmpegBin,

  [Parameter(Mandatory = $true)]
  [string]$LogDir,

  [int]$NumShards = 4,

  [int]$Parallelism = 3,

  [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedRunDir = (Resolve-Path $RunDir).Path
$resolvedResultsRoot = (Resolve-Path $ResultsRoot).Path
$resolvedPython = (Resolve-Path $Python).Path
$resolvedFfmpegBin = (Resolve-Path $FfmpegBin).Path

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
Set-Location $repoRoot

$env:PATH = "$resolvedFfmpegBin;$env:PATH"
$env:VERIFAKE_AUDIO_PYTHON = $resolvedPython
$env:VERIFAKE_AUDIO_DEVICE = $Device
$env:VERIFAKE_CUDA_DEVICE = "0"
$env:TF_ENABLE_ONEDNN_OPTS = "0"

if ($Parallelism -lt 1) {
  throw "Parallelism must be at least 1"
}

$running = @()

function Wait-OneShard {
  param([array]$Processes)

  while ($true) {
    foreach ($entry in @($Processes)) {
      if ($entry.Process.HasExited) {
        $entry.Process.Refresh()
        if ($entry.Process.ExitCode -ne 0) {
          throw "Audio shard $($entry.ShardIndex) failed with exit code $($entry.Process.ExitCode). See $($entry.StderrPath)"
        }
        "Finished shard $($entry.ShardIndex) of $NumShards at $(Get-Date -Format o)" | Add-Content -Path $entry.StdoutPath
        return @($Processes | Where-Object { $_.ShardIndex -ne $entry.ShardIndex })
      }
    }
    Start-Sleep -Seconds 5
  }
}

for ($shardIndex = 0; $shardIndex -lt $NumShards; $shardIndex++) {
  while ($running.Count -ge $Parallelism) {
    $running = @(Wait-OneShard -Processes $running)
  }

  $stdoutPath = Join-Path $LogDir "shard_${shardIndex}_stdout.log"
  $stderrPath = Join-Path $LogDir "shard_${shardIndex}_stderr.log"
  "Starting shard $shardIndex of $NumShards at $(Get-Date -Format o)" | Add-Content -Path $stdoutPath

  $arguments = @(
    "-m", "services.ai.evaluation.dataset_eval",
    "--config", "fakeavcelebeval_audio_only.ini",
    "--run-dir", $resolvedRunDir,
    "--shard-index", "$shardIndex",
    "--num-shards", "$NumShards",
    "--results-root", $resolvedResultsRoot
  )
  $process = Start-Process -FilePath $resolvedPython `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -PassThru `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath
  $running += [pscustomobject]@{
    ShardIndex = $shardIndex
    Process = $process
    StdoutPath = $stdoutPath
    StderrPath = $stderrPath
  }
}

while ($running.Count -gt 0) {
  $running = @(Wait-OneShard -Processes $running)
}

$aggregateStdout = Join-Path $LogDir "aggregate_stdout.log"
$aggregateStderr = Join-Path $LogDir "aggregate_stderr.log"
& $resolvedPython scripts\aggregate_audio_rerun_results.py `
  --run-dir $resolvedRunDir `
  --results-root $resolvedResultsRoot `
  --num-shards $NumShards `
  1>> $aggregateStdout `
  2>> $aggregateStderr
if ($LASTEXITCODE -ne 0) {
  throw "Aggregate step failed with exit code $LASTEXITCODE. See $aggregateStderr"
}

"Audio rerun completed at $(Get-Date -Format o)" | Add-Content -Path (Join-Path $LogDir "completed.log")
