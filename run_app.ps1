<#
Usage:
  .\run_app.ps1                       # auto-detect GPU/CPU and run app.py
  .\run_app.ps1 -Backend amd          # force backend: cpu | amd | nvidia
  .\run_app.ps1 -Entry serve.py -Args "--port 7860"
  .\run_app.ps1 -EchoDevice           # print which device torch will use
#>

param(
  [string]$Backend = "",                 # cpu | amd | nvidia (empty = auto)
  [string]$Entry   = "app.py",           # your main file
  [string]$Args    = "",                 # extra args to pass to your app
  [switch]$EchoDevice                    # optional health print
)

$ErrorActionPreference = "Stop"
$root        = (Get-Location).Path
$venvCPU     = Join-Path $root "venv"
$venvAMD     = Join-Path $root "venvAMD"
$venvNVIDIA  = Join-Path $root "venvNVIDIA"

function Python-FromVenv([string]$venvPath) {
  $py = Join-Path $venvPath "Scripts\python.exe"
  if (-not (Test-Path $py)) { throw "Python not found in $venvPath" }
  return $py
}

function Has-Nvidia {
  try { & nvidia-smi *> $null; return $LASTEXITCODE -eq 0 } catch { return $false }
}
function Gpu-Vendor {
  try {
    $names = (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join " "
    $n = $names.ToLower()
    if ($n -match "nvidia") { return "nvidia" }
    if ($n -match "amd|advanced micro devices|ati") { return "amd" }
    return ""
  } catch { return "" }
}

# --- Decide backend (prefer NVIDIA if both present) ---
if (-not $Backend) {
  $vendor = Gpu-Vendor
  if ($vendor -eq "nvidia" -and (Has-Nvidia)) { $Backend = "nvidia" }
  elseif ($vendor -eq "amd")                  { $Backend = "amd" }
  else                                        { $Backend = "cpu" }
}

# --- Pick python from the right venv & set env flags per backend ---
switch ($Backend.ToLower()) {
  "nvidia" {
    $python = Python-FromVenv $venvNVIDIA
    # allow xFormers if installed in venvNVIDIA
    Remove-Item Env:\AUDIOCRAFT_USE_XFORMERS -ErrorAction SilentlyContinue
    Remove-Item Env:\USE_DML -ErrorAction SilentlyContinue
    Write-Host "▶ Backend: NVIDIA (CUDA)  | venv: venvNVIDIA"
  }
  "amd" {
    $python = Python-FromVenv $venvAMD
    $env:USE_DML = "1"                     # hint for your code if needed
    $env:AUDIOCRAFT_USE_XFORMERS = "0"     # disable xformers on DirectML
    Write-Host "▶ Backend: AMD (DirectML) | venv: venvAMD"
  }
  "cpu" {
    $python = Python-FromVenv $venvCPU
    $env:AUDIOCRAFT_USE_XFORMERS = "0"     # disable xformers on CPU
    Remove-Item Env:\USE_DML -ErrorAction SilentlyContinue
    Write-Host "▶ Backend: CPU            | venv: venv"
  }
  default { throw "Unknown backend '$Backend' (use cpu | amd | nvidia)" }
}

# --- Optional device echo ---
if ($EchoDevice) {
  Write-Host "→ Checking torch device…" -ForegroundColor DarkGray
  $pyCode = @"
import os, sys
try:
    import torch
    print("torch", torch.__version__)
    if os.environ.get("USE_DML") == "1":
        try:
            import torch_directml as dml
            print("device: DirectML ->", dml.device())
        except Exception as e:
            print("DirectML not available:", e)
    elif torch.cuda.is_available():
        print("device: CUDA ->", torch.cuda.get_device_name(0))
    else:
        print("device: CPU")
except Exception as e:
    print("torch not importable:", e)
"@
  & $python -c $pyCode
}

# --- Run the app ---
$entryPath = Join-Path $root $Entry
if (-not (Test-Path $entryPath)) { throw "Entry script '$Entry' not found in $root" }

Write-Host "Launching: $Entry $Args"
& $python $Entry $Args