$ErrorActionPreference = "Stop"

$root      = (Get-Location).Path
$venvAMD   = Join-Path $root "venvAMD"
$venvNVIDIA= Join-Path $root "venvNVIDIA"

function New-Venv($path) {
    if (-not (Test-Path $path)) {
        Write-Host "Creating venv: $path"
        python -m venv $path
    } else {
        Write-Host "Venv already exists: $path"
    }
}

# --- AMD venv
New-Venv $venvAMD
& "$venvAMD\Scripts\python.exe" -m pip install --upgrade pip
& "$venvAMD\Scripts\pip.exe" install torch-directml
& "$venvAMD\Scripts\pip.exe" install torch torchvision torchaudio

# --- NVIDIA venv
New-Venv $venvNVIDIA
& "$venvNVIDIA\Scripts\python.exe" -m pip install --upgrade pip
# Adjust cuXXX to match your CUDA version (e.g. cu121 for CUDA 12.1)
& "$venvNVIDIA\Scripts\pip.exe" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "✅ AMD and NVIDIA venvs ready."