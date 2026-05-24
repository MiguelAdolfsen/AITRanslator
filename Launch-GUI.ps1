$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"

Set-Location $ProjectRoot

$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TRANSFORMERS_NO_ADVISORY_WARNINGS = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

if (-not (Test-Path $PythonPath)) {
    Write-Host "Creating virtual environment..."
    py -3 -m venv .venv
    Write-Host "Preparing pip..."
    & $PythonPath -m pip install --quiet --upgrade pip
}

Write-Host "Checking dependencies..."
& $PythonPath -m pip install --quiet --disable-pip-version-check -r requirements.txt

Write-Host "Launching GUI..."
& $PythonPath -m manga_local_translator.gui
