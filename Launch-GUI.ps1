# Native tools (pip, venv) write warnings to stderr, which "Stop" turns into a
# terminating error. Failures are caught via $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"

function Stop-Launcher([string]$Message) {
    Write-Host $Message -ForegroundColor Red
    # Keep the window open when started with "Run with PowerShell".
    Read-Host "Press Enter to close"
    exit 1
}

Set-Location $ProjectRoot

$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TRANSFORMERS_NO_ADVISORY_WARNINGS = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

if (-not (Test-Path $PythonPath)) {
    Write-Host "Creating virtual environment..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv .venv
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv .venv
    }
    else {
        Stop-Launcher "Python was not found. Install Python 3.10 or newer from python.org, then run this launcher again."
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $PythonPath)) {
        Stop-Launcher "Could not create the virtual environment in $VenvPath."
    }
    Write-Host "Preparing pip..."
    & $PythonPath -m pip install --quiet --upgrade pip
    if ($LASTEXITCODE -ne 0) { Stop-Launcher "Could not upgrade pip in the virtual environment." }
}

Write-Host "Checking dependencies..."
& $PythonPath -m pip install --quiet --disable-pip-version-check -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Stop-Launcher "Installing dependencies from requirements.txt failed. See the pip output above."
}

Write-Host "Launching GUI..."
& $PythonPath -m manga_local_translator.gui
if ($LASTEXITCODE -ne 0) { Stop-Launcher "The GUI exited with an error (code $LASTEXITCODE)." }
