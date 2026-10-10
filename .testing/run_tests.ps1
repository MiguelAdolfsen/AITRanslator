param(
    [switch]$WithSmoke,
    # Defaults to a "mangafolder" folder next to the repo.
    [string]$SmokeImage
)

# Native tools (unittest, compileall) write normal output to stderr, which
# "Stop" turns into a terminating error. Failures are caught via $LASTEXITCODE.
$ErrorActionPreference = "Continue"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $repo ".venv\Scripts\python.exe"
$pythonPrefix = @()
if (-not (Test-Path $python)) {
    $python = "py"
    $pythonPrefix = @("-3")
}

if ($SmokeImage) {
    $SmokeImage = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine((Get-Location).Path, $SmokeImage))
}

Push-Location $repo
try {
    $env:PYTHONDONTWRITEBYTECODE = "1"

    & $python @pythonPrefix -m compileall manga_local_translator
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $python @pythonPrefix -m manga_local_translator --help | Out-Null
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $python @pythonPrefix -m manga_local_translator.quality_eval --help | Out-Null
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $python @pythonPrefix -m manga_local_translator.review_report --help | Out-Null
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $python @pythonPrefix -m unittest discover -s .testing\tests -p "test_*.py"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if ($WithSmoke) {
        $outputDir = Join-Path $repo ".testing\output"
        New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
        $sample = $SmokeImage
        if (-not $sample) {
            $sample = Join-Path $repo "..\mangafolder\1.png"
        }
        if (-not (Test-Path $sample -PathType Leaf)) {
            $expected = [System.IO.Path]::GetFullPath($sample)
            Write-Host "Smoke test failed: sample page not found at $expected" -ForegroundColor Red
            Write-Host "The smoke test needs a local manga page that is not part of the repo." -ForegroundColor Red
            Write-Host "Pass one with -SmokeImage <path>, put one at ..\mangafolder\1.png next to the repo, or run without -WithSmoke." -ForegroundColor Red
            exit 1
        }
        $sample = (Resolve-Path $sample).Path
        $output = Join-Path $outputDir "smoke-none.png"
        & $python @pythonPrefix -m manga_local_translator $sample $output --detector ctd --ocr-engine manga-ocr --translator none --debug --overwrite
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
finally {
    Pop-Location
}
