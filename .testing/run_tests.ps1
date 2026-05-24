param(
    [switch]$WithSmoke
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $repo ".venv\Scripts\python.exe"
$pythonPrefix = @()
if (-not (Test-Path $python)) {
    $python = "py"
    $pythonPrefix = @("-3")
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
        $sample = Resolve-Path (Join-Path $repo "..\mangafolder\1.png")
        $output = Join-Path $outputDir "smoke-none.png"
        & $python @pythonPrefix -m manga_local_translator $sample $output --detector ctd --ocr-engine manga-ocr --translator none --debug --overwrite
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
finally {
    Pop-Location
}
