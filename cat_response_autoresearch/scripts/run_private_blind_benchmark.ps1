param(
    [string]$RunId = "",
    [string]$Profile = "production",
    [int]$Repeats = 4,
    [switch]$NoUpdateBest,
    [string]$OutputRoot = "",
    [string]$Results = "",
    [string]$FakeOutputs = ""
)

$ErrorActionPreference = "Stop"

function Resolve-RequiredPath([string]$PathValue, [string]$Name) {
    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        throw "$Name is required."
    }
    return (Resolve-Path -LiteralPath $PathValue).Path
}

function Test-IsInsidePath([string]$Candidate, [string]$Parent) {
    $candidateFull = [System.IO.Path]::GetFullPath($Candidate).TrimEnd('\', '/')
    $parentFull = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
    return $candidateFull.Equals($parentFull, [System.StringComparison]::OrdinalIgnoreCase) -or
        $candidateFull.StartsWith($parentFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $scriptDir "..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "py"
}

$benchmark = Resolve-RequiredPath $env:CAT_PRIVATE_BENCHMARK "CAT_PRIVATE_BENCHMARK"
if (Test-IsInsidePath $benchmark $repoRoot) {
    throw "CAT_PRIVATE_BENCHMARK must point outside the repo workspace: $repoRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $benchmark "cases.jsonl"))) {
    throw "Private benchmark is missing cases.jsonl: $benchmark"
}
if (-not (Test-Path -LiteralPath (Join-Path $benchmark "references.jsonl"))) {
    throw "Private benchmark is missing references.jsonl: $benchmark"
}

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "private_blind_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repoRoot "cat_response_autoresearch\runs"
}
if ([string]::IsNullOrWhiteSpace($Results)) {
    $Results = Join-Path $repoRoot "cat_response_autoresearch\results\results.tsv"
}

$output = Join-Path $OutputRoot $RunId

$privateOutput = $env:CAT_PRIVATE_OUTPUT
if (-not [string]::IsNullOrWhiteSpace($privateOutput)) {
    $privateParent = Split-Path -Parent $privateOutput
    if (-not [string]::IsNullOrWhiteSpace($privateParent)) {
        New-Item -ItemType Directory -Force -Path $privateParent | Out-Null
    }
    $privateFull = [System.IO.Path]::GetFullPath($privateOutput)
    if (Test-IsInsidePath $privateFull $repoRoot) {
        throw "CAT_PRIVATE_OUTPUT must point outside the repo workspace: $repoRoot"
    }
}

& $python (Join-Path $repoRoot "cat_response_autoresearch\scripts\validate_fixtures.py") --benchmark $benchmark
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$argsList = @(
    (Join-Path $repoRoot "cat_response_autoresearch\scripts\eval_cat_responses.py"),
    "--benchmark", $benchmark,
    "--output", $output,
    "--results", $Results,
    "--run-id", $RunId,
    "--profile", $Profile,
    "--repeats", [string]$Repeats,
    "--blind-report"
)
if ($NoUpdateBest) {
    $argsList += "--no-update-best"
}
if (-not [string]::IsNullOrWhiteSpace($privateOutput)) {
    $argsList += @("--private-output", $privateOutput)
}
if (-not [string]::IsNullOrWhiteSpace($FakeOutputs)) {
    $fakeFull = Resolve-RequiredPath $FakeOutputs "FakeOutputs"
    $argsList += @("--fake-outputs", $fakeFull)
}

& $python @argsList
exit $LASTEXITCODE
