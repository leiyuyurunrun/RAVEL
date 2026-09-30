$ErrorActionPreference = "Stop"

$Runner = (Resolve-Path (Join-Path $PSScriptRoot "run_v6_eval_latest.ps1")).Path
if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Shared v6 latest-eval runner not found: $Runner"
}

& $Runner @args -EvaluationSplit hard
exit $LASTEXITCODE
