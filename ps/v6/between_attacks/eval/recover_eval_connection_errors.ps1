param(
    [Parameter(Mandatory = $true)]
    [int]$Episode,

    [string]$LogFile = "",
    [string]$OutputPrefix = "",
    [string]$RuleFile = "",
    [string]$PlanFile = "",
    [string]$PosCsv = "",
    [string]$NegCsv = "",
    [string]$LlmModel = "",
    [string]$LlmProvider = "",
    [string]$PlannerModel = "",
    [string]$JudgeModel = "",
    [string]$EnvModel = "",
    [string]$ZhipuApiKey = "",
    [string]$MiniMaxApiKey = "",
    [string]$XunfeiApiKey = "",
    [ValidateSet("auto", "enabled", "disabled")]
    [string]$GlmEnAnthropic = "auto",
    [bool]$ScanOnly = $false,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Tag = "[V6-EVAL-CONNECTION-RECOVERY]"
$StartedAt = Get-Date
$OldZhipuApiKey = $env:ZHIPU_API_KEY
$OldMiniMaxApiKey = $env:MINIMAX_API_KEY
$OldXunfeiApiKey = $env:XUNFEI_API_KEY

function Restore-Environment {
    if ($null -eq $OldZhipuApiKey) {
        Remove-Item Env:ZHIPU_API_KEY -ErrorAction SilentlyContinue
    }
    else {
        $env:ZHIPU_API_KEY = $OldZhipuApiKey
    }
    if ($null -eq $OldMiniMaxApiKey) {
        Remove-Item Env:MINIMAX_API_KEY -ErrorAction SilentlyContinue
    }
    else {
        $env:MINIMAX_API_KEY = $OldMiniMaxApiKey
    }
    if ($null -eq $OldXunfeiApiKey) {
        Remove-Item Env:XUNFEI_API_KEY -ErrorAction SilentlyContinue
    }
    else {
        $env:XUNFEI_API_KEY = $OldXunfeiApiKey
    }
}

trap {
    $FailedAt = Get-Date
    $Elapsed = $FailedAt - $StartedAt
    Write-Host "$Tag Failed after $($Elapsed.ToString()): $_"
    Restore-Environment
    throw
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath ".\.venv\Scripts\Activate.ps1")) {
    throw "Virtual environment activation script not found: .\.venv\Scripts\Activate.ps1"
}
. .\.venv\Scripts\Activate.ps1

if (-not [string]::IsNullOrWhiteSpace($ZhipuApiKey)) {
    $env:ZHIPU_API_KEY = $ZhipuApiKey
}
if (-not [string]::IsNullOrWhiteSpace($MiniMaxApiKey)) {
    $env:MINIMAX_API_KEY = $MiniMaxApiKey
}
if (-not [string]::IsNullOrWhiteSpace($XunfeiApiKey)) {
    $env:XUNFEI_API_KEY = $XunfeiApiKey
}

$ArgsList = @(
    "experiments\recover_eval_connection_errors.py",
    "--episode", "$Episode",
    "--glm-en-anthropic", $GlmEnAnthropic
)

$OptionalArgs = @(
    @("--log-file", $LogFile),
    @("--output-prefix", $OutputPrefix),
    @("--rule-file", $RuleFile),
    @("--plan-file", $PlanFile),
    @("--pos-csv", $PosCsv),
    @("--neg-csv", $NegCsv),
    @("--llm-model", $LlmModel),
    @("--llm-provider", $LlmProvider),
    @("--planner-model", $PlannerModel),
    @("--judge-model", $JudgeModel),
    @("--env-model", $EnvModel)
)
foreach ($Pair in $OptionalArgs) {
    if (-not [string]::IsNullOrWhiteSpace([string]$Pair[1])) {
        $ArgsList += @([string]$Pair[0], [string]$Pair[1])
    }
}

if ($ScanOnly) {
    $ArgsList += "--scan-only"
}
else {
    $ArgsList += "--rerun-and-apply"
}
if ($DryRun) {
    $ArgsList += "--dry-run"
}

Write-Host "$Tag Episode: $Episode"
Write-Host "$Tag Mode: $(if ($ScanOnly) { 'scan-only' } else { 'rerun-and-apply' })"
Write-Host "$Tag Command: python $($ArgsList -join ' ')"
& python @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "$Tag Python recovery failed with exit code $LASTEXITCODE"
}

$FinishedAt = Get-Date
$Elapsed = $FinishedAt - $StartedAt
Write-Host "$Tag Finished at: $($FinishedAt.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "$Tag Elapsed: $($Elapsed.ToString())"
Restore-Environment
