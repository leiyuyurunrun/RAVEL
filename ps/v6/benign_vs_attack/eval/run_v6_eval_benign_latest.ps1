param(
    [Parameter(Mandatory = $true)]
    [int]$Episode,

    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "access_control",
        "flashloans",
        "insufficient_validation",
        "market_manipulation",
        "price_manipulation",
        "protocol_accounting_exploitation",
        "reentrancy",
        "token_semantic_exploitation"
    )]
    [string]$AttackLabel,

    [string]$LlmModel = "glm-5.2",
    [string]$LlmProvider = "glm-en",
    [Alias("enable_anthropic")]
    [bool]$EnableAnthropic = $true,
    [string]$PlannerModel = "",
    [string]$JudgeModel = "",
    [string]$EnvModel = "",
    [string]$ZhipuApiKey = "",
    [string]$MiniMaxApiKey = "",
    [string]$XunfeiApiKey = "",
    [ValidateSet("stage-default", "disabled", "adaptive")]
    [string]$MiniMaxThinkingMode = "stage-default",
    [ValidateSet("default", "disabled", "adaptive")]
    [string]$JudgeThinking = "disabled",
    [ValidateSet("default", "disabled", "adaptive")]
    [string]$AggregatorThinking = "disabled",

    [string]$RuleFile = "",
    [string]$PlanFile = "",
    [string]$CsvRoot = "data\csv\v7",
    [ValidateSet("standard", "hard")]
    [string]$EvaluationSplit = "standard",
    [string]$OutputPrefix = "eval_benign",
    [string]$EvalPosCsv = "",
    [string]$EvalNegCsv = "",
    [int]$OnlyHead = -1,

    [int]$MaxViewChars = 80000,
    [int]$MaxContextChars = 200000,
    [bool]$ForceRebuildPacket = $true,
    [bool]$EnableSourceTools = $true,
    [bool]$ForceRefreshSource = $false,
    [bool]$UseEnvironment = $false,
    [bool]$SaveLlmTranscripts = $false,
    [bool]$AdaptiveEvidence = $false,
    [string]$AdaptiveEvidenceMode = "off",
    [bool]$AdaptiveEvidenceDebug = $false,
    [ValidateRange(1, 1000000)]
    [int]$AdaptiveEvidenceMaxDirectTraceNodes = 80,
    [ValidateRange(1000, 1000000)]
    [int]$AdaptiveEvidenceMaxDirectTraceChars = 45000,
    [ValidateRange(1, 1000000)]
    [int]$AdaptiveEvidenceMaxMediumTraceNodes = 250,
    [bool]$ParallelJudge = $true,
    [int]$JudgeConcurrency = 2,
    [ValidateRange(0, 32768)]
    [int]$JudgeMaxTokens = 0,
    [ValidateRange(1, 32768)]
    [int]$AggregatorMaxTokens = 8192,
    [bool]$StructuredJudgeLogs = $true,
    [bool]$EnableIvStatefulRuntime = $true,
    [ValidateSet("prompt", "stateful")]
    [string]$AccessControlBindingMode = "stateful",
    [ValidateSet("disabled", "soft", "stateful")]
    [string]$ReentrancyBindingMode = "soft",
    [bool]$DisableStatefulBindings = $false,
    [bool]$EnableDynamicAggregation = $false,
    [ValidateSet("legacy", "unified")]
    [string]$FollowupContextMode = "unified",
    [ValidateSet("plan", "disabled", "expanded")]
    [string]$JudgeFollowupMode = "plan",
    [ValidateRange(0, 20)]
    [int]$JudgeExpandedFollowupViews = 2,
    [bool]$RateLimitSerialFallback = $true,
    [int]$RateLimitRetryAttempts = 3,
    [double]$RateLimitRetryDelaySeconds = 7.0,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Tag = if ($EvaluationSplit -eq "hard") { "[V6-BENIGN-EVAL-HARD]" } else { "[V6-BENIGN-EVAL]" }
$StartedAt = Get-Date

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Set-Location $ProjectRoot

$OldEnv = @{
    GLM_EN_ENABLE_ANTHROPIC = $env:GLM_EN_ENABLE_ANTHROPIC
    ZHIPU_API_KEY = $env:ZHIPU_API_KEY
    MINIMAX_API_KEY = $env:MINIMAX_API_KEY
    XUNFEI_API_KEY = $env:XUNFEI_API_KEY
    MINIMAX_THINKING_OVERRIDE = $env:MINIMAX_THINKING_OVERRIDE
}

function Restore-Environment {
    foreach ($Name in $OldEnv.Keys) {
        $Value = $OldEnv[$Name]
        if ($null -eq $Value) {
            Remove-Item "Env:$Name" -ErrorAction SilentlyContinue
        }
        else {
            Set-Item "Env:$Name" $Value
        }
    }
}

trap {
    $FailedAt = Get-Date
    $Elapsed = $FailedAt - $StartedAt
    $ElapsedText = "{0:00}:{1:00}:{2:00}.{3:000}" -f `
        [int]$Elapsed.TotalHours, $Elapsed.Minutes, $Elapsed.Seconds, $Elapsed.Milliseconds
    Write-Host ("$Tag Failed at: {0}" -f $FailedAt.ToString("yyyy-MM-dd HH:mm:ss"))
    Write-Host "$Tag Elapsed before failure: $ElapsedText"
    Write-Host "$Tag Error: $_"
    Restore-Environment
    throw
}

function Resolve-AttackConfig {
    param([Parameter(Mandatory = $true)][string]$Value)
    switch ($Value) {
        "access_control" {
            return [pscustomobject]@{ Label = "access_control"; PositiveLabel = "Access control"; CsvDir = "Access control"; Prefix = "access_control" }
        }
        "flashloans" {
            return [pscustomobject]@{ Label = "flashloans"; PositiveLabel = "Flashloans"; CsvDir = "Flashloans"; Prefix = "flashloans" }
        }
        "insufficient_validation" {
            return [pscustomobject]@{ Label = "insufficient_validation"; PositiveLabel = "Insufficient validation"; CsvDir = "Insufficient validation"; Prefix = "insufficient_validation" }
        }
        "market_manipulation" {
            return [pscustomobject]@{ Label = "market_manipulation"; PositiveLabel = "Market manipulation"; CsvDir = "Market manipulation"; Prefix = "market_manipulation" }
        }
        "price_manipulation" {
            return [pscustomobject]@{ Label = "price_manipulation"; PositiveLabel = "Price manipulation"; CsvDir = "Price manipulation"; Prefix = "price_manipulation" }
        }
        "protocol_accounting_exploitation" {
            return [pscustomobject]@{ Label = "protocol_accounting_exploitation"; PositiveLabel = "Protocol accounting exploitation"; CsvDir = "Protocol accounting exploitation"; Prefix = "protocol_accounting_exploitation" }
        }
        "reentrancy" {
            return [pscustomobject]@{ Label = "reentrancy"; PositiveLabel = "Reentrancy"; CsvDir = "Reentrancy"; Prefix = "reentrancy" }
        }
        "token_semantic_exploitation" {
            return [pscustomobject]@{ Label = "token_semantic_exploitation"; PositiveLabel = "Token semantic exploitation"; CsvDir = "Token semantic exploitation"; Prefix = "token_semantic_exploitation" }
        }
    }
}

function Assert-ExistingFile {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue,
        [Parameter(Mandatory = $true)][string]$Name
    )
    if ([string]::IsNullOrWhiteSpace($PathValue) -or -not (Test-Path -LiteralPath $PathValue)) {
        throw "$Name not found: $PathValue"
    }
}

function Resolve-InitialArtifactFromReport {
    param(
        [Parameter(Mandatory = $true)][string]$ArtifactRoot,
        [Parameter(Mandatory = $true)][string]$Kind
    )
    $RootLeaf = Split-Path -Leaf $ArtifactRoot
    if ([string]::IsNullOrWhiteSpace($RootLeaf)) {
        return ""
    }
    $FinalReportPath = Join-Path (Join-Path "data\results" $RootLeaf) "final_report.json"
    if (-not (Test-Path -LiteralPath $FinalReportPath)) {
        return ""
    }
    try {
        $Report = Get-Content -LiteralPath $FinalReportPath -Raw | ConvertFrom-Json
    }
    catch {
        Write-Warning "$Tag Could not parse final_report for fallback artifact lookup: $FinalReportPath"
        return ""
    }
    if ($Kind -eq "rule") {
        return [string]$Report.detector_context.initial_rule.rule_path
    }
    return [string]$Report.detector_context.initial_plan.plan_path
}

function Resolve-LatestArtifact {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][ValidateSet("rule", "plan")][string]$Kind
    )
    if ($Kind -eq "rule") {
        $LatestCandidates = @(
            (Join-Path $Root "$Label`__latest.json"),
            (Join-Path $Root "$Label`_evidence_rule__latest.json")
        )
        $LatestFilter = "$Label*__latest.json"
        $RetrospectiveCandidates = @(
            (Join-Path $Root "$Label`__retrospective.json"),
            (Join-Path $Root "$Label`_evidence_rule__retrospective.json")
        )
        $RetrospectiveFilter = "$Label*__retrospective*.json"
        $FallbackCandidates = @(
            (Join-Path $Root "$Label`__v1.json"),
            (Join-Path $Root "$Label`_evidence_rule__v1.json")
        )
        $FallbackFilter = "$Label*__v1.json"
        $VersionedFilter = "$Label*__v*.json"
    }
    else {
        $LatestCandidates = @(
            (Join-Path $Root "$Label`__plan_latest.json"),
            (Join-Path $Root "$Label`_evidence_rule__plan_latest.json")
        )
        $LatestFilter = "$Label*__plan_latest.json"
        $RetrospectiveCandidates = @(
            (Join-Path $Root "$Label`__plan_retrospective.json"),
            (Join-Path $Root "$Label`_evidence_rule__plan_retrospective.json")
        )
        $RetrospectiveFilter = "$Label*__plan_retrospective*.json"
        $FallbackCandidates = @(
            (Join-Path $Root "$Label`__plan_v1.json"),
            (Join-Path $Root "$Label`_evidence_rule__plan_v1.json")
        )
        $FallbackFilter = "$Label*__plan_v1.json"
        $VersionedFilter = "$Label*__plan_v*.json"
    }

    foreach ($Candidate in $LatestCandidates) {
        if (Test-Path -LiteralPath $Candidate) {
            return $Candidate
        }
    }
    $Matches = @(Get-ChildItem -LiteralPath $Root -Filter $LatestFilter -ErrorAction SilentlyContinue)
    if ($Matches.Count -eq 1) {
        return $Matches[0].FullName
    }
    if ($Matches.Count -gt 1) {
        throw "Multiple latest $Kind artifacts matched in ${Root}: $($Matches.FullName -join '; '). Pass -$($Kind.Substring(0, 1).ToUpperInvariant() + $Kind.Substring(1))File explicitly."
    }

    foreach ($Candidate in $RetrospectiveCandidates) {
        if (Test-Path -LiteralPath $Candidate) {
            Write-Warning "$Tag No latest $Kind artifact found in $Root for label $Label; using retrospective artifact: $Candidate"
            return $Candidate
        }
    }
    $RetrospectiveMatches = @(
        Get-ChildItem -LiteralPath $Root -Filter $RetrospectiveFilter -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending
    )
    if ($RetrospectiveMatches.Count -ge 1) {
        Write-Warning "$Tag No latest $Kind artifact found in $Root for label $Label; using newest retrospective artifact: $($RetrospectiveMatches[0].FullName)"
        return $RetrospectiveMatches[0].FullName
    }

    $InitialFromReport = Resolve-InitialArtifactFromReport -ArtifactRoot $Root -Kind $Kind
    if (-not [string]::IsNullOrWhiteSpace($InitialFromReport) -and (Test-Path -LiteralPath $InitialFromReport)) {
        Write-Warning "$Tag No latest or retrospective $Kind artifact found in $Root for label $Label; falling back to initial artifact from final_report: $InitialFromReport"
        return $InitialFromReport
    }

    foreach ($Candidate in $FallbackCandidates) {
        if (Test-Path -LiteralPath $Candidate) {
            Write-Warning "$Tag No latest or retrospective $Kind artifact found in $Root for label $Label; falling back to v1: $Candidate"
            return $Candidate
        }
    }
    $FallbackMatches = @(Get-ChildItem -LiteralPath $Root -Filter $FallbackFilter -ErrorAction SilentlyContinue)
    if ($FallbackMatches.Count -eq 1) {
        Write-Warning "$Tag No latest or retrospective $Kind artifact found in $Root for label $Label; falling back to v1: $($FallbackMatches[0].FullName)"
        return $FallbackMatches[0].FullName
    }
    if ($FallbackMatches.Count -gt 1) {
        throw "Multiple v1 fallback $Kind artifacts matched in ${Root}: $($FallbackMatches.FullName -join '; '). Pass -$($Kind.Substring(0, 1).ToUpperInvariant() + $Kind.Substring(1))File explicitly."
    }

    $VersionedMatches = @(
        Get-ChildItem -LiteralPath $Root -Filter $VersionedFilter -ErrorAction SilentlyContinue |
            Sort-Object Name
    )
    if ($VersionedMatches.Count -ge 1) {
        Write-Warning "$Tag No latest, retrospective, final_report initial, or v1 $Kind artifact found in $Root for label $Label; using first versioned artifact: $($VersionedMatches[0].FullName)"
        return $VersionedMatches[0].FullName
    }
    throw "No latest, retrospective, initial, v1, or versioned fallback $Kind artifact found in $Root for label $Label."
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$ArgsList
    )
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) {
        $Python = "python"
    }
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "$Tag Starting: $Name"
    Write-Host "============================================================"
    & $Python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "$Tag $Name failed with code $LASTEXITCODE"
    }
}

$Config = Resolve-AttackConfig -Value $AttackLabel
$EffectiveIvStatefulRuntime = (
    $EnableIvStatefulRuntime -and
    $Config.Label -eq "insufficient_validation"
)

$env:GLM_EN_ENABLE_ANTHROPIC = if ($EnableAnthropic) { "1" } else { "0" }
if (-not [string]::IsNullOrWhiteSpace($ZhipuApiKey)) {
    $env:ZHIPU_API_KEY = $ZhipuApiKey
}
if (-not [string]::IsNullOrWhiteSpace($MiniMaxApiKey)) {
    $env:MINIMAX_API_KEY = $MiniMaxApiKey
}
if (-not [string]::IsNullOrWhiteSpace($XunfeiApiKey)) {
    $env:XUNFEI_API_KEY = $XunfeiApiKey
}
if ($MiniMaxThinkingMode -eq "stage-default") {
    Remove-Item Env:MINIMAX_THINKING_OVERRIDE -ErrorAction SilentlyContinue
}
else {
    $env:MINIMAX_THINKING_OVERRIDE = $MiniMaxThinkingMode
    Write-Warning (
        "$Tag MiniMaxThinkingMode=$MiniMaxThinkingMode is a global override; " +
        "it takes precedence over JudgeThinking and AggregatorThinking."
    )
}

$EffectiveJudgeThinking = $JudgeThinking
if ($MiniMaxThinkingMode -in @("disabled", "adaptive")) {
    $EffectiveJudgeThinking = $MiniMaxThinkingMode
}
$EffectiveJudgeMaxTokens = $JudgeMaxTokens
if ($EffectiveJudgeMaxTokens -le 0) {
    $EffectiveJudgeMaxTokens = if ($EffectiveJudgeThinking -eq "adaptive") {
        16384
    }
    else {
        8192
    }
}

$RuleRoot = "data\rules\$Episode"
$PlanRoot = "data\plans\$Episode"
if ([string]::IsNullOrWhiteSpace($RuleFile)) {
    $RuleFile = Resolve-LatestArtifact -Root $RuleRoot -Label $Config.Label -Kind "rule"
}
if ([string]::IsNullOrWhiteSpace($PlanFile)) {
    $PlanFile = Resolve-LatestArtifact -Root $PlanRoot -Label $Config.Label -Kind "plan"
}

$EvalCsvStem = if ($EvaluationSplit -eq "hard") { "evaluation_hard" } else { "evaluation" }
$CsvDirPath = Join-Path $CsvRoot $Config.CsvDir
if ([string]::IsNullOrWhiteSpace($EvalPosCsv)) {
    $EvalPosCsv = Join-Path $CsvDirPath "$($Config.Prefix)_${EvalCsvStem}_pos.csv"
}
if ([string]::IsNullOrWhiteSpace($EvalNegCsv)) {
    $EvalNegCsv = Join-Path $CsvDirPath "$($Config.Prefix)_${EvalCsvStem}_neg.csv"
}

Assert-ExistingFile -PathValue $RuleFile -Name "RuleFile"
Assert-ExistingFile -PathValue $PlanFile -Name "PlanFile"
Assert-ExistingFile -PathValue $EvalPosCsv -Name "EvalPosCsv"
Assert-ExistingFile -PathValue $EvalNegCsv -Name "EvalNegCsv"

$EffectiveOutputPrefix = $OutputPrefix.Trim()
if ([string]::IsNullOrWhiteSpace($EffectiveOutputPrefix)) {
    $ResultsOutPath = "data\results"
    $SummaryOutPath = "data\results"
    $SlimResultsOutPath = "data\results"
    $FailedCsvOutPath = "data\results"
}
else {
    $ResultsOutPath = "data\results\$($EffectiveOutputPrefix)_results.json"
    $SummaryOutPath = "data\results\$($EffectiveOutputPrefix)_summary.json"
    $SlimResultsOutPath = "data\results\$($EffectiveOutputPrefix)_slim_results.json"
    $FailedCsvOutPath = "data\results\$($EffectiveOutputPrefix)_failed.csv"
}
$TranscriptDir = if ([string]::IsNullOrWhiteSpace($EffectiveOutputPrefix)) {
    "data\results\$Episode\llm_transcripts"
}
else {
    "data\results\$Episode\llm_transcripts_$EffectiveOutputPrefix"
}

$EvalArgs = @(
    "experiments\evaluate.py",
    "--pos-csv", $EvalPosCsv,
    "--neg-csv", $EvalNegCsv,
    "--label", $Config.Label,
    "--positive-label", $Config.PositiveLabel,
    "--rule-file", $RuleFile,
    "--plan-file", $PlanFile,
    "--llm-model", $LlmModel,
    "--llm-provider", $LlmProvider,
    "--base-cache-dir", "data\cache",
    "--source-cache-dir", "data\cache\contracts",
    "--logs-dir", "data\log",
    "--episode", "$Episode",
    "--max-view-chars", "$MaxViewChars",
    "--max-context-chars", "$MaxContextChars",
    "--judge-max-tokens", "$EffectiveJudgeMaxTokens",
    "--aggregator-max-tokens", "$AggregatorMaxTokens",
    "--judge-thinking", $JudgeThinking,
    "--aggregator-thinking", $AggregatorThinking,
    "--access-control-binding-mode", $AccessControlBindingMode,
    "--reentrancy-binding-mode", $ReentrancyBindingMode,
    "--followup-context-mode", $FollowupContextMode,
    "--judge-followup-mode", $JudgeFollowupMode,
    "--judge-expanded-followup-views", "$JudgeExpandedFollowupViews",
    "--rate-limit-retry-attempts", "$RateLimitRetryAttempts",
    "--rate-limit-retry-delay-seconds", "$RateLimitRetryDelaySeconds",
    "--results-out", $ResultsOutPath,
    "--out", $SummaryOutPath,
    "--slim-results-out", $SlimResultsOutPath,
    "--failed-csv-out", $FailedCsvOutPath
)

if (-not [string]::IsNullOrWhiteSpace($PlannerModel)) {
    $EvalArgs += @("--planner-model", $PlannerModel)
}
if (-not [string]::IsNullOrWhiteSpace($JudgeModel)) {
    $EvalArgs += @("--judge-model", $JudgeModel)
}
if (-not [string]::IsNullOrWhiteSpace($EnvModel)) {
    $EvalArgs += @("--env-model", $EnvModel)
}
if ($OnlyHead -ge 0) {
    $EvalArgs += @("--only-head", "$OnlyHead")
}
if ($UseEnvironment) {
    $EvalArgs += @("--use-environment")
}
if ($EnableSourceTools) {
    $EvalArgs += @("--enable-source-tools")
}
if ($ForceRefreshSource) {
    $EvalArgs += @("--force-refresh-source")
}
if ($ForceRebuildPacket) {
    $EvalArgs += @("--force-rebuild-packet")
}
if ($EffectiveIvStatefulRuntime) {
    $EvalArgs += @("--enable-iv-stateful-runtime")
}
else {
    $EvalArgs += @("--disable-iv-stateful-runtime")
}
if ($DisableStatefulBindings) {
    $EvalArgs += @("--disable-stateful-bindings")
}
if ($EnableDynamicAggregation) {
    $EvalArgs += @("--enable-dynamic-aggregation")
}
else {
    $EvalArgs += @("--disable-dynamic-aggregation")
}
if ($RateLimitSerialFallback) {
    $EvalArgs += @("--rate-limit-serial-fallback")
}
else {
    $EvalArgs += @("--disable-rate-limit-serial-fallback")
}
if ($AdaptiveEvidence) {
    $EvalArgs += @(
        "--adaptive-evidence",
        "--adaptive-evidence-mode", $AdaptiveEvidenceMode,
        "--adaptive-evidence-max-direct-trace-nodes", "$AdaptiveEvidenceMaxDirectTraceNodes",
        "--adaptive-evidence-max-direct-trace-chars", "$AdaptiveEvidenceMaxDirectTraceChars",
        "--adaptive-evidence-max-medium-trace-nodes", "$AdaptiveEvidenceMaxMediumTraceNodes"
    )
    if ($AdaptiveEvidenceDebug) {
        $EvalArgs += @("--adaptive-evidence-debug")
    }
}
if ($ParallelJudge) {
    $EvalArgs += @("--parallel-judge", "--judge-concurrency", "$JudgeConcurrency")
}
if ($StructuredJudgeLogs) {
    $EvalArgs += @("--structured-judge-logs")
}
else {
    $EvalArgs += @("--no-structured-judge-logs")
}
if ($SaveLlmTranscripts) {
    $EvalArgs += @(
        "--save-llm-transcripts",
        "--llm-transcripts-dir", $TranscriptDir
    )
}

Write-Host ""
Write-Host "============================================================"
Write-Host "$Tag Attack-vs-benign latest-rule evaluation"
Write-Host "$Tag Started at: $($StartedAt.ToString("yyyy-MM-dd HH:mm:ss"))"
Write-Host "============================================================"
Write-Host "$Tag Episode: $Episode"
Write-Host "$Tag Evaluation split: $EvaluationSplit"
Write-Host "$Tag Attack label: $($Config.Label) positive_label=$($Config.PositiveLabel)"
Write-Host "$Tag Rule: $RuleFile"
Write-Host "$Tag Plan: $PlanFile"
Write-Host "$Tag Eval pos: $EvalPosCsv"
Write-Host "$Tag Eval neg: $EvalNegCsv"
Write-Host "$Tag Model: $LlmModel provider=$LlmProvider"
Write-Host "$Tag glm-en Anthropic transport: $EnableAnthropic"
Write-Host "$Tag MiniMax thinking mode: $MiniMaxThinkingMode"
Write-Host "$Tag Judge thinking: $JudgeThinking; Aggregator thinking: $AggregatorThinking"
Write-Host "$Tag IV stateful runtime: $EffectiveIvStatefulRuntime"
Write-Host "$Tag Access-control binding mode: $AccessControlBindingMode"
Write-Host "$Tag Reentrancy binding mode: $ReentrancyBindingMode"
Write-Host "$Tag Dynamic aggregation: $EnableDynamicAggregation"
Write-Host "$Tag Follow-up context mode: $FollowupContextMode"
Write-Host "$Tag Judge follow-up: mode=$JudgeFollowupMode expanded_views=$JudgeExpandedFollowupViews"
Write-Host "$Tag Completion tokens: judge=$EffectiveJudgeMaxTokens (auto=$($JudgeMaxTokens -eq 0), effective_thinking=$EffectiveJudgeThinking) aggregator=$AggregatorMaxTokens"
Write-Host "$Tag 429 fallback: enabled=$RateLimitSerialFallback retries=$RateLimitRetryAttempts initial_delay=${RateLimitRetryDelaySeconds}s"
Write-Host "$Tag Parallel judge: $ParallelJudge concurrency=$JudgeConcurrency"
if (-not [string]::IsNullOrWhiteSpace($EffectiveOutputPrefix)) {
    Write-Host "$Tag Output prefix: $EffectiveOutputPrefix"
}

if ($DryRun) {
    Write-Host ""
    Write-Host "$Tag Dry run only. Eval command:"
    Write-Host "python $($EvalArgs -join ' ')"
    Restore-Environment
    return
}

Invoke-PythonStep -Name "$($Config.PositiveLabel) benign evaluation episode $Episode" -ArgsList $EvalArgs

$FinishedAt = Get-Date
$Elapsed = $FinishedAt - $StartedAt
$ElapsedText = "{0:00}:{1:00}:{2:00}.{3:000}" -f `
    [int]$Elapsed.TotalHours, $Elapsed.Minutes, $Elapsed.Seconds, $Elapsed.Milliseconds

Write-Host ""
Write-Host "============================================================"
Write-Host "$Tag Evaluation complete."
Write-Host ("$Tag Completed at: {0}" -f $FinishedAt.ToString("yyyy-MM-dd HH:mm:ss"))
Write-Host "$Tag Total elapsed: $ElapsedText"
if ([string]::IsNullOrWhiteSpace($EffectiveOutputPrefix)) {
    Write-Host "$Tag Eval summary: data\results\$Episode\eval_summary.json"
    Write-Host "$Tag Eval slim: data\results\$Episode\eval_slim_results.json"
    Write-Host "$Tag Failed CSV: data\results\$Episode\failed.csv"
}
else {
    Write-Host "$Tag Eval summary: data\results\$Episode\$($EffectiveOutputPrefix)_summary.json"
    Write-Host "$Tag Eval slim: data\results\$Episode\$($EffectiveOutputPrefix)_slim_results.json"
    Write-Host "$Tag Failed CSV: data\results\$Episode\$($EffectiveOutputPrefix)_failed.csv"
}
Write-Host "============================================================"

Restore-Environment
