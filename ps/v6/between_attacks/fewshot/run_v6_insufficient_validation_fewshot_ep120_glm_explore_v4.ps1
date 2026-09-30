param(
    [int]$Episode = 120,
    [string]$LlmModel = "glm-5.1",
    [string]$LlmProvider = "glm",
    [Alias("enable_anthropic")]
    [bool]$EnableAnthropic = $false,
    [string]$ZhipuApiKey = "",
    [string]$MiniMaxApiKey = "",
    [ValidateSet("stage-default", "disabled", "adaptive")]
    [string]$MiniMaxThinkingMode = "stage-default",
    [ValidateSet("disabled", "adaptive")]
    [string]$JudgeThinking = "adaptive",
    [ValidateSet("disabled", "adaptive")]
    [string]$AggregatorThinking = "disabled",
    [ValidateSet("disabled", "adaptive")]
    [string]$ReviewThinking = "adaptive",
    [ValidateSet("disabled", "adaptive")]
    [string]$UpdateThinking = "adaptive",
    [int]$MaxRounds = 4,
    [string]$EvolutionMode = "explore",
    [string]$FinalValidationMode = "strict",
    [ValidateSet("auto", "reuse", "rerun")]
    [string]$FinalValidationResultSource = "auto",
    [ValidateSet("auto", "rule", "plan")]
    [string]$UpdateTarget = "auto",
    [bool]$ForceFinalValidationRerun = $false,
    [ValidateSet("auto", "legacy", "error-focused")]
    [string]$ReviewCompactMode = "error-focused",
    [int]$MaxReviewPromptChars = 250000,
    [int]$StopErrors = 0,
    [int]$MaxViewChars = 80000,
    [int]$MaxContextChars = 200000,
    [bool]$ForceRebuildPacket = $true,
    [bool]$EnableSourceTools = $true,
    [bool]$ForceRefreshSource = $false,
    [bool]$SaveLlmTranscripts = $true,
    [bool]$AdaptiveEvidence = $false,
    [string]$AdaptiveEvidenceMode = "off",
    [bool]$AdaptiveEvidenceDebug = $false,
    [ValidateRange(1, 1000000)]
    [int]$AdaptiveEvidenceMaxDirectTraceNodes = 80,
    [ValidateRange(1000, 1000000)]
    [int]$AdaptiveEvidenceMaxDirectTraceChars = 45000,
    [ValidateRange(1, 1000000)]
    [int]$AdaptiveEvidenceMaxMediumTraceNodes = 250,
    [bool]$ParallelJudge = $false,
    [int]$JudgeConcurrency = 1,
    [ValidateRange(1, 32768)]
    [int]$JudgeMaxTokens = 24576,
    [ValidateRange(1, 32768)]
    [int]$AggregatorMaxTokens = 8192,
    [bool]$StructuredJudgeLogs = $true,
    [bool]$EnableCandidateJudgeReuse = $true,
    [bool]$EnableRulePartialAcceptance = $true,
    [ValidateRange(2, 20)]
    [int]$RulePartialMaxDeltas = 6,
    [ValidateRange(1, 50)]
    [int]$RulePartialMaxEvaluations = 10,
    [bool]$EnableRuntimeEarlyStop = $false,
    [bool]$EnableIvStatefulRuntime = $true,
    [bool]$EnableDynamicAggregation = $false,
    [ValidateSet("legacy", "unified")]
    [string]$FollowupContextMode = "unified",
    [bool]$RateLimitSerialFallback = $true,
    [int]$RateLimitRetryAttempts = 3,
    [double]$RateLimitRetryDelaySeconds = 7.0,
    [ValidateRange(0, 1000000)]
    [int]$MaxFpIncrease = 1,
    [ValidateRange(0, 1000000)]
    [int]$MaxFnIncrease = 1,
    [int]$MaxUncertainIncrease = 2,
    [int]$MaxUncertainBenignIncrease = 1,
    [int]$MaxGuardFpIncrease = 1,
    [int]$MaxGuardErrorIncrease = 1,
    [int]$MaxGuardUncertainIncrease = 1,
    [bool]$AllowGuardRepair = $false,
    [int]$TemporaryFpIncrease = 3,
    [int]$RepairMinFnDecrease = 1,
    [int]$MaxRepairRounds = 2,
    [bool]$EnableCandidateRepair = $true,
    [bool]$EnableRecallRepair = $true,
    [int]$RecallRepairMinFixedFp = 1,
    [int]$RecallRepairMaxNewFn = 10,
    [bool]$CompressRule = $false,
    [ValidateSet("benign_only", "attack_contrastive", "mixed")]
    [string]$NegativeTrainingMode = "attack_contrastive",
    [string]$RuleFile = "data\results\rule_plan_pairs\insufficient_validation_stateful_v6_rule-from_gpt.json",
    [string]$PlanFile = "data\results\rule_plan_pairs\insufficient_validation_stateful_v6_plan-from_gpt.json",
    [string]$FewshotPosCsv = "data\csv\v4\Insufficient validation\insufficient_validation_Fewshot_pos.csv",
    [string]$FewshotNegCsv = "data\csv\v4\Insufficient validation\insufficient_validation_Fewshot_neg.csv",
    # v4 has no dedicated IV guard; keep the v2 guard instead of leaking evaluation data.
    [string]$GuardNegCsv = "data\csv\v4\Insufficient validation\insufficient_validation_guard.csv",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$OldGlmEnEnableAnthropic = $env:GLM_EN_ENABLE_ANTHROPIC
function Set-GlmEnAnthropicOverride {
    if ($EnableAnthropic) {
        $env:GLM_EN_ENABLE_ANTHROPIC = "1"
    }
    else {
        $env:GLM_EN_ENABLE_ANTHROPIC = "0"
    }
}

function Restore-GlmEnAnthropicOverride {
    if ($null -eq $OldGlmEnEnableAnthropic) {
        Remove-Item Env:GLM_EN_ENABLE_ANTHROPIC -ErrorAction SilentlyContinue
    }
    else {
        $env:GLM_EN_ENABLE_ANTHROPIC = $OldGlmEnEnableAnthropic
    }
}
$Tag = "[IV-V6-E$Episode-$($LlmProvider.ToUpperInvariant())-Explore]"
$TaskStartedAt = Get-Date
$OldZhipuApiKey = $env:ZHIPU_API_KEY
$OldMiniMaxApiKey = $env:MINIMAX_API_KEY
$OldMiniMaxThinkingOverride = $env:MINIMAX_THINKING_OVERRIDE

function Restore-MiniMaxThinkingOverride {
    if ($null -eq $OldMiniMaxThinkingOverride) {
        Remove-Item Env:MINIMAX_THINKING_OVERRIDE -ErrorAction SilentlyContinue
    }
    else {
        $env:MINIMAX_THINKING_OVERRIDE = $OldMiniMaxThinkingOverride
    }
}

function Restore-ProviderEnvironment {
    Restore-GlmEnAnthropicOverride
    if (-not [string]::IsNullOrWhiteSpace($ZhipuApiKey)) {
        $env:ZHIPU_API_KEY = $OldZhipuApiKey
    }
    if (-not [string]::IsNullOrWhiteSpace($MiniMaxApiKey)) {
        $env:MINIMAX_API_KEY = $OldMiniMaxApiKey
    }
    Restore-MiniMaxThinkingOverride
}

trap {
    $FailedAt = Get-Date
    $Elapsed = $FailedAt - $TaskStartedAt
    $ElapsedText = "{0:00}:{1:00}:{2:00}.{3:000}" -f `
        [int]$Elapsed.TotalHours, $Elapsed.Minutes, $Elapsed.Seconds, $Elapsed.Milliseconds
    Write-Host ("$Tag Failed at: {0}" -f $FailedAt.ToString("yyyy-MM-dd HH:mm:ss"))
    Write-Host "$Tag Elapsed before failure: $ElapsedText"
    Write-Host "$Tag Error: $_"
    Restore-ProviderEnvironment
    throw
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Set-Location $ProjectRoot
Set-GlmEnAnthropicOverride
Write-Host "$Tag glm-en Anthropic transport: $EnableAnthropic"
Write-Host "$Tag Project root: $ProjectRoot"

if (-not (Test-Path -LiteralPath ".\.venv\Scripts\Activate.ps1")) {
    throw "Virtual environment activation script not found: .\.venv\Scripts\Activate.ps1"
}

Write-Host "$Tag Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

if (-not [string]::IsNullOrWhiteSpace($ZhipuApiKey)) {
    $env:ZHIPU_API_KEY = $ZhipuApiKey
    Write-Host "$Tag Using ZHIPU_API_KEY from script parameter."
}
if (-not [string]::IsNullOrWhiteSpace($MiniMaxApiKey)) {
    $env:MINIMAX_API_KEY = $MiniMaxApiKey
    Write-Host "$Tag Using MINIMAX_API_KEY from script parameter."
}

if ($MiniMaxThinkingMode -eq "stage-default") {
    Remove-Item Env:MINIMAX_THINKING_OVERRIDE -ErrorAction SilentlyContinue
}
else {
    $env:MINIMAX_THINKING_OVERRIDE = $MiniMaxThinkingMode
    Write-Warning (
        "$Tag MiniMaxThinkingMode=$MiniMaxThinkingMode globally overrides " +
        "JudgeThinking, AggregatorThinking, ReviewThinking, and UpdateThinking."
    )
}

$Label = "insufficient_validation"
$PositiveLabel = "Insufficient validation"
$AttackDescription = "Insufficient validation: a value-sensitive protocol path consumes user-supplied, external, callback, return, oracle, state, or business-assumption data; the data/state is invalid, stale, inconsistent, boundary-breaking, or adversarial; missing or inadequate validation lets the invalid value reach sensitive logic and causes an incorrect outcome, invariant break, excessive payout, asset release, accounting/state harm, or user/protocol loss."

function Assert-ConfiguredFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )
    if ([string]::IsNullOrWhiteSpace($PathValue) -or $PathValue.StartsWith("TODO_")) {
        throw "Please fill `$${Name} before running this script."
    }
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "Input file not found for `$${Name}: $PathValue"
    }
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string[]]$ArgsList
    )
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "$Tag Starting: $Name"
    Write-Host "============================================================"
    & python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "$Tag $Name failed with code $LASTEXITCODE"
    }
    Write-Host "$Tag Finished: $Name"
}

foreach ($item in @(
    @($RuleFile, "RuleFile"),
    @($PlanFile, "PlanFile"),
    @($FewshotPosCsv, "FewshotPosCsv"),
    @($FewshotNegCsv, "FewshotNegCsv"),
    @($GuardNegCsv, "GuardNegCsv")
)) {
    Assert-ConfiguredFile -PathValue $item[0] -Name $item[1]
}

$PlanJson = Get-Content -LiteralPath $PlanFile -Raw | ConvertFrom-Json
$PlanStatefulMode = [string]$PlanJson.metadata.stateful_runtime.mode
if ($EnableIvStatefulRuntime -and $PlanStatefulMode -ne "insufficient_validation_object_binding_v1") {
    Write-Warning (
        "$Tag Plan does not already declare IV stateful metadata. Runtime will " +
        "re-annotate its C1/C2/C3 semantic flow before execution."
    )
}
if ($EnableIvStatefulRuntime -and $ParallelJudge) {
    Write-Warning (
        "$Tag ParallelJudge was requested, but IV stateful dependencies require " +
        "serial Judge execution. Runtime will disable parallel Judge."
    )
}

$TranscriptDir = "data\results\$Episode\llm_transcripts"
$ArgsList = @(
    "experiments\train_fewshot.py",
    "--pos-csv", $FewshotPosCsv,
    "--neg-csv", $FewshotNegCsv,
    "--guard-neg-csv", $GuardNegCsv,
    "--label", $Label,
    "--positive-label", $PositiveLabel,
    "--attack-description", $AttackDescription,
    "--llm-model", $LlmModel,
    "--llm-provider", $LlmProvider,
    "--rule-file", $RuleFile,
    "--plan-file", $PlanFile,
    "--max-rounds", "$MaxRounds",
    "--stop-errors", "$StopErrors",
    "--evolution-mode", $EvolutionMode,
    "--final-validation-mode", $FinalValidationMode,
    "--final-validation-result-source", $FinalValidationResultSource,
    "--update-target", $UpdateTarget,
    "--review-compact-mode", $ReviewCompactMode,
    "--negative-training-mode", $NegativeTrainingMode,
    "--rules-dir", "data\rules",
    "--plans-dir", "data\plans",
    "--artifacts-dir", "data\results",
    "--logs-dir", "data\log",
    "--episode", "$Episode",
    "--base-cache-dir", "data\cache",
    "--source-cache-dir", "data\cache\contracts",
    "--max-view-chars", "$MaxViewChars",
    "--max-context-chars", "$MaxContextChars",
    "--judge-max-tokens", "$JudgeMaxTokens",
    "--aggregator-max-tokens", "$AggregatorMaxTokens",
    "--judge-thinking", $JudgeThinking,
    "--aggregator-thinking", $AggregatorThinking,
    "--review-thinking", $ReviewThinking,
    "--update-thinking", $UpdateThinking,
    "--max-fp-increase", "$MaxFpIncrease",
    "--max-fn-increase", "$MaxFnIncrease",
    "--max-uncertain-increase", "$MaxUncertainIncrease",
    "--max-uncertain-benign-increase", "$MaxUncertainBenignIncrease",
    "--max-guard-fp-increase", "$MaxGuardFpIncrease",
    "--max-guard-error-increase", "$MaxGuardErrorIncrease",
    "--max-guard-uncertain-increase", "$MaxGuardUncertainIncrease",
    "--temporary-fp-increase", "$TemporaryFpIncrease",
    "--repair-min-fn-decrease", "$RepairMinFnDecrease",
    "--max-repair-rounds", "$MaxRepairRounds",
    "--recall-repair-min-fixed-fp", "$RecallRepairMinFixedFp",
    "--recall-repair-max-new-fn", "$RecallRepairMaxNewFn",
    "--followup-context-mode", $FollowupContextMode,
    "--rate-limit-retry-attempts", "$RateLimitRetryAttempts",
    "--rate-limit-retry-delay-seconds", "$RateLimitRetryDelaySeconds",
    "--rule-partial-max-deltas", "$RulePartialMaxDeltas",
    "--rule-partial-max-evaluations", "$RulePartialMaxEvaluations"
)

if ($MaxReviewPromptChars -gt 0) {
    $ArgsList += @("--max-review-prompt-chars", "$MaxReviewPromptChars")
}
if ($ForceFinalValidationRerun) {
    $ArgsList += @("--force-final-validation-rerun")
}
if ($EnableCandidateRepair) {
    $ArgsList += @("--enable-candidate-repair")
}
else {
    $ArgsList += @("--disable-candidate-repair")
}
if ($EnableRecallRepair) {
    $ArgsList += @("--enable-recall-repair")
}
else {
    $ArgsList += @("--disable-recall-repair")
}
if ($EnableSourceTools) {
    $ArgsList += @("--enable-source-tools")
}
if ($ForceRefreshSource) {
    $ArgsList += @("--force-refresh-source")
}
if ($ForceRebuildPacket) {
    $ArgsList += @("--force-rebuild-packet")
}
if ($EnableCandidateJudgeReuse) {
    $ArgsList += @("--enable-candidate-judge-reuse")
}
else {
    $ArgsList += @("--disable-candidate-judge-reuse")
}
if ($EnableRulePartialAcceptance) {
    $ArgsList += @("--enable-rule-partial-acceptance")
}
else {
    $ArgsList += @("--disable-rule-partial-acceptance")
}
if ($EnableRuntimeEarlyStop) {
    $ArgsList += @("--enable-runtime-early-stop")
}
if ($EnableIvStatefulRuntime) {
    $ArgsList += @("--enable-iv-stateful-runtime")
}
else {
    $ArgsList += @("--disable-iv-stateful-runtime")
}
if ($EnableDynamicAggregation) {
    $ArgsList += @("--enable-dynamic-aggregation")
}
else {
    $ArgsList += @("--disable-dynamic-aggregation")
}
if ($RateLimitSerialFallback) {
    $ArgsList += @("--rate-limit-serial-fallback")
}
else {
    $ArgsList += @("--disable-rate-limit-serial-fallback")
}
if ($AdaptiveEvidence) {
    $ArgsList += @(
        "--adaptive-evidence",
        "--adaptive-evidence-mode", $AdaptiveEvidenceMode,
        "--adaptive-evidence-max-direct-trace-nodes", "$AdaptiveEvidenceMaxDirectTraceNodes",
        "--adaptive-evidence-max-direct-trace-chars", "$AdaptiveEvidenceMaxDirectTraceChars",
        "--adaptive-evidence-max-medium-trace-nodes", "$AdaptiveEvidenceMaxMediumTraceNodes"
    )
    if ($AdaptiveEvidenceDebug) {
        $ArgsList += @("--adaptive-evidence-debug")
    }
}
if ($AllowGuardRepair) {
    $ArgsList += @("--allow-guard-repair")
}
if ($CompressRule) {
    $ArgsList += @("--compress-rule")
}
if ($ParallelJudge) {
    $ArgsList += @("--parallel-judge", "--judge-concurrency", "$JudgeConcurrency")
}
if ($StructuredJudgeLogs) {
    $ArgsList += @("--structured-judge-logs")
}
else {
    $ArgsList += @("--no-structured-judge-logs")
}
if ($SaveLlmTranscripts) {
    $ArgsList += @(
        "--save-llm-transcripts",
        "--llm-transcripts-dir", $TranscriptDir
    )
}

Write-Host ""
Write-Host "============================================================"
Write-Host "$Tag Insufficient Validation stateful few-shot"
Write-Host "$Tag Started at: $($TaskStartedAt.ToString("yyyy-MM-dd HH:mm:ss"))"
Write-Host "============================================================"
Write-Host "$Tag Episode: $Episode"
Write-Host "$Tag Rule: $RuleFile"
Write-Host "$Tag Plan: $PlanFile"
Write-Host "$Tag Plan stateful mode: $PlanStatefulMode"
Write-Host "$Tag IV stateful runtime: $EnableIvStatefulRuntime"
Write-Host "$Tag Pos: $FewshotPosCsv"
Write-Host "$Tag Neg: $FewshotNegCsv"
Write-Host "$Tag Guard neg: $GuardNegCsv"
Write-Host "$Tag Model: $LlmModel provider=$LlmProvider"
Write-Host "$Tag MiniMax thinking mode: $MiniMaxThinkingMode"
Write-Host "$Tag Judge thinking: $JudgeThinking; Aggregator thinking: $AggregatorThinking"
Write-Host "$Tag Review thinking: $ReviewThinking; Update thinking: $UpdateThinking"
Write-Host "$Tag GLM thinking from environment: $($env:GLM_THINKING)"
Write-Host "$Tag Evolution: mode=$EvolutionMode final_validation=$FinalValidationMode"
Write-Host "$Tag Final validation result source: $FinalValidationResultSource"
Write-Host "$Tag Update target: $UpdateTarget; compress_rule=$CompressRule"
Write-Host "$Tag Review compact mode: $ReviewCompactMode prompt_cap=$MaxReviewPromptChars"
Write-Host "$Tag Negative training mode: $NegativeTrainingMode"
Write-Host "$Tag Dynamic aggregation: $EnableDynamicAggregation"
Write-Host "$Tag Follow-up context mode: $FollowupContextMode"
Write-Host "$Tag Completion tokens: judge=$JudgeMaxTokens aggregator=$AggregatorMaxTokens"
Write-Host "$Tag Parallel judge requested: $ParallelJudge concurrency=$JudgeConcurrency"
Write-Host "$Tag Candidate reuse: $EnableCandidateJudgeReuse; guard_repair=$AllowGuardRepair"
Write-Host "$Tag Rule partial acceptance: enabled=$EnableRulePartialAcceptance max_deltas=$RulePartialMaxDeltas max_evaluations=$RulePartialMaxEvaluations"
Write-Host "$Tag 429 fallback: enabled=$RateLimitSerialFallback retries=$RateLimitRetryAttempts initial_delay=${RateLimitRetryDelaySeconds}s"

if ($DryRun) {
    Write-Host ""
    Write-Host "$Tag Dry run only. Few-shot command:"
    Write-Host "python $($ArgsList -join ' ')"
}
else {
    Invoke-PythonStep -Name "Insufficient Validation few-shot episode $Episode" -ArgsList $ArgsList
}

$TaskFinishedAt = Get-Date
$TaskElapsed = $TaskFinishedAt - $TaskStartedAt
$TaskElapsedText = "{0:00}:{1:00}:{2:00}.{3:000}" -f `
    [int]$TaskElapsed.TotalHours, $TaskElapsed.Minutes, $TaskElapsed.Seconds, $TaskElapsed.Milliseconds

if (-not $DryRun) {
    Write-Host ""
    Write-Host "$Tag Finished episode $Episode."
    Write-Host ("$Tag Completed at: {0}" -f $TaskFinishedAt.ToString("yyyy-MM-dd HH:mm:ss"))
    Write-Host "$Tag Total elapsed: $TaskElapsedText"
    Write-Host "$Tag Report: data\results\$Episode\final_report.json"
    Write-Host "$Tag Rule: data\rules\$Episode\insufficient_validation__latest.json"
    Write-Host "$Tag Plan: data\plans\$Episode\insufficient_validation__plan_latest.json"
}

Restore-ProviderEnvironment
