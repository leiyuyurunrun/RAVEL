param(
    [int]$Episode = 62,
    [int]$EvalEpisode = 62,
    [string]$LlmModel = "glm-5.2",
    [string]$ColdStartModel = "",
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
    [bool]$ReviewLabelRationale = $false,
    [string]$SeedRuleFile = "",
    [string]$SeedPlanFile = "",
    [bool]$GenerateInitialPlanWithAgent = $false,
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
    [bool]$ParallelJudge = $true,
    [int]$JudgeConcurrency = 2,
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
    [bool]$EnableIvStatefulRuntime = $false,
    [ValidateSet("prompt", "stateful")]
    [string]$AccessControlBindingMode = "stateful",
    [bool]$EnableDynamicAggregation = $false,
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
    [string]$NegativeTrainingMode = "attack_contrastive",
    [string]$FewshotPosCsv = "data\csv\v4\Access control\access_control_Fewshot_pos.csv",
    [string]$FewshotNegCsv = "data\csv\v4\Access control\access_control_Fewshot_neg.csv",
    [string]$GuardNegCsv = "data\csv\v4\Access control\access_control_guard.csv",
    [string]$EvalPosCsv = "data\csv\v4\Access control\access_control_evaluation_pos.csv",
    [string]$EvalNegCsv = "data\csv\v4\Access control\access_control_evaluation_neg.csv",
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
$UseSeedArtifacts = (
    -not [string]::IsNullOrWhiteSpace($SeedRuleFile) -or
    -not [string]::IsNullOrWhiteSpace($SeedPlanFile)
)
if ($UseSeedArtifacts) {
    $Tag = "[AC-V6-Seed-$($LlmProvider.ToUpperInvariant())]"
}
else {
    $Tag = "[AC-V6-ColdStart-$($LlmProvider.ToUpperInvariant())]"
}

$PipelineStartedAt = Get-Date
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

trap {
    $FailedAt = Get-Date
    $Elapsed = $FailedAt - $PipelineStartedAt
    $ElapsedText = "{0:00}:{1:00}:{2:00}.{3:000}" -f [int]$Elapsed.TotalHours, $Elapsed.Minutes, $Elapsed.Seconds, $Elapsed.Milliseconds
    Write-Host ("$Tag Failed at: {0}" -f $FailedAt.ToString("yyyy-MM-dd HH:mm:ss"))
    Write-Host "$Tag Elapsed before failure: $ElapsedText"
    Write-Host "$Tag Error: $_"
    if (-not [string]::IsNullOrWhiteSpace($ZhipuApiKey)) {
        $env:ZHIPU_API_KEY = $OldZhipuApiKey
    }
    if (-not [string]::IsNullOrWhiteSpace($MiniMaxApiKey)) {
        $env:MINIMAX_API_KEY = $OldMiniMaxApiKey
    }
    Restore-MiniMaxThinkingOverride
    Restore-GlmEnAnthropicOverride
    throw
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Set-Location $ProjectRoot
Set-GlmEnAnthropicOverride
Write-Host "$Tag glm-en Anthropic transport: $EnableAnthropic"

Write-Host "$Tag Project root: $ProjectRoot"

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
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
        "$Tag MiniMaxThinkingMode=$MiniMaxThinkingMode is a global override; " +
        "it takes precedence over JudgeThinking, AggregatorThinking, " +
        "ReviewThinking, and UpdateThinking."
    )
}

$Label = "access_control"
$PositiveLabel = "Access control"
$AttackDescription = "Access control: a transaction reaches a sensitive or protected operation such as permission changes, initializer/configuration updates, callback-protected paths, privileged mint/burn/withdraw/redeem/transfer actions, or movement of protocol/victim/shared assets; the actor, caller, callback sender, beneficiary, or execution path lacks legitimate authorization, entitlement, role, signature, or caller validation; the invocation causes protected-state mutation, privilege escalation, unauthorized approval, or release/appropriation of protected assets beyond legitimate entitlement."

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
    if (-not (Test-Path $PathValue)) {
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

Assert-ConfiguredFile -PathValue $FewshotPosCsv -Name "FewshotPosCsv"
Assert-ConfiguredFile -PathValue $FewshotNegCsv -Name "FewshotNegCsv"
Assert-ConfiguredFile -PathValue $GuardNegCsv -Name "GuardNegCsv"
Assert-ConfiguredFile -PathValue $EvalPosCsv -Name "EvalPosCsv"
Assert-ConfiguredFile -PathValue $EvalNegCsv -Name "EvalNegCsv"

if ($UseSeedArtifacts) {
    if (
        [string]::IsNullOrWhiteSpace($SeedRuleFile) -or
        [string]::IsNullOrWhiteSpace($SeedPlanFile)
    ) {
        throw "Seed mode requires both `$SeedRuleFile and `$SeedPlanFile."
    }
    Assert-ConfiguredFile -PathValue $SeedRuleFile -Name "SeedRuleFile"
    Assert-ConfiguredFile -PathValue $SeedPlanFile -Name "SeedPlanFile"
}

$FewshotTranscriptDir = "data\results\$Episode\llm_transcripts"
$FewshotArgs = @(
    "experiments\train_fewshot.py",
    "--pos-csv", $FewshotPosCsv,
    "--neg-csv", $FewshotNegCsv,
    "--guard-neg-csv", $GuardNegCsv,
    "--label", $Label,
    "--positive-label", $PositiveLabel,
    "--attack-description", $AttackDescription,
    "--llm-model", $LlmModel,
    "--llm-provider", $LlmProvider,
    "--max-rounds", "$MaxRounds",
    "--stop-errors", "$StopErrors",
    "--evolution-mode", $EvolutionMode,
    "--final-validation-mode", $FinalValidationMode,
    "--final-validation-result-source", $FinalValidationResultSource,
    "--update-target", $UpdateTarget,
    "--review-compact-mode", $ReviewCompactMode,
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
    "--negative-training-mode", "$NegativeTrainingMode",
    "--access-control-binding-mode", $AccessControlBindingMode,
    "--followup-context-mode", $FollowupContextMode,
    "--rate-limit-retry-attempts", "$RateLimitRetryAttempts",
    "--rate-limit-retry-delay-seconds", "$RateLimitRetryDelaySeconds",
    "--rule-partial-max-deltas", "$RulePartialMaxDeltas",
    "--rule-partial-max-evaluations", "$RulePartialMaxEvaluations"
)

if (-not [string]::IsNullOrWhiteSpace($ColdStartModel)) {
    $FewshotArgs += @("--cold-start-model", $ColdStartModel)
}

if ($UseSeedArtifacts) {
    $FewshotArgs += @(
        "--rule-file", $SeedRuleFile,
        "--plan-file", $SeedPlanFile
    )
}

if ($MaxReviewPromptChars -gt 0) {
    $FewshotArgs += @("--max-review-prompt-chars", "$MaxReviewPromptChars")
}
if ($ReviewLabelRationale) {
    $FewshotArgs += @("--enable-review-label-rationale")
}
else {
    $FewshotArgs += @("--disable-review-label-rationale")
}
if ($GenerateInitialPlanWithAgent -and -not $UseSeedArtifacts) {
    $FewshotArgs += @("--generate-initial-plan-with-agent")
}
if ($ForceFinalValidationRerun) {
    $FewshotArgs += @("--force-final-validation-rerun")
}
if ($EnableCandidateRepair) {
    $FewshotArgs += @("--enable-candidate-repair")
}
else {
    $FewshotArgs += @("--disable-candidate-repair")
}
if ($EnableRecallRepair) {
    $FewshotArgs += @("--enable-recall-repair")
}
else {
    $FewshotArgs += @("--disable-recall-repair")
}
if ($EnableSourceTools) {
    $FewshotArgs += @("--enable-source-tools")
}
if ($ForceRefreshSource) {
    $FewshotArgs += @("--force-refresh-source")
}
if ($ForceRebuildPacket) {
    $FewshotArgs += @("--force-rebuild-packet")
}
if ($EnableCandidateJudgeReuse) {
    $FewshotArgs += @("--enable-candidate-judge-reuse")
}
else {
    $FewshotArgs += @("--disable-candidate-judge-reuse")
}
if ($EnableRulePartialAcceptance) {
    $FewshotArgs += @("--enable-rule-partial-acceptance")
}
else {
    $FewshotArgs += @("--disable-rule-partial-acceptance")
}
if ($EnableRuntimeEarlyStop) {
    $FewshotArgs += @("--enable-runtime-early-stop")
}
if ($EnableIvStatefulRuntime) {
    $FewshotArgs += @("--enable-iv-stateful-runtime")
}
else {
    $FewshotArgs += @("--disable-iv-stateful-runtime")
}
if ($EnableDynamicAggregation) {
    $FewshotArgs += @("--enable-dynamic-aggregation")
}
else {
    $FewshotArgs += @("--disable-dynamic-aggregation")
}
if ($RateLimitSerialFallback) {
    $FewshotArgs += @("--rate-limit-serial-fallback")
}
else {
    $FewshotArgs += @("--disable-rate-limit-serial-fallback")
}
if ($AdaptiveEvidence) {
    $FewshotArgs += @(
        "--adaptive-evidence",
        "--adaptive-evidence-mode", $AdaptiveEvidenceMode,
        "--adaptive-evidence-max-direct-trace-nodes", "$AdaptiveEvidenceMaxDirectTraceNodes",
        "--adaptive-evidence-max-direct-trace-chars", "$AdaptiveEvidenceMaxDirectTraceChars",
        "--adaptive-evidence-max-medium-trace-nodes", "$AdaptiveEvidenceMaxMediumTraceNodes"
    )
    if ($AdaptiveEvidenceDebug) {
        $FewshotArgs += @("--adaptive-evidence-debug")
    }
}
if ($AllowGuardRepair) {
    $FewshotArgs += @("--allow-guard-repair")
}
if ($CompressRule) {
    $FewshotArgs += @("--compress-rule")
}
if ($ParallelJudge) {
    $FewshotArgs += @(
        "--parallel-judge",
        "--judge-concurrency", "$JudgeConcurrency"
    )
}
if ($StructuredJudgeLogs) {
    $FewshotArgs += @("--structured-judge-logs")
}
else {
    $FewshotArgs += @("--no-structured-judge-logs")
}
if ($SaveLlmTranscripts) {
    $FewshotArgs += @(
        "--save-llm-transcripts",
        "--llm-transcripts-dir", $FewshotTranscriptDir
    )
}

Write-Host ""
Write-Host "============================================================"
Write-Host "$Tag Access-control fewshot + eval pipeline"
Write-Host "$Tag Started at: $($PipelineStartedAt.ToString("yyyy-MM-dd HH:mm:ss"))"
Write-Host "============================================================"
Write-Host "$Tag Episode: fewshot=$Episode eval=$EvalEpisode"
if ($UseSeedArtifacts) {
    Write-Host "$Tag Initialization: synthesized seed artifacts"
    Write-Host "$Tag Seed rule: $SeedRuleFile"
    Write-Host "$Tag Seed plan: $SeedPlanFile"
}
else {
    Write-Host "$Tag Initialization: LLM cold start; no --rule-file or --plan-file"
}
Write-Host "$Tag Initial plan Agent: $GenerateInitialPlanWithAgent"
Write-Host "$Tag Fewshot pos: $FewshotPosCsv"
Write-Host "$Tag Fewshot neg: $FewshotNegCsv"
Write-Host "$Tag Guard neg: $GuardNegCsv"
Write-Host "$Tag Eval pos: $EvalPosCsv"
Write-Host "$Tag Eval neg: $EvalNegCsv"
Write-Host "$Tag Evolution mode: $EvolutionMode final_validation=$FinalValidationMode"
Write-Host "$Tag Final validation result source: $FinalValidationResultSource"
Write-Host "$Tag Force final validation rerun: $ForceFinalValidationRerun"
Write-Host "$Tag Update target: $UpdateTarget; compress_rule=$CompressRule"
Write-Host "$Tag Review compact mode: $ReviewCompactMode prompt_cap=$MaxReviewPromptChars"
Write-Host "$Tag Repair: candidate=$EnableCandidateRepair recall=$EnableRecallRepair max_rounds=$MaxRepairRounds"
if ($UseSeedArtifacts) {
    Write-Host "$Tag LLM: $LlmModel ($LlmProvider), cold-start model unused"
}
else {
    $ResolvedColdStartModel = if ([string]::IsNullOrWhiteSpace($ColdStartModel)) {
        $LlmModel
    }
    else {
        $ColdStartModel
    }
    Write-Host "$Tag LLM: $LlmModel ($LlmProvider), cold_start=$ResolvedColdStartModel"
}
Write-Host "$Tag MiniMax thinking mode: $MiniMaxThinkingMode"
Write-Host "$Tag Judge thinking: $JudgeThinking; Aggregator thinking: $AggregatorThinking"
Write-Host "$Tag Review thinking: $ReviewThinking; Update thinking: $UpdateThinking"
Write-Host "$Tag Dynamic aggregation: $EnableDynamicAggregation"
Write-Host "$Tag Near-miss safety: new_observation_required=true; exhausted_true_upgrade=false; uncertain_true_requires_target_evidence=true; max_once_per_blocker=true"
Write-Host "$Tag Follow-up context mode: $FollowupContextMode"
Write-Host "$Tag Completion tokens: judge=$JudgeMaxTokens aggregator=$AggregatorMaxTokens"
Write-Host "$Tag GLM thinking from environment: $($env:GLM_THINKING)"
Write-Host "$Tag 429 fallback: enabled=$RateLimitSerialFallback retries=$RateLimitRetryAttempts initial_delay=${RateLimitRetryDelaySeconds}s"
Write-Host "$Tag Parallel judge: $ParallelJudge concurrency=$JudgeConcurrency"
Write-Host "$Tag Candidate reuse: $EnableCandidateJudgeReuse; guard_repair=$AllowGuardRepair"
Write-Host "$Tag Rule partial acceptance: enabled=$EnableRulePartialAcceptance max_deltas=$RulePartialMaxDeltas max_evaluations=$RulePartialMaxEvaluations"
Write-Host "$Tag IV stateful runtime: $EnableIvStatefulRuntime"
Write-Host "$Tag Access Control binding mode: $AccessControlBindingMode"

if ($DryRun) {
    Write-Host ""
    Write-Host "$Tag Dry run only. Fewshot command:"
    Write-Host "python $($FewshotArgs -join ' ')"
}
else {
    Invoke-PythonStep `
        -Name "Access control fewshot episode $Episode" `
        -ArgsList $FewshotArgs
}

$LatestRuleFile = "data\rules\$Episode\access_control__latest.json"
$LatestPlanFile = "data\plans\$Episode\access_control__plan_latest.json"
if (-not (Test-Path $LatestRuleFile)) {
    $LatestRuleFile = "data\rules\$Episode\access_control_evidence_rule__latest.json"
}
if (-not (Test-Path $LatestPlanFile)) {
    $LatestPlanFile = "data\plans\$Episode\access_control_evidence_rule__plan_latest.json"
}

if (-not (Test-Path $LatestRuleFile)) {
    if ($DryRun) {
        $LatestRuleFile = "data\rules\$Episode\access_control__latest.json"
    }
    else {
        throw "Fewshot rule output not found for episode $Episode."
    }
}
if (-not (Test-Path $LatestPlanFile)) {
    if ($DryRun) {
        $LatestPlanFile = "data\plans\$Episode\access_control__plan_latest.json"
    }
    else {
        throw "Fewshot plan output not found for episode $Episode."
    }
}

# $EvalTranscriptDir = "data\results\$EvalEpisode\llm_transcripts"
# $EvalArgs = @(
#     "experiments\evaluate.py",
#     "--pos-csv", $EvalPosCsv,
#     "--neg-csv", $EvalNegCsv,
#     "--label", $Label,
#     "--positive-label", $PositiveLabel,
#     "--rule-file", $LatestRuleFile,
#     "--plan-file", $LatestPlanFile,
#     "--llm-model", $LlmModel,
#     "--llm-provider", $LlmProvider,
#     "--base-cache-dir", "data\cache",
#     "--source-cache-dir", "data\cache\contracts",
#     "--logs-dir", "data\log",
#     "--episode", "$EvalEpisode",
#     "--max-view-chars", "$MaxViewChars",
#     "--max-context-chars", "$MaxContextChars",
#     "--judge-max-tokens", "$JudgeMaxTokens",
#     "--aggregator-max-tokens", "$AggregatorMaxTokens",
#     "--judge-thinking", $JudgeThinking,
#     "--aggregator-thinking", $AggregatorThinking,
#     "--access-control-binding-mode", $AccessControlBindingMode,
#     "--results-out", "data\results",
#     "--out", "data\results",
#     "--slim-results-out", "data\results",
#     "--failed-csv-out", "data\results",
#     "--followup-context-mode", $FollowupContextMode,
#     "--rate-limit-retry-attempts", "$RateLimitRetryAttempts",
#     "--rate-limit-retry-delay-seconds", "$RateLimitRetryDelaySeconds"
# )

# if ($EnableSourceTools) {
#     $EvalArgs += @("--enable-source-tools")
# }
# if ($ForceRefreshSource) {
#     $EvalArgs += @("--force-refresh-source")
# }
# if ($ForceRebuildPacket) {
#     $EvalArgs += @("--force-rebuild-packet")
# }
# if ($EnableDynamicAggregation) {
#     $EvalArgs += @("--enable-dynamic-aggregation")
# }
# else {
#     $EvalArgs += @("--disable-dynamic-aggregation")
# }
# if ($RateLimitSerialFallback) {
#     $EvalArgs += @("--rate-limit-serial-fallback")
# }
# else {
#     $EvalArgs += @("--disable-rate-limit-serial-fallback")
# }
# if ($AdaptiveEvidence) {
#     $EvalArgs += @(
#         "--adaptive-evidence",
#         "--adaptive-evidence-mode", $AdaptiveEvidenceMode,
#         "--adaptive-evidence-max-direct-trace-nodes", "$AdaptiveEvidenceMaxDirectTraceNodes",
#         "--adaptive-evidence-max-direct-trace-chars", "$AdaptiveEvidenceMaxDirectTraceChars",
#         "--adaptive-evidence-max-medium-trace-nodes", "$AdaptiveEvidenceMaxMediumTraceNodes"
#     )
#     if ($AdaptiveEvidenceDebug) {
#         $EvalArgs += @("--adaptive-evidence-debug")
#     }
# }
# if ($ParallelJudge) {
#     $EvalArgs += @(
#         "--parallel-judge",
#         "--judge-concurrency", "$JudgeConcurrency"
#     )
# }
# if ($StructuredJudgeLogs) {
#     $EvalArgs += @("--structured-judge-logs")
# }
# else {
#     $EvalArgs += @("--no-structured-judge-logs")
# }
# if ($SaveLlmTranscripts) {
#     $EvalArgs += @(
#         "--save-llm-transcripts",
#         "--llm-transcripts-dir", $EvalTranscriptDir
#     )
# }

# if ($DryRun) {
#     Write-Host ""
#     Write-Host "$Tag Dry run only. Eval command:"
#     Write-Host "python $($EvalArgs -join ' ')"
#     if (-not [string]::IsNullOrWhiteSpace($ZhipuApiKey)) {
#         $env:ZHIPU_API_KEY = $OldZhipuApiKey
#     }
#     return
# }

# Write-Host "$Tag Evaluation rule: $LatestRuleFile"
# Write-Host "$Tag Evaluation plan: $LatestPlanFile"

# Invoke-PythonStep `
#     -Name "Access control evaluation episode $EvalEpisode" `
#     -ArgsList $EvalArgs

# $PipelineFinishedAt = Get-Date
# $PipelineElapsed = $PipelineFinishedAt - $PipelineStartedAt
# $PipelineElapsedText = "{0:00}:{1:00}:{2:00}.{3:000}" -f [int]$PipelineElapsed.TotalHours, $PipelineElapsed.Minutes, $PipelineElapsed.Seconds, $PipelineElapsed.Milliseconds

# Write-Host ""
# Write-Host "============================================================"
# Write-Host "$Tag Pipeline complete."
# Write-Host ("$Tag Completed at: {0}" -f $PipelineFinishedAt.ToString("yyyy-MM-dd HH:mm:ss"))
# Write-Host "$Tag Total elapsed: $PipelineElapsedText"
# Write-Host "$Tag Fewshot report: data\results\$Episode\final_report.json"
# Write-Host "$Tag Eval summary: data\results\$EvalEpisode\eval_summary.json"
# Write-Host "$Tag Eval slim: data\results\$EvalEpisode\eval_slim_results.json"
# Write-Host "$Tag Failed CSV: data\results\$EvalEpisode\failed.csv"
# Write-Host "============================================================"

# if (-not [string]::IsNullOrWhiteSpace($ZhipuApiKey)) {
#     $env:ZHIPU_API_KEY = $OldZhipuApiKey
# }

if (-not [string]::IsNullOrWhiteSpace($ZhipuApiKey)) {
    $env:ZHIPU_API_KEY = $OldZhipuApiKey
}
if (-not [string]::IsNullOrWhiteSpace($MiniMaxApiKey)) {
    $env:MINIMAX_API_KEY = $OldMiniMaxApiKey
}
Restore-MiniMaxThinkingOverride
Restore-GlmEnAnthropicOverride
