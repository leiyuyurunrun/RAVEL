param(
    [int]$Episode = 67,
    [string]$LlmModel = "glm-5.1",
    [string]$LlmProvider = "glm",
    [Alias("enable_anthropic")]
    [bool]$EnableAnthropic = $false,
    [string]$XunfeiApiKey = "",
    [ValidateSet("disabled", "adaptive")]
    [string]$JudgeThinking = "disabled",
    [ValidateSet("disabled", "adaptive")]
    [string]$AggregatorThinking = "disabled",
    [ValidateSet("disabled", "adaptive")]
    [string]$ReviewThinking = "adaptive",
    [ValidateSet("disabled", "adaptive")]
    [string]$UpdateThinking = "adaptive",
    [string]$RuleFile = "",
    [string]$PlanFile = "",
    [bool]$GenerateInitialPlanWithAgent = $false,
    [int]$MaxRounds = 4,
    [string]$EvolutionMode = "explore",
    [string]$FinalValidationMode = "strict",
    [ValidateSet("auto", "reuse", "rerun")]
    [string]$FinalValidationResultSource = "auto",
    [ValidateSet("auto", "rule", "plan")]
    [string]$UpdateTarget = "auto",
    [bool]$ForceFinalValidationRerun = $false,
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
    [bool]$EnableDynamicAggregation = $false,
    [string]$FollowupContextMode = "unified",
    [bool]$RateLimitSerialFallback = $true,
    [int]$RateLimitRetryAttempts = 3,
    [double]$RateLimitRetryDelaySeconds = 2.0,
    [ValidateRange(0, 1000000)]
    [int]$MaxFpIncrease = 1,
    [ValidateRange(0, 1000000)]
    [int]$MaxFnIncrease = 1,
    [int]$MaxUncertainIncrease = 2,
    [int]$MaxUncertainBenignIncrease = 1,
    [int]$MaxGuardFpIncrease = 1,
    [int]$MaxGuardErrorIncrease = 1,
    [int]$MaxGuardUncertainIncrease = 1,
    [int]$TemporaryFpIncrease = 3,
    [int]$RepairMinFnDecrease = 1,
    [int]$MaxRepairRounds = 2,
    [bool]$EnableCandidateRepair = $true,
    [bool]$EnableRecallRepair = $true,
    [int]$RecallRepairMinFixedFp = 1,
    [int]$RecallRepairMaxNewFn = 10,
    [bool]$CompressRule = $false,
    [string]$NegativeTrainingMode = "attack_contrastive",
    [string]$FewshotPosCsv = "data\csv\v4\Flashloans\flashloans_Fewshot_pos.csv",
    [string]$FewshotNegCsv = "data\csv\v4\Flashloans\flashloans_Fewshot_neg.csv",
    [string]$GuardNegCsv = "data\csv\v4\Flashloans\flashloans_guard.csv",
    [string]$Remind = "",
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
$Tag = "[FL-V6-E$Episode-$($LlmProvider.ToUpperInvariant())]"
$TaskStartedAt = Get-Date
$OldXunfeiApiKey = $env:XUNFEI_API_KEY
$ReminderFile = ""
$ExperimentRunId = $TaskStartedAt.ToString("yyyyMMdd_HHmmss_fff")

function Append-ExperimentReminderText {
    param(
        [AllowEmptyString()]
        [string[]]$Lines = @()
    )
    if ([string]::IsNullOrWhiteSpace($ReminderFile)) {
        return
    }
    try {
        $Text = ($Lines -join [Environment]::NewLine) + `
            ([Environment]::NewLine * 2)
        [System.IO.File]::AppendAllText(
            $ReminderFile,
            $Text,
            [System.Text.UTF8Encoding]::new($false)
        )
    }
    catch {
        Write-Warning "$Tag Failed to append experiment reminder: $_"
    }
}

function Append-ExperimentReminderStatus {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Status,
        [Parameter(Mandatory = $true)]
        [datetime]$Timestamp,
        [string]$Detail = ""
    )
    $StatusLines = @(
        "### Status: $Status",
        "- run_id: $ExperimentRunId",
        "- timestamp: $($Timestamp.ToString('yyyy-MM-dd HH:mm:ss'))"
    )
    if (-not [string]::IsNullOrWhiteSpace($Detail)) {
        $StatusLines += "- detail: $Detail"
    }
    Append-ExperimentReminderText -Lines $StatusLines
}

trap {
    $FailedAt = Get-Date
    $Elapsed = $FailedAt - $TaskStartedAt
    $ElapsedText = "{0:00}:{1:00}:{2:00}.{3:000}" -f `
        [int]$Elapsed.TotalHours, $Elapsed.Minutes, $Elapsed.Seconds, $Elapsed.Milliseconds
    Write-Host ("$Tag Failed at: {0}" -f $FailedAt.ToString("yyyy-MM-dd HH:mm:ss"))
    Write-Host "$Tag Elapsed before failure: $ElapsedText"
    Write-Host "$Tag Error: $_"
    Append-ExperimentReminderStatus `
        -Status "failed" `
        -Timestamp $FailedAt `
        -Detail "elapsed=$ElapsedText; error=$($_.Exception.Message)"
    if (-not [string]::IsNullOrWhiteSpace($XunfeiApiKey)) {
        $env:XUNFEI_API_KEY = $OldXunfeiApiKey
    }
    Restore-GlmEnAnthropicOverride
    throw
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Set-Location $ProjectRoot
Set-GlmEnAnthropicOverride
Write-Host "$Tag glm-en Anthropic transport: $EnableAnthropic"
Write-Host "$Tag Project root: $ProjectRoot"

if (-not $DryRun -and -not [string]::IsNullOrWhiteSpace($Remind)) {
    $ReminderDirectory = Join-Path $ProjectRoot "data\results"
    New-Item -ItemType Directory -Path $ReminderDirectory -Force | Out-Null
    $ReminderFile = Join-Path $ReminderDirectory "experiment_reminders.md"
    $PlanMode = if ($GenerateInitialPlanWithAgent) { "agent" } else { "baseline" }
    $GitCommit = (& git rev-parse --short HEAD 2>$null | Select-Object -First 1)
    if ([string]::IsNullOrWhiteSpace($GitCommit)) {
        $GitCommit = "unavailable"
    }
    $GitDirtyCount = @(& git status --short 2>$null).Count
    Append-ExperimentReminderText -Lines @(
        "## Experiment $ExperimentRunId",
        "- status: started",
        "- started_at: $($TaskStartedAt.ToString('yyyy-MM-dd HH:mm:ss'))",
        "- episode: $Episode",
        "- model: $LlmModel",
        "- provider: $LlmProvider",
        "- initial_plan_mode: $PlanMode",
        "- git_commit: $GitCommit",
        "- git_dirty_file_count: $GitDirtyCount",
        "- evolution_mode: $EvolutionMode",
        "- final_validation_mode: $FinalValidationMode",
        "- negative_training_mode: $NegativeTrainingMode",
        "",
        "**Reminder**",
        "",
        $Remind.Trim()
    )
    Write-Host "$Tag Experiment reminder appended: $ReminderFile"
}

if (-not (Test-Path -LiteralPath ".\.venv\Scripts\Activate.ps1")) {
    throw "Virtual environment activation script not found: .\.venv\Scripts\Activate.ps1"
}

Write-Host "$Tag Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

if (-not [string]::IsNullOrWhiteSpace($XunfeiApiKey)) {
    $env:XUNFEI_API_KEY = $XunfeiApiKey
    Write-Host "$Tag Using XUNFEI_API_KEY from script parameter."
}

$Label = "flashloans"
$PositiveLabel = "Flashloans"
$AttackDescription = "Flash-loan exploitation: a transaction uses uncollateralized same-transaction borrowing as a necessary causal enabler for an exploit mechanism, such as bypassing protocol assumptions, amplifying a vulnerable accounting or validation path, manipulating state or price-dependent logic, or obtaining temporary voting, balance, or liquidity power. Merely borrowing and repaying a flash loan, arbitrage, liquidation, or ordinary leveraged execution is non-target unless the flash loan is causally necessary to the exploit outcome."

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

foreach ($item in @(
    @($FewshotPosCsv, "FewshotPosCsv"),
    @($FewshotNegCsv, "FewshotNegCsv"),
    @($GuardNegCsv, "GuardNegCsv")
)) {
    Assert-ConfiguredFile -PathValue $item[0] -Name $item[1]
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

if (-not [string]::IsNullOrWhiteSpace($RuleFile)) {
    if (-not (Test-Path -LiteralPath $RuleFile)) {
        throw "Rule file not found: $RuleFile"
    }
    $ArgsList += @("--rule-file", $RuleFile)
}
if (-not [string]::IsNullOrWhiteSpace($PlanFile)) {
    if (-not (Test-Path -LiteralPath $PlanFile)) {
        throw "Plan file not found: $PlanFile"
    }
    $ArgsList += @("--plan-file", $PlanFile)
}
if ($MaxReviewPromptChars -gt 0) {
    $ArgsList += @("--max-review-prompt-chars", "$MaxReviewPromptChars")
}
if ($GenerateInitialPlanWithAgent) {
    $ArgsList += @("--generate-initial-plan-with-agent")
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
if ($ParallelJudge) {
    $ArgsList += @("--parallel-judge", "--judge-concurrency", "$JudgeConcurrency")
}
if ($CompressRule) {
    $ArgsList += @("--compress-rule")
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
Write-Host "$Tag Flashloans cold-start few-shot"
Write-Host "$Tag Started at: $($TaskStartedAt.ToString("yyyy-MM-dd HH:mm:ss"))"
Write-Host "============================================================"
Write-Host "$Tag Episode: $Episode"
$PlanText = if ([string]::IsNullOrWhiteSpace($PlanFile)) { "no --plan-file" } else { "plan=$PlanFile" }
if ([string]::IsNullOrWhiteSpace($RuleFile)) {
    Write-Host "$Tag Initialization: LLM cold start; no --rule-file; $PlanText"
}
else {
    Write-Host "$Tag Initialization: existing rule=$RuleFile; $PlanText"
}
Write-Host "$Tag Initial plan Agent: $GenerateInitialPlanWithAgent"
Write-Host "$Tag Model: $LlmModel provider=$LlmProvider"
Write-Host "$Tag Judge thinking: $JudgeThinking; Aggregator thinking: $AggregatorThinking"
Write-Host "$Tag Review thinking: $ReviewThinking; Update thinking: $UpdateThinking"
Write-Host "$Tag Negative training mode: $NegativeTrainingMode"
Write-Host "$Tag Fewshot pos: $FewshotPosCsv"
Write-Host "$Tag Fewshot neg: $FewshotNegCsv"
Write-Host "$Tag Guard neg: $GuardNegCsv"
Write-Host "$Tag Evolution: mode=$EvolutionMode final_validation=$FinalValidationMode"
Write-Host "$Tag Final validation result source: $FinalValidationResultSource"
Write-Host "$Tag Force final validation rerun: $ForceFinalValidationRerun"
Write-Host "$Tag Update target: $UpdateTarget; compress_rule=$CompressRule"
Write-Host "$Tag Review compact: $ReviewCompactMode prompt_cap=$MaxReviewPromptChars"
Write-Host "$Tag Dynamic aggregation: $EnableDynamicAggregation"
Write-Host "$Tag Follow-up context mode: $FollowupContextMode"
Write-Host "$Tag Completion tokens: judge=$JudgeMaxTokens aggregator=$AggregatorMaxTokens"
Write-Host "$Tag 429 fallback: enabled=$RateLimitSerialFallback retries=$RateLimitRetryAttempts initial_delay=${RateLimitRetryDelaySeconds}s"
Write-Host "$Tag Parallel judge: $ParallelJudge concurrency=$JudgeConcurrency"
Write-Host "$Tag Candidate reuse: $EnableCandidateJudgeReuse"
Write-Host "$Tag Rule partial acceptance: enabled=$EnableRulePartialAcceptance max_deltas=$RulePartialMaxDeltas max_evaluations=$RulePartialMaxEvaluations"
if (-not [string]::IsNullOrWhiteSpace($Remind)) {
    if ($DryRun) {
        Write-Host "$Tag Reminder supplied (dry run; not saved): $Remind"
    }
    else {
        Write-Host "$Tag Reminder file: $ReminderFile"
    }
}

if ($DryRun) {
    Write-Host ""
    Write-Host "$Tag Dry run only. Command:"
    Write-Host "python $($ArgsList -join ' ')"
    if (-not [string]::IsNullOrWhiteSpace($XunfeiApiKey)) {
        $env:XUNFEI_API_KEY = $OldXunfeiApiKey
    }
    return
}

& python @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "$Tag train_fewshot failed with code $LASTEXITCODE"
}

$TaskFinishedAt = Get-Date
$TaskElapsed = $TaskFinishedAt - $TaskStartedAt
$TaskElapsedText = "{0:00}:{1:00}:{2:00}.{3:000}" -f `
    [int]$TaskElapsed.TotalHours, $TaskElapsed.Minutes, `
    $TaskElapsed.Seconds, $TaskElapsed.Milliseconds

Write-Host ""
Write-Host "============================================================"
Write-Host "$Tag Few-shot complete."
Write-Host ("$Tag Completed at: {0}" -f $TaskFinishedAt.ToString("yyyy-MM-dd HH:mm:ss"))
Write-Host "$Tag Total elapsed: $TaskElapsedText"
Write-Host "$Tag Report: data\results\$Episode\final_report.json"
Write-Host "$Tag Rule: data\rules\$Episode\flashloans__latest.json"
Write-Host "$Tag Plan: data\plans\$Episode\flashloans__plan_latest.json"
Write-Host "============================================================"

$FinalReportPath = "data\results\$Episode\final_report.json"
$ResultDetail = "few-shot completed; elapsed=$TaskElapsedText; report=$FinalReportPath"
if (Test-Path -LiteralPath $FinalReportPath) {
    try {
        $FinalReport = Get-Content -LiteralPath $FinalReportPath -Raw | ConvertFrom-Json
        $FinalSummary = $FinalReport.final_validation.final_summary
        if ($null -eq $FinalSummary) {
            $FinalSummary = $FinalReport.evolution.final_summary
        }
        if ($null -ne $FinalSummary) {
            $ResultDetail = (
                "few-shot completed; elapsed=$TaskElapsedText; " +
                "total=$($FinalSummary.total); correct=$($FinalSummary.correct); " +
                "errors=$($FinalSummary.errors); fp=$($FinalSummary.fp); " +
                "fn=$($FinalSummary.fn); uncertain=$($FinalSummary.uncertain); " +
                "report=$FinalReportPath"
            )
        }
    }
    catch {
        Write-Warning "$Tag Could not read final summary for reminder: $_"
    }
}
Append-ExperimentReminderStatus `
    -Status "completed" `
    -Timestamp $TaskFinishedAt `
    -Detail $ResultDetail

if (-not [string]::IsNullOrWhiteSpace($XunfeiApiKey)) {
    $env:XUNFEI_API_KEY = $OldXunfeiApiKey
}

Restore-GlmEnAnthropicOverride
