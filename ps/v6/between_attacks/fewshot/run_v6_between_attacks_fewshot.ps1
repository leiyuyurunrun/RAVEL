param(
    [Parameter(Mandatory = $true)]
    [int]$Episode,

    [ValidateSet(
        "price_manipulation",
        "access_control",
        "insufficient_validation",
        "flashloans",
        "reentrancy",
        "token_semantic_exploitation",
        "protocol_accounting_exploitation",
        "market_manipulation"
    )]
    [string]$AttackLabel = "access_control",

    [string]$LlmModel = "glm-5.2",
    [string]$ColdStartModel = "",
    [string]$PlannerModel = "",
    [string]$JudgeModel = "",
    [string]$EnvModel = "",
    [string]$ReviewModel = "",
    [string]$UpdateModel = "",
    [string]$LlmProvider = "glm-en",
    [string]$AdaptiveModel = "",
    [string]$AdaptiveProvider = "",
    [Alias("enable_anthropic")]
    [bool]$EnableAnthropic = $true,
    [string]$ZhipuApiKey = "",
    [string]$MiniMaxApiKey = "",

    [ValidateSet("stage-default", "disabled", "adaptive")]
    [string]$MiniMaxThinkingMode = "stage-default",
    [ValidateSet("default", "disabled", "adaptive")]
    [string]$ColdStartThinking = "adaptive",
    [ValidateSet("default", "disabled", "adaptive")]
    [string]$PlannerThinking = "adaptive",
    [ValidateSet("default", "disabled", "adaptive")]
    [string]$JudgeThinking = "disabled",
    [ValidateSet("default", "disabled", "adaptive")]
    [string]$AggregatorThinking = "disabled",
    [ValidateSet("default", "disabled", "adaptive")]
    [string]$ReviewThinking = "adaptive",
    [ValidateSet("default", "disabled", "adaptive")]
    [string]$UpdateThinking = "adaptive",

    [ValidateRange(1, 32768)]
    [int]$ColdStartMaxTokens = 32768,
    [ValidateRange(1, 32768)]
    [int]$PlannerMaxTokens = 32768,
    [ValidateRange(1, 32768)]
    [int]$ReviewMaxTokens = 32768,
    [ValidateRange(1, 32768)]
    [int]$UpdateMaxTokens = 16384,

    [string]$RuleFile = "",
    [string]$PlanFile = "",
    [bool]$GenerateInitialPlanWithAgent = $false,
    [int]$MaxRounds = 4,
    [ValidateSet("explore", "strict")]
    [string]$EvolutionMode = "explore",
    [ValidateSet("none", "same", "strict")]
    [string]$FinalValidationMode = "strict",
    [ValidateSet("auto", "reuse", "rerun")]
    [string]$FinalValidationResultSource = "auto",
    [ValidateSet("auto", "rule", "plan")]
    [string]$UpdateTarget = "auto",
    [ValidateSet("auto", "continue", "stop")]
    [string]$RejectedCandidatePolicy = "auto",
    [int]$RejectedCandidateRetryRounds = 2,
    [bool]$ForceFinalValidationRerun = $false,
    [ValidateSet("auto", "legacy", "error-focused")]
    [string]$ReviewCompactMode = "error-focused",
    [int]$MaxReviewPromptChars = 250000,
    [bool]$ReviewLabelRationale = $true,

    [int]$StopErrors = 0,
    [int]$MaxViewChars = 80000,
    [int]$MaxContextChars = 200000,
    [bool]$ForceRebuildPacket = $true,
    [bool]$EnableSourceTools = $true,
    [bool]$ForceRefreshSource = $false,
    [bool]$SaveLlmTranscripts = $true,
    [bool]$AdaptiveEvidence = $false,
    [ValidateSet("off", "auto", "planned", "small_trace_direct", "medium_hybrid", "direct_trace")]
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

    [bool]$EnableCandidateJudgeReuse = $true,
    [bool]$EnableCandidateCanary = $false,
    [bool]$EnablePhase3CandidatePortfolio = $false,
    [bool]$EnableExperimentalRuleHypotheses = $false,
    [bool]$EnableRulePartialAcceptance = $false,
    [ValidateRange(2, 20)]
    [int]$RulePartialMaxDeltas = 6,
    [ValidateRange(1, 50)]
    [int]$RulePartialMaxEvaluations = 10,
    [bool]$EnableRuntimeEarlyStop = $false,

    [ValidateSet("auto", "enabled", "disabled")]
    [string]$IvStatefulRuntime = "auto",
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
    [int]$MaxRepairRounds = 0,
    [bool]$EnableCandidateRepair = $false,
    [bool]$EnableRecallRepair = $false,
    [int]$RecallRepairMinFixedFp = 1,
    [int]$RecallRepairMaxNewFn = 10,
    [bool]$CompressRule = $false,

    [ValidateSet("auto", "benign_only", "attack_contrastive", "mixed")]
    [string]$NegativeTrainingMode = "attack_contrastive",
    [string]$FewshotPosCsv = "",
    [string]$FewshotNegCsv = "",
    [bool]$EnableGuard = $true,
    [string]$GuardNegCsv = "",
    [string]$EvalPosCsv = "",
    [string]$EvalNegCsv = "",
    [string]$AttackDescription = "",
    [string]$Remind = "",

    [bool]$RunEvaluation = $false,
    [int]$EvalOnlyHead = -1,
    [string]$EvalOutputPrefix = "",

    [bool]$RequireLatest = $false,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Set-Location $ProjectRoot

$AttackDefaults = @{
    "price_manipulation" = @{
        Folder = "Price manipulation"
        Prefix = "price_manipulation"
        Description = "Price manipulation: a transaction or atomic sequence distorts or consumes manipulated price, reserve, oracle, exchange-rate, collateral, liquidation, or valuation state in a value-sensitive protocol path, then extracts value, creates bad debt, triggers unfair liquidation, or causes protocol/user loss. Normal arbitrage, ordinary swaps, liquidations, and capital-backed payouts should be excluded unless protocol-sensitive pricing or valuation state is causally manipulated or consumed."
    }
    "access_control" = @{
        Folder = "Access control"
        Prefix = "access_control"
        Description = "Access control detects transactions where an unauthorized or insufficiently authorized actor reaches a sensitive protocol action such as privileged mint, burn, withdraw, upgrade, ownership/role change, parameter change, rescue, or protected value movement, and the authorization gap causally enables value movement or protocol-state compromise."
    }
    "insufficient_validation" = @{
        Folder = "Insufficient validation"
        Prefix = "insufficient_validation"
        Description = "Insufficient validation detects transactions where a protocol accepts unchecked or weakly checked user-controlled input, state, price, amount, recipient, route, proof, signature, or accounting value, and that missing validation causally enables unauthorized value extraction, accounting corruption, or protected-state mutation."
    }
    "flashloans" = @{
        Folder = "Flashloans"
        Prefix = "flashloans"
        Description = "Flashloans detects attacks where borrowed temporary liquidity materially enables the exploit path, such as capital-amplified manipulation, liquidation, voting, accounting, or protocol interaction, and where the flash-loan-funded sequence is causally tied to the harmful outcome rather than merely incidental funding."
    }
    "reentrancy" = @{
        Folder = "Reentrancy"
        Prefix = "reentrancy"
        Description = "Reentrancy detects transactions where external control flow re-enters a protocol before a necessary state transition or accounting finalization, allowing repeated consumption, stale-state use, duplicated release, or unauthorized nested effects that cause value extraction or protocol-state corruption."
    }
    "token_semantic_exploitation" = @{
        Folder = "Token semantic exploitation"
        Prefix = "token_semantic_exploitation"
        Description = "Token semantic exploitation detects transactions where abnormal token-level semantics, such as fee-on-transfer, reflection or rebase, self-transfer inflation, locked-balance exposure, transfer hooks, supply mutation, or event/balance/accounting mismatch, are exploited to create inconsistent protocol or AMM state and extract value."
    }
    "protocol_accounting_exploitation" = @{
        Folder = "Protocol accounting exploitation"
        Prefix = "protocol_accounting_exploitation"
        Description = "Protocol accounting exploitation detects transactions where reward, yield, staking, lending, claim, share, collateral, debt, vault, or strategy accounting is incorrectly updated, consumed, or checked, causing repeated claims, excessive rewards, inflated shares, undercollateralized borrowing, abnormal withdrawals, or value extraction disproportionate to the attacker's legitimate contribution."
    }
    "market_manipulation" = @{
        Folder = "Market manipulation"
        Prefix = "market_manipulation"
        Description = "Market manipulation detects transactions where on-chain market mechanisms such as AMM reserves, pool balances, oracle prices, slippage checks, skim/sync behavior, donation effects, sandwich ordering, MEV positioning, or arbitrage-like price imbalance are exploited to create abnormal trading, valuation, settlement, or liquidity outcomes."
    }
}

$Default = $AttackDefaults[$AttackLabel]
$Folder = [string]$Default.Folder
$Prefix = [string]$Default.Prefix
if ([string]::IsNullOrWhiteSpace($AttackDescription)) {
    $AttackDescription = [string]$Default.Description
}
if ([string]::IsNullOrWhiteSpace($FewshotPosCsv)) {
    $FewshotPosCsv = "data\csv\v4\$Folder\$($Prefix)_Fewshot_pos.csv"
}
if ([string]::IsNullOrWhiteSpace($FewshotNegCsv)) {
    $FewshotNegCsv = "data\csv\v4\$Folder\$($Prefix)_Fewshot_neg.csv"
}
if ($EnableGuard -and [string]::IsNullOrWhiteSpace($GuardNegCsv)) {
    $GuardNegCsv = "data\csv\v4\$Folder\$($Prefix)_guard.csv"
}
if ([string]::IsNullOrWhiteSpace($EvalPosCsv)) {
    $EvalPosCsv = "data\csv\v4\$Folder\$($Prefix)_evaluation_pos_new.csv"
}
if ([string]::IsNullOrWhiteSpace($EvalNegCsv)) {
    $EvalNegCsv = "data\csv\v4\$Folder\$($Prefix)_evaluation_neg_new.csv"
}

$Tag = "[V6-BETWEEN-FEWSHOT-$($AttackLabel.ToUpperInvariant())-E$Episode-$($LlmProvider.ToUpperInvariant())]"
$StartedAt = Get-Date

$OldGlmEnEnableAnthropic = $env:GLM_EN_ENABLE_ANTHROPIC
$OldMiniMaxThinkingOverride = $env:MINIMAX_THINKING_OVERRIDE
$OldZhipuApiKey = $env:ZHIPU_API_KEY
$OldMiniMaxApiKey = $env:MINIMAX_API_KEY

function Restore-Env {
    if ($null -eq $OldGlmEnEnableAnthropic) {
        Remove-Item Env:GLM_EN_ENABLE_ANTHROPIC -ErrorAction SilentlyContinue
    }
    else {
        $env:GLM_EN_ENABLE_ANTHROPIC = $OldGlmEnEnableAnthropic
    }
    if ($null -eq $OldMiniMaxThinkingOverride) {
        Remove-Item Env:MINIMAX_THINKING_OVERRIDE -ErrorAction SilentlyContinue
    }
    else {
        $env:MINIMAX_THINKING_OVERRIDE = $OldMiniMaxThinkingOverride
    }
    if (-not [string]::IsNullOrWhiteSpace($ZhipuApiKey)) {
        if ($null -eq $OldZhipuApiKey) {
            Remove-Item Env:ZHIPU_API_KEY -ErrorAction SilentlyContinue
        }
        else {
            $env:ZHIPU_API_KEY = $OldZhipuApiKey
        }
    }
    if (-not [string]::IsNullOrWhiteSpace($MiniMaxApiKey)) {
        if ($null -eq $OldMiniMaxApiKey) {
            Remove-Item Env:MINIMAX_API_KEY -ErrorAction SilentlyContinue
        }
        else {
            $env:MINIMAX_API_KEY = $OldMiniMaxApiKey
        }
    }
}

trap {
    $FailedAt = Get-Date
    $Elapsed = $FailedAt - $StartedAt
    $ElapsedText = "{0:00}:{1:00}:{2:00}.{3:000}" -f [int]$Elapsed.TotalHours, $Elapsed.Minutes, $Elapsed.Seconds, $Elapsed.Milliseconds
    Write-Host "$Tag Failed at: $($FailedAt.ToString("yyyy-MM-dd HH:mm:ss"))"
    Write-Host "$Tag Elapsed before failure: $ElapsedText"
    Write-Host "$Tag Error: $_"
    Restore-Env
    throw
}

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    throw "$Tag Missing virtualenv: .\.venv\Scripts\Activate.ps1"
}
. .\.venv\Scripts\Activate.ps1

if (-not [string]::IsNullOrWhiteSpace($ZhipuApiKey)) {
    $env:ZHIPU_API_KEY = $ZhipuApiKey
}
if (-not [string]::IsNullOrWhiteSpace($MiniMaxApiKey)) {
    $env:MINIMAX_API_KEY = $MiniMaxApiKey
}
if ($EnableAnthropic) {
    $env:GLM_EN_ENABLE_ANTHROPIC = "1"
}
else {
    $env:GLM_EN_ENABLE_ANTHROPIC = "0"
}
if ($MiniMaxThinkingMode -eq "stage-default") {
    Remove-Item Env:MINIMAX_THINKING_OVERRIDE -ErrorAction SilentlyContinue
}
else {
    $env:MINIMAX_THINKING_OVERRIDE = $MiniMaxThinkingMode
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

function Assert-File {
    param([string]$PathValue, [string]$Name)
    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return
    }
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Name not found: $PathValue"
    }
}

Assert-File -PathValue $FewshotPosCsv -Name "FewshotPosCsv"
Assert-File -PathValue $FewshotNegCsv -Name "FewshotNegCsv"
if ($EnableGuard) {
    Assert-File -PathValue $GuardNegCsv -Name "GuardNegCsv"
}
Assert-File -PathValue $EvalPosCsv -Name "EvalPosCsv"
Assert-File -PathValue $EvalNegCsv -Name "EvalNegCsv"
Assert-File -PathValue $RuleFile -Name "RuleFile"
Assert-File -PathValue $PlanFile -Name "PlanFile"

$EnableIvStatefulRuntime = $false
if ($IvStatefulRuntime -eq "enabled") {
    $EnableIvStatefulRuntime = $true
}
elseif ($IvStatefulRuntime -eq "auto" -and $AttackLabel -eq "insufficient_validation") {
    $EnableIvStatefulRuntime = $true
}

$TranscriptDir = "data\results\$Episode\llm_transcripts"
$ArgsList = @(
    "experiments\train_fewshot.py",
    "--pos-csv", $FewshotPosCsv,
    "--neg-csv", $FewshotNegCsv,
    "--label", $AttackLabel,
    "--positive-label", $AttackLabel,
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
    "--cold-start-max-tokens", "$ColdStartMaxTokens",
    "--planner-max-tokens", "$PlannerMaxTokens",
    "--review-max-tokens", "$ReviewMaxTokens",
    "--update-max-tokens", "$UpdateMaxTokens",
    "--judge-max-tokens", "$EffectiveJudgeMaxTokens",
    "--aggregator-max-tokens", "$AggregatorMaxTokens",
    "--cold-start-thinking", $ColdStartThinking,
    "--planner-thinking", $PlannerThinking,
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
    "--judge-followup-mode", $JudgeFollowupMode,
    "--judge-expanded-followup-views", "$JudgeExpandedFollowupViews",
    "--rate-limit-retry-attempts", "$RateLimitRetryAttempts",
    "--rate-limit-retry-delay-seconds", "$RateLimitRetryDelaySeconds",
    "--rule-partial-max-deltas", "$RulePartialMaxDeltas",
    "--rule-partial-max-evaluations", "$RulePartialMaxEvaluations",
    "--access-control-binding-mode", $AccessControlBindingMode,
    "--reentrancy-binding-mode", $ReentrancyBindingMode
)

if ($EnableGuard) {
    $ArgsList += @("--guard-neg-csv", $GuardNegCsv)
}

if (-not [string]::IsNullOrWhiteSpace($ColdStartModel)) {
    $ArgsList += @("--cold-start-model", $ColdStartModel)
}
if (-not [string]::IsNullOrWhiteSpace($PlannerModel)) {
    $ArgsList += @("--planner-model", $PlannerModel)
}
if (-not [string]::IsNullOrWhiteSpace($JudgeModel)) {
    $ArgsList += @("--judge-model", $JudgeModel)
}
if (-not [string]::IsNullOrWhiteSpace($EnvModel)) {
    $ArgsList += @("--env-model", $EnvModel)
}
if (-not [string]::IsNullOrWhiteSpace($ReviewModel)) {
    $ArgsList += @("--review-model", $ReviewModel)
}
if (-not [string]::IsNullOrWhiteSpace($UpdateModel)) {
    $ArgsList += @("--update-model", $UpdateModel)
}
if (-not [string]::IsNullOrWhiteSpace($AdaptiveModel)) {
    $ArgsList += @("--adaptive-model", $AdaptiveModel)
}
if (-not [string]::IsNullOrWhiteSpace($AdaptiveProvider)) {
    $ArgsList += @("--adaptive-provider", $AdaptiveProvider)
}
if (-not [string]::IsNullOrWhiteSpace($RuleFile)) {
    $ArgsList += @("--rule-file", $RuleFile)
}
if (-not [string]::IsNullOrWhiteSpace($PlanFile)) {
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
if ($RejectedCandidatePolicy -eq "continue") {
    $ArgsList += @("--continue-after-rejected-candidate")
}
elseif ($RejectedCandidatePolicy -eq "stop") {
    $ArgsList += @("--stop-after-rejected-candidate")
}
if ($RejectedCandidateRetryRounds -ge 0) {
    $ArgsList += @("--rejected-candidate-retry-rounds", "$RejectedCandidateRetryRounds")
}
if ($ReviewLabelRationale) {
    $ArgsList += @("--enable-review-label-rationale")
}
else {
    $ArgsList += @("--disable-review-label-rationale")
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
if ($EnableCandidateCanary) {
    $ArgsList += @("--enable-candidate-canary")
}
else {
    $ArgsList += @("--disable-candidate-canary")
}
if ($EnablePhase3CandidatePortfolio) {
    $ArgsList += @("--enable-phase3-candidate-portfolio")
}
else {
    $ArgsList += @("--disable-phase3-candidate-portfolio")
}
if ($EnableExperimentalRuleHypotheses) {
    $ArgsList += @("--enable-experimental-rule-hypotheses")
}
else {
    $ArgsList += @("--disable-experimental-rule-hypotheses")
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
if ($DisableStatefulBindings) {
    $ArgsList += @("--disable-stateful-bindings")
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
    $ArgsList += @("--save-llm-transcripts", "--llm-transcripts-dir", $TranscriptDir)
}

Write-Host ""
Write-Host "============================================================"
Write-Host "$Tag Unified between-attacks fewshot"
Write-Host "$Tag Started at: $($StartedAt.ToString("yyyy-MM-dd HH:mm:ss"))"
Write-Host "============================================================"
Write-Host "$Tag Project root: $ProjectRoot"
Write-Host "$Tag Worktree root: $ProjectRoot"
Write-Host "$Tag AttackLabel: $AttackLabel"
Write-Host "$Tag Episode: $Episode"
Write-Host "$Tag Model: $LlmModel provider=$LlmProvider glm_en_anthropic=$EnableAnthropic"
Write-Host "$Tag Adaptive route: model=$(if ($AdaptiveModel) { $AdaptiveModel } else { '<inherit>' }) provider=$(if ($AdaptiveProvider) { $AdaptiveProvider } else { '<inherit>' })"
Write-Host "$Tag Thinking: cold=$ColdStartThinking planner=$PlannerThinking judge=$JudgeThinking aggregator=$AggregatorThinking review=$ReviewThinking update=$UpdateThinking"
Write-Host "$Tag Completion tokens: cold=$ColdStartMaxTokens planner=$PlannerMaxTokens review=$ReviewMaxTokens update=$UpdateMaxTokens judge=$EffectiveJudgeMaxTokens (auto=$($JudgeMaxTokens -eq 0), effective_thinking=$EffectiveJudgeThinking) aggregator=$AggregatorMaxTokens"
Write-Host "$Tag NegativeTrainingMode: $NegativeTrainingMode"
Write-Host "$Tag Stateful: iv=$EnableIvStatefulRuntime ac=$AccessControlBindingMode reentrancy=$ReentrancyBindingMode disable_all=$DisableStatefulBindings"
Write-Host "$Tag Candidate reuse=$EnableCandidateJudgeReuse canary=$EnableCandidateCanary portfolio=$EnablePhase3CandidatePortfolio experimental=$EnableExperimentalRuleHypotheses partial_acceptance=$EnableRulePartialAcceptance repair=$EnableCandidateRepair dynamic_aggregation=$EnableDynamicAggregation"
Write-Host "$Tag Judge follow-up: mode=$JudgeFollowupMode expanded_views=$JudgeExpandedFollowupViews context=$FollowupContextMode"
Write-Host "$Tag Run evaluation after few-shot: $RunEvaluation"
Write-Host "$Tag Fewshot pos: $FewshotPosCsv"
Write-Host "$Tag Fewshot neg: $FewshotNegCsv"
Write-Host "$Tag Guard neg: $(if ($EnableGuard) { $GuardNegCsv } else { '<disabled>' })"
Write-Host "$Tag Eval pos: $EvalPosCsv"
Write-Host "$Tag Eval neg: $EvalNegCsv"
if (-not [string]::IsNullOrWhiteSpace($Remind)) {
    Write-Host "$Tag Remind: $Remind"
}

if ($DryRun) {
    Write-Host "$Tag Dry run command:"
    Write-Host "python $($ArgsList -join ' ')"
    Restore-Env
    return
}

python @ArgsList
$ExitCode = $LASTEXITCODE
if ($ExitCode -ne 0) {
    throw "$Tag fewshot failed with code $ExitCode"
}

$RuleLatest = "data\rules\$Episode\$($Prefix)__latest.json"
$PlanLatest = "data\plans\$Episode\$($Prefix)__plan_latest.json"
$FinalReport = "data\results\$Episode\final_report.json"
Write-Host "$Tag Final report: $FinalReport"
if (Test-Path -LiteralPath $RuleLatest) {
    Write-Host "$Tag Latest rule: $RuleLatest"
}
else {
    Write-Host "$Tag Latest rule not found: $RuleLatest"
    if ($RequireLatest) {
        throw "$Tag RequireLatest enabled but latest rule was not written."
    }
}
if (Test-Path -LiteralPath $PlanLatest) {
    Write-Host "$Tag Latest plan: $PlanLatest"
}
else {
    Write-Host "$Tag Latest plan not found: $PlanLatest"
    if ($RequireLatest) {
        throw "$Tag RequireLatest enabled but latest plan was not written."
    }
}

if ($RunEvaluation) {
    $EvalScript = Join-Path $ProjectRoot "ps\v6\between_attacks\eval\run_v6_eval_latest.ps1"
    $EvalArgs = @{
        Episode = $Episode
        AttackLabel = $AttackLabel
        LlmModel = $LlmModel
        LlmProvider = $LlmProvider
        EnableAnthropic = $EnableAnthropic
        ZhipuApiKey = $ZhipuApiKey
        MiniMaxApiKey = $MiniMaxApiKey
        MiniMaxThinkingMode = $MiniMaxThinkingMode
        JudgeThinking = $JudgeThinking
        AggregatorThinking = $AggregatorThinking
        EvalPosCsv = $EvalPosCsv
        EvalNegCsv = $EvalNegCsv
        OnlyHead = $EvalOnlyHead
        MaxViewChars = $MaxViewChars
        MaxContextChars = $MaxContextChars
        ForceRebuildPacket = $ForceRebuildPacket
        EnableSourceTools = $EnableSourceTools
        ForceRefreshSource = $ForceRefreshSource
        SaveLlmTranscripts = $SaveLlmTranscripts
        AdaptiveEvidence = $AdaptiveEvidence
        AdaptiveEvidenceMode = $AdaptiveEvidenceMode
        AdaptiveEvidenceDebug = $AdaptiveEvidenceDebug
        AdaptiveEvidenceMaxDirectTraceNodes = $AdaptiveEvidenceMaxDirectTraceNodes
        AdaptiveEvidenceMaxDirectTraceChars = $AdaptiveEvidenceMaxDirectTraceChars
        AdaptiveEvidenceMaxMediumTraceNodes = $AdaptiveEvidenceMaxMediumTraceNodes
        ParallelJudge = $ParallelJudge
        JudgeConcurrency = $JudgeConcurrency
        JudgeMaxTokens = $EffectiveJudgeMaxTokens
        AggregatorMaxTokens = $AggregatorMaxTokens
        StructuredJudgeLogs = $StructuredJudgeLogs
        EnableIvStatefulRuntime = $EnableIvStatefulRuntime
        AccessControlBindingMode = $AccessControlBindingMode
        ReentrancyBindingMode = $ReentrancyBindingMode
        DisableStatefulBindings = $DisableStatefulBindings
        EnableDynamicAggregation = $EnableDynamicAggregation
        FollowupContextMode = $FollowupContextMode
        JudgeFollowupMode = $JudgeFollowupMode
        JudgeExpandedFollowupViews = $JudgeExpandedFollowupViews
        RateLimitSerialFallback = $RateLimitSerialFallback
        RateLimitRetryAttempts = $RateLimitRetryAttempts
        RateLimitRetryDelaySeconds = $RateLimitRetryDelaySeconds
    }
    if (Test-Path -LiteralPath $RuleLatest) {
        $EvalArgs["RuleFile"] = $RuleLatest
    }
    if (Test-Path -LiteralPath $PlanLatest) {
        $EvalArgs["PlanFile"] = $PlanLatest
    }
    if (-not [string]::IsNullOrWhiteSpace($EvalOutputPrefix)) {
        $EvalArgs["OutputPrefix"] = $EvalOutputPrefix
    }
    Write-Host "$Tag Starting evaluation: $EvalScript"
    & $EvalScript @EvalArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Tag evaluation failed with code $LASTEXITCODE"
    }
}

$FinishedAt = Get-Date
$Elapsed = $FinishedAt - $StartedAt
$ElapsedText = "{0:00}:{1:00}:{2:00}.{3:000}" -f [int]$Elapsed.TotalHours, $Elapsed.Minutes, $Elapsed.Seconds, $Elapsed.Milliseconds
Write-Host "$Tag Finished at: $($FinishedAt.ToString("yyyy-MM-dd HH:mm:ss"))"
Write-Host "$Tag Total elapsed: $ElapsedText"
Restore-Env
