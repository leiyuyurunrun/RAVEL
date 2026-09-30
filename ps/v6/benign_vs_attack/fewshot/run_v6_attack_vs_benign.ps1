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
    [string]$LlmModel = "glm-5.1",
    [string]$LlmProvider = "glm-en",
    [string]$AdaptiveModel = "",
    [string]$AdaptiveProvider = "",
    [Alias("enable_anthropic")]
    [bool]$EnableAnthropic = $true,
    [string]$ZhipuApiKey = "",
    [string]$MiniMaxApiKey = "",
    [string]$XunfeiApiKey = "",
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
    [string]$ReviewThinking = "disabled",
    [ValidateSet("default", "disabled", "adaptive")]
    [string]$UpdateThinking = "disabled",
    [ValidateRange(1, 32768)]
    [int]$ColdStartMaxTokens = 32768,
    [ValidateRange(1, 32768)]
    [int]$PlannerMaxTokens = 32768,
    [ValidateRange(1, 32768)]
    [int]$ReviewMaxTokens = 32768,
    [ValidateRange(1, 32768)]
    [int]$UpdateMaxTokens = 16384,
    [ValidateSet("auto", "benign_only", "attack_contrastive", "mixed")]
    [string]$NegativeTrainingMode = "benign_only",
    [bool]$ReviewLabelRationale = $false,
    [string]$CsvRoot = "data\csv\v7",
    [string]$FewshotPosCsv = "",
    [string]$FewshotNegCsv = "",
    [bool]$EnableGuard = $true,
    [string]$GuardNegCsv = "",
    [string]$EvalPosCsv = "",
    [string]$EvalNegCsv = "",
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
    [int]$AdaptiveEvidenceMaxDirectTraceNodes = 80,
    [int]$AdaptiveEvidenceMaxDirectTraceChars = 45000,
    [int]$AdaptiveEvidenceMaxMediumTraceNodes = 250,
    [bool]$ParallelJudge = $true,
    [int]$JudgeConcurrency = 2,
    [ValidateRange(0, 32768)]
    [int]$JudgeMaxTokens = 0,
    [ValidateRange(1, 32768)]
    [int]$AggregatorMaxTokens = 8192,
    [bool]$StructuredJudgeLogs = $true,
    [bool]$EnableCandidateJudgeReuse = $true,
    [bool]$EnableRulePartialAcceptance = $false,
    [int]$RulePartialMaxDeltas = 6,
    [int]$RulePartialMaxEvaluations = 10,
    [bool]$EnableRuntimeEarlyStop = $false,
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
    [int]$MaxFpIncrease = 1,
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
    [bool]$RunEvaluation = $false,
    [int]$EvalOnlyHead = -1,
    [string]$EvalOutputPrefix = "",
    [string]$Remind = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Set-Location $ProjectRoot
$StartedAt = Get-Date
$Tag = "[V6-BENIGN-$($AttackLabel.ToUpperInvariant())-E$Episode]"

function Resolve-AttackConfig {
    param([Parameter(Mandatory = $true)][string]$Value)
    switch ($Value) {
        "access_control" {
            return [pscustomobject]@{
                Label = "access_control"
                PositiveLabel = "Access control"
                CsvDir = "Access control"
                Prefix = "access_control"
                Description = "Access control: a transaction reaches a sensitive or protected operation such as permission changes, initializer/configuration updates, callback-protected paths, privileged mint/burn/withdraw/redeem/transfer actions, or movement of protocol/victim/shared assets; the actor, caller, callback sender, beneficiary, or execution path lacks legitimate authorization, entitlement, role, signature, or caller validation; the invocation causes protected-state mutation, privilege escalation, unauthorized approval, or release/appropriation of protected assets beyond legitimate entitlement."
            }
        }
        "flashloans" {
            return [pscustomobject]@{
                Label = "flashloans"
                PositiveLabel = "Flashloans"
                CsvDir = "Flashloans"
                Prefix = "flashloans"
                Description = "Flash-loan exploitation: a transaction uses uncollateralized same-transaction borrowing as a necessary causal enabler for an exploit mechanism, such as bypassing protocol assumptions, amplifying a vulnerable accounting or validation path, manipulating state or price-dependent logic, or obtaining temporary voting, balance, or liquidity power. Merely borrowing and repaying a flash loan, arbitrage, liquidation, or ordinary leveraged execution is non-target unless the flash loan is causally necessary to the exploit outcome."
            }
        }
        "insufficient_validation" {
            return [pscustomobject]@{
                Label = "insufficient_validation"
                PositiveLabel = "Insufficient validation"
                CsvDir = "Insufficient validation"
                Prefix = "insufficient_validation"
                Description = "Insufficient validation: a value-sensitive protocol path consumes user-supplied, external, callback, return, oracle, state, or business-assumption data; the data/state is invalid, stale, inconsistent, boundary-breaking, or adversarial; missing or inadequate validation lets the invalid value reach sensitive logic and causes an incorrect outcome, invariant break, excessive payout, asset release, accounting/state harm, or user/protocol loss."
            }
        }
        "market_manipulation" {
            return [pscustomobject]@{
                Label = "market_manipulation"
                PositiveLabel = "Market manipulation"
                CsvDir = "Market manipulation"
                Prefix = "market_manipulation"
                Description = "Market manipulation detects transactions where on-chain market mechanisms such as AMM reserves, pool balances, oracle prices, slippage checks, skim/sync behavior, donation effects, sandwich ordering, MEV positioning, or arbitrage-like price imbalance are exploited to create abnormal trading, valuation, settlement, or liquidity outcomes. The core issue is manipulation or misuse of market-facing state and execution context, not abnormal token transfer semantics."
            }
        }
        "price_manipulation" {
            return [pscustomobject]@{
                Label = "price_manipulation"
                PositiveLabel = "Price manipulation"
                CsvDir = "Price manipulation"
                Prefix = "price_manipulation"
                Description = "Price manipulation: a transaction or atomic sequence distorts or consumes manipulated price, reserve, oracle, exchange-rate, collateral, liquidation, or valuation state in a value-sensitive protocol path, then extracts value, creates bad debt, triggers unfair liquidation, or causes protocol/user loss. Normal arbitrage, ordinary swaps, liquidations, and capital-backed payouts are non-target unless protocol-sensitive pricing or valuation state is causally manipulated or consumed."
            }
        }
        "protocol_accounting_exploitation" {
            return [pscustomobject]@{
                Label = "protocol_accounting_exploitation"
                PositiveLabel = "Protocol accounting exploitation"
                CsvDir = "Protocol accounting exploitation"
                Prefix = "protocol_accounting_exploitation"
                Description = "Protocol accounting exploitation detects transactions where reward, yield, staking, lending, claim, share, collateral, debt, vault, or strategy accounting is incorrectly updated, consumed, or checked, causing repeated claims, excessive rewards, inflated shares, undercollateralized borrowing, abnormal withdrawals, or value extraction disproportionate to legitimate contribution."
            }
        }
        "reentrancy" {
            return [pscustomobject]@{
                Label = "reentrancy"
                PositiveLabel = "Reentrancy"
                CsvDir = "Reentrancy"
                Prefix = "reentrancy"
                Description = "Reentrancy: a transaction exploits nested, repeated, or callback-driven execution so protocol state, accounting, balances, locks, debt, or entitlement checks are consumed or updated in the wrong order. The stale or intermediate state causally enables repeated withdrawal, double claim, excess redemption, unauthorized value release, bad debt, or an invariant violation."
            }
        }
        "token_semantic_exploitation" {
            return [pscustomobject]@{
                Label = "token_semantic_exploitation"
                PositiveLabel = "Token semantic exploitation"
                CsvDir = "Token semantic exploitation"
                Prefix = "token_semantic_exploitation"
                Description = "Token semantic exploitation detects transactions where abnormal token-level semantics, such as fee-on-transfer, reflection or rebase, self-transfer inflation, locked-balance exposure, transfer hooks, supply mutation, or event/balance/accounting mismatch, are exploited to create inconsistent protocol or AMM state and extract value."
            }
        }
    }
}

function Assert-File {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue,
        [Parameter(Mandatory = $true)][string]$Name
    )
    if ([string]::IsNullOrWhiteSpace($PathValue) -or -not (Test-Path -LiteralPath $PathValue)) {
        throw "$Name not found: $PathValue"
    }
}

function Add-BoolFlag {
    param(
        [Parameter(Mandatory = $true)][System.Collections.Generic.List[string]]$ArgsList,
        [Parameter(Mandatory = $true)][bool]$Enabled,
        [Parameter(Mandatory = $true)][string]$EnabledFlag,
        [string]$DisabledFlag = ""
    )
    if ($Enabled) {
        $ArgsList.Add($EnabledFlag)
    }
    elseif (-not [string]::IsNullOrWhiteSpace($DisabledFlag)) {
        $ArgsList.Add($DisabledFlag)
    }
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)][string[]]$ArgsList,
        [Parameter(Mandatory = $true)][string]$Name
    )
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) {
        $Python = "python"
    }
    Write-Host "$Tag Running: $Name"
    & $Python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
}

$Config = Resolve-AttackConfig -Value $AttackLabel
$EffectiveIvStatefulRuntime = (
    $EnableIvStatefulRuntime -and
    $Config.Label -eq "insufficient_validation"
)
$CsvDirPath = Join-Path $CsvRoot $Config.CsvDir
if ([string]::IsNullOrWhiteSpace($FewshotPosCsv)) {
    $FewshotPosCsv = Join-Path $CsvDirPath "$($Config.Prefix)_Fewshot_pos.csv"
}
if ([string]::IsNullOrWhiteSpace($FewshotNegCsv)) {
    $FewshotNegCsv = Join-Path $CsvDirPath "$($Config.Prefix)_Fewshot_neg.csv"
}
if ([string]::IsNullOrWhiteSpace($EvalPosCsv)) {
    $EvalPosCsv = Join-Path $CsvDirPath "$($Config.Prefix)_evaluation_pos.csv"
}
if ([string]::IsNullOrWhiteSpace($EvalNegCsv)) {
    $EvalNegCsv = Join-Path $CsvDirPath "$($Config.Prefix)_evaluation_neg.csv"
}

Assert-File -PathValue $FewshotPosCsv -Name "FewshotPosCsv"
Assert-File -PathValue $FewshotNegCsv -Name "FewshotNegCsv"
Assert-File -PathValue $EvalPosCsv -Name "EvalPosCsv"
Assert-File -PathValue $EvalNegCsv -Name "EvalNegCsv"
if ($EnableGuard -and -not [string]::IsNullOrWhiteSpace($GuardNegCsv)) {
    Assert-File -PathValue $GuardNegCsv -Name "GuardNegCsv"
}
if (-not [string]::IsNullOrWhiteSpace($RuleFile)) {
    Assert-File -PathValue $RuleFile -Name "RuleFile"
}
if (-not [string]::IsNullOrWhiteSpace($PlanFile)) {
    Assert-File -PathValue $PlanFile -Name "PlanFile"
}

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
    Restore-Environment
    throw
}

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

$FewshotArgs = [System.Collections.Generic.List[string]]::new()
@(
    "experiments\train_fewshot.py",
    "--pos-csv", $FewshotPosCsv,
    "--neg-csv", $FewshotNegCsv,
    "--negative-training-mode", $NegativeTrainingMode,
    "--label", $Config.Label,
    "--positive-label", $Config.PositiveLabel,
    "--attack-description", $Config.Description,
    "--llm-model", $LlmModel,
    "--llm-provider", $LlmProvider,
    "--max-rounds", "$MaxRounds",
    "--stop-errors", "$StopErrors",
    "--evolution-mode", $EvolutionMode,
    "--final-validation-mode", $FinalValidationMode,
    "--final-validation-result-source", $FinalValidationResultSource,
    "--update-target", $UpdateTarget,
    "--review-compact-mode", $ReviewCompactMode,
    "--max-review-prompt-chars", "$MaxReviewPromptChars",
    "--access-control-binding-mode", $AccessControlBindingMode,
    "--reentrancy-binding-mode", $ReentrancyBindingMode,
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
    "--rule-partial-max-evaluations", "$RulePartialMaxEvaluations"
) | ForEach-Object { $FewshotArgs.Add([string]$_) }

if (-not [string]::IsNullOrWhiteSpace($AdaptiveModel)) {
    $FewshotArgs.Add("--adaptive-model")
    $FewshotArgs.Add($AdaptiveModel)
}
if (-not [string]::IsNullOrWhiteSpace($AdaptiveProvider)) {
    $FewshotArgs.Add("--adaptive-provider")
    $FewshotArgs.Add($AdaptiveProvider)
}
if ($EnableGuard -and -not [string]::IsNullOrWhiteSpace($GuardNegCsv)) {
    $FewshotArgs.Add("--guard-neg-csv")
    $FewshotArgs.Add($GuardNegCsv)
}
if (-not [string]::IsNullOrWhiteSpace($RuleFile)) {
    $FewshotArgs.Add("--rule-file")
    $FewshotArgs.Add($RuleFile)
}
if (-not [string]::IsNullOrWhiteSpace($PlanFile)) {
    $FewshotArgs.Add("--plan-file")
    $FewshotArgs.Add($PlanFile)
}
if ($GenerateInitialPlanWithAgent -and [string]::IsNullOrWhiteSpace($PlanFile)) {
    $FewshotArgs.Add("--generate-initial-plan-with-agent")
}
if ($ReviewLabelRationale) {
    $FewshotArgs.Add("--enable-review-label-rationale")
}
else {
    $FewshotArgs.Add("--disable-review-label-rationale")
}
if ($ForceFinalValidationRerun) {
    $FewshotArgs.Add("--force-final-validation-rerun")
}
Add-BoolFlag $FewshotArgs $EnableCandidateRepair "--enable-candidate-repair" "--disable-candidate-repair"
Add-BoolFlag $FewshotArgs $EnableRecallRepair "--enable-recall-repair" "--disable-recall-repair"
Add-BoolFlag $FewshotArgs $EnableCandidateJudgeReuse "--enable-candidate-judge-reuse" "--disable-candidate-judge-reuse"
Add-BoolFlag $FewshotArgs $EnableRulePartialAcceptance "--enable-rule-partial-acceptance" "--disable-rule-partial-acceptance"
Add-BoolFlag $FewshotArgs $EffectiveIvStatefulRuntime "--enable-iv-stateful-runtime" "--disable-iv-stateful-runtime"
if ($DisableStatefulBindings) {
    $FewshotArgs.Add("--disable-stateful-bindings")
}
Add-BoolFlag $FewshotArgs $EnableDynamicAggregation "--enable-dynamic-aggregation" "--disable-dynamic-aggregation"
Add-BoolFlag $FewshotArgs $RateLimitSerialFallback "--rate-limit-serial-fallback" "--disable-rate-limit-serial-fallback"
Add-BoolFlag $FewshotArgs $StructuredJudgeLogs "--structured-judge-logs" "--no-structured-judge-logs"
if ($EnableRuntimeEarlyStop) {
    $FewshotArgs.Add("--enable-runtime-early-stop")
}
if ($EnableSourceTools) {
    $FewshotArgs.Add("--enable-source-tools")
}
if ($ForceRefreshSource) {
    $FewshotArgs.Add("--force-refresh-source")
}
if ($ForceRebuildPacket) {
    $FewshotArgs.Add("--force-rebuild-packet")
}
if ($AdaptiveEvidence) {
    @(
        "--adaptive-evidence",
        "--adaptive-evidence-mode", $AdaptiveEvidenceMode,
        "--adaptive-evidence-max-direct-trace-nodes", "$AdaptiveEvidenceMaxDirectTraceNodes",
        "--adaptive-evidence-max-direct-trace-chars", "$AdaptiveEvidenceMaxDirectTraceChars",
        "--adaptive-evidence-max-medium-trace-nodes", "$AdaptiveEvidenceMaxMediumTraceNodes"
    ) | ForEach-Object { $FewshotArgs.Add([string]$_) }
    if ($AdaptiveEvidenceDebug) {
        $FewshotArgs.Add("--adaptive-evidence-debug")
    }
}
if ($AllowGuardRepair) {
    $FewshotArgs.Add("--allow-guard-repair")
}
if ($CompressRule) {
    $FewshotArgs.Add("--compress-rule")
}
if ($ParallelJudge) {
    $FewshotArgs.Add("--parallel-judge")
    $FewshotArgs.Add("--judge-concurrency")
    $FewshotArgs.Add("$JudgeConcurrency")
}
if ($SaveLlmTranscripts) {
    $FewshotArgs.Add("--save-llm-transcripts")
    $FewshotArgs.Add("--llm-transcripts-dir")
    $FewshotArgs.Add("data\results\$Episode\llm_transcripts")
}

Write-Host ""
Write-Host "============================================================"
Write-Host "$Tag Attack-vs-benign few-shot + evaluation"
Write-Host "============================================================"
Write-Host "$Tag Worktree root: $ProjectRoot"
Write-Host "$Tag Model/provider: $LlmModel / $LlmProvider"
Write-Host "$Tag Adaptive route: model=$(if ($AdaptiveModel) { $AdaptiveModel } else { '<inherit>' }) provider=$(if ($AdaptiveProvider) { $AdaptiveProvider } else { '<inherit>' })"
Write-Host "$Tag Negative mode: $NegativeTrainingMode"
Write-Host "$Tag Run evaluation after few-shot: $RunEvaluation"
Write-Host "$Tag Review Cause supervision: $ReviewLabelRationale"
Write-Host "$Tag Few-shot pos/neg: $FewshotPosCsv / $FewshotNegCsv"
Write-Host "$Tag Guard: $(if (-not $EnableGuard) { '<disabled>' } elseif ([string]::IsNullOrWhiteSpace($GuardNegCsv)) { '<none>' } else { $GuardNegCsv })"
Write-Host "$Tag Eval pos/neg: $EvalPosCsv / $EvalNegCsv"
Write-Host "$Tag Thinking: cold=$ColdStartThinking planner=$PlannerThinking judge=$JudgeThinking aggregator=$AggregatorThinking review=$ReviewThinking update=$UpdateThinking"
Write-Host "$Tag Completion tokens: cold=$ColdStartMaxTokens planner=$PlannerMaxTokens review=$ReviewMaxTokens update=$UpdateMaxTokens judge=$EffectiveJudgeMaxTokens (auto=$($JudgeMaxTokens -eq 0), effective_thinking=$EffectiveJudgeThinking) aggregator=$AggregatorMaxTokens"

if (-not [string]::IsNullOrWhiteSpace($Remind)) {
    $ReminderPath = "data\experiment_reminders.md"
    $ReminderLines = @(
        "## Episode $Episode - $($Config.Label) attack-vs-benign",
        "- started_at: $($StartedAt.ToString('yyyy-MM-dd HH:mm:ss'))",
        "- model: $LlmModel",
        "- provider: $LlmProvider",
        "- negative_training_mode: $NegativeTrainingMode",
        "- judge_followup_mode: $JudgeFollowupMode",
        "- judge_expanded_followup_views: $JudgeExpandedFollowupViews",
        "- guard: $(if (-not $EnableGuard) { 'disabled' } elseif ([string]::IsNullOrWhiteSpace($GuardNegCsv)) { 'none' } else { $GuardNegCsv })",
        "- remind: $Remind",
        ""
    )
    [System.IO.File]::AppendAllLines(
        (Join-Path $ProjectRoot $ReminderPath),
        $ReminderLines,
        [System.Text.UTF8Encoding]::new($false)
    )
}

if ($DryRun) {
    Write-Host "$Tag Dry run few-shot command:"
    Write-Host "python $($FewshotArgs -join ' ')"
}
else {
    Invoke-Python -ArgsList $FewshotArgs.ToArray() -Name "$($Config.PositiveLabel) benign-only few-shot"
}

if ($RunEvaluation) {
    $EvalScript = Join-Path $ProjectRoot "ps\v6\benign_vs_attack\eval\run_v6_eval_benign_latest.ps1"
    $EvalArgs = @{
        Episode = $Episode
        AttackLabel = $Config.Label
        LlmModel = $LlmModel
        LlmProvider = $LlmProvider
        EnableAnthropic = $EnableAnthropic
        ZhipuApiKey = $ZhipuApiKey
        MiniMaxApiKey = $MiniMaxApiKey
        XunfeiApiKey = $XunfeiApiKey
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
        EnableIvStatefulRuntime = $EffectiveIvStatefulRuntime
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
    if (-not [string]::IsNullOrWhiteSpace($EvalOutputPrefix)) {
        $EvalArgs["OutputPrefix"] = $EvalOutputPrefix
    }
    if ($DryRun) {
        Write-Host "$Tag Dry run evaluation invocation:"
        Write-Host "& $EvalScript -Episode $Episode -AttackLabel $($Config.Label) -EvalPosCsv `"$EvalPosCsv`" -EvalNegCsv `"$EvalNegCsv`""
    }
    else {
        Write-Host "$Tag Starting benign evaluation: $EvalScript"
        & $EvalScript @EvalArgs
    }
}

$FinishedAt = Get-Date
$Elapsed = $FinishedAt - $StartedAt
Write-Host "$Tag Complete in $($Elapsed.ToString())."
Restore-Environment
