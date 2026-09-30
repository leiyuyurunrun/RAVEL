param(
    [int]$BaseEpisode = 1050,
    [int]$EpisodeStride = 50,
    [string]$LlmModel = "glm-5.2",
    [string]$LlmProvider = "glm-en",
    [Alias("enable_anthropic")]
    [bool]$EnableAnthropic = $true,
    [string]$ZhipuApiKey = "",
    [string]$MiniMaxApiKey = "",
    [string]$XunfeiApiKey = "",
    [ValidateSet("stage-default", "disabled", "adaptive")]
    [string]$MiniMaxThinkingMode = "stage-default",
    [ValidateSet("disabled", "adaptive")]
    [string]$JudgeThinking = "adaptive",
    [ValidateSet("disabled", "adaptive")]
    [string]$AggregatorThinking = "adaptive",
    [ValidateSet("disabled", "adaptive")]
    [string]$ReviewThinking = "adaptive",
    [ValidateSet("disabled", "adaptive")]
    [string]$UpdateThinking = "adaptive",
    [ValidateSet("auto", "benign_only", "attack_contrastive", "mixed")]
    [string]$NegativeTrainingMode = "benign_only",
    [bool]$ReviewLabelRationale = $false,
    [string]$CsvRoot = "data\csv\v7",
    [bool]$UseGuard = $true,
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
    [int]$MaxViewChars = 80000,
    [int]$MaxContextChars = 200000,
    [bool]$ForceRebuildPacket = $true,
    [bool]$EnableSourceTools = $true,
    [bool]$ForceRefreshSource = $false,
    [bool]$SaveLlmTranscripts = $true,
    [bool]$ParallelJudge = $true,
    [int]$JudgeConcurrency = 2,
    [ValidateRange(1, 32768)]
    [int]$JudgeMaxTokens = 24576,
    [ValidateRange(1, 32768)]
    [int]$AggregatorMaxTokens = 8192,
    [bool]$EnableCandidateJudgeReuse = $true,
    [bool]$EnableRulePartialAcceptance = $false,
    [bool]$EnableCandidateRepair = $false,
    [bool]$EnableRecallRepair = $false,
    [bool]$EnableIvStatefulRuntime = $true,
    [ValidateSet("prompt", "stateful")]
    [string]$AccessControlBindingMode = "stateful",
    [ValidateSet("disabled", "soft", "stateful")]
    [string]$ReentrancyBindingMode = "soft",
    [bool]$DisableStatefulBindings = $false,
    [bool]$EnableDynamicAggregation = $false,
    [ValidateSet("legacy", "unified")]
    [string]$FollowupContextMode = "unified",
    [bool]$RateLimitSerialFallback = $true,
    [int]$RateLimitRetryAttempts = 3,
    [double]$RateLimitRetryDelaySeconds = 7.0,
    [bool]$RunEvaluation = $false,
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
    [string[]]$OnlyAttack = @(),
    [string]$Remind = "",
    [switch]$ContinueOnError,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $ProjectRoot
$Tag = "[V6-ALL-ATTACK-VS-BENIGN]"
$ChildScript = Join-Path $PSScriptRoot "run_v6_attack_vs_benign.ps1"

$Tasks = @(
    [ordered]@{ Attack = "access_control"; CsvDir = "Access control"; Prefix = "access_control"; Offset = 0 },
    [ordered]@{ Attack = "flashloans"; CsvDir = "Flashloans"; Prefix = "flashloans"; Offset = 1 },
    [ordered]@{ Attack = "insufficient_validation"; CsvDir = "Insufficient validation"; Prefix = "insufficient_validation"; Offset = 2 },
    [ordered]@{ Attack = "market_manipulation"; CsvDir = "Market manipulation"; Prefix = "market_manipulation"; Offset = 3 },
    [ordered]@{ Attack = "price_manipulation"; CsvDir = "Price manipulation"; Prefix = "price_manipulation"; Offset = 4 },
    [ordered]@{ Attack = "protocol_accounting_exploitation"; CsvDir = "Protocol accounting exploitation"; Prefix = "protocol_accounting_exploitation"; Offset = 5 },
    [ordered]@{ Attack = "reentrancy"; CsvDir = "Reentrancy"; Prefix = "reentrancy"; Offset = 6 },
    [ordered]@{ Attack = "token_semantic_exploitation"; CsvDir = "Token semantic exploitation"; Prefix = "token_semantic_exploitation"; Offset = 7 }
)

if ($OnlyAttack.Count -gt 0) {
    $Wanted = @{}
    foreach ($Attack in $OnlyAttack) {
        $Wanted[$Attack] = $true
    }
    $Tasks = @($Tasks | Where-Object { $Wanted.ContainsKey($_.Attack) })
}

if (-not (Test-Path -LiteralPath $ChildScript)) {
    throw "Child script not found: $ChildScript"
}

Write-Host "$Tag Project root: $ProjectRoot"
Write-Host "$Tag Episodes: base=$BaseEpisode stride=$EpisodeStride count=$($Tasks.Count)"
Write-Host "$Tag Model/provider: $LlmModel / $LlmProvider"
Write-Host "$Tag Negative mode: $NegativeTrainingMode"
Write-Host "$Tag Review Cause supervision: $ReviewLabelRationale"
Write-Host "$Tag v7 guard enabled: $UseGuard"
Write-Host "$Tag Run evaluation: $RunEvaluation"

$Results = [System.Collections.Generic.List[object]]::new()
foreach ($Task in $Tasks) {
    $Episode = $BaseEpisode + ([int]$Task.Offset * $EpisodeStride)
    $GuardNegCsv = ""
    if ($UseGuard) {
        $GuardNegCsv = Join-Path (Join-Path $CsvRoot $Task.CsvDir) "$($Task.Prefix)_guard.csv"
        if (-not (Test-Path -LiteralPath $GuardNegCsv)) {
            throw "$Tag Guard requested but not found for $($Task.Attack): $GuardNegCsv"
        }
    }
    $TaskArgs = @{
        Episode = $Episode
        AttackLabel = $Task.Attack
        LlmModel = $LlmModel
        LlmProvider = $LlmProvider
        EnableAnthropic = $EnableAnthropic
        ZhipuApiKey = $ZhipuApiKey
        MiniMaxApiKey = $MiniMaxApiKey
        XunfeiApiKey = $XunfeiApiKey
        MiniMaxThinkingMode = $MiniMaxThinkingMode
        JudgeThinking = $JudgeThinking
        AggregatorThinking = $AggregatorThinking
        ReviewThinking = $ReviewThinking
        UpdateThinking = $UpdateThinking
        NegativeTrainingMode = $NegativeTrainingMode
        ReviewLabelRationale = $ReviewLabelRationale
        CsvRoot = $CsvRoot
        GuardNegCsv = $GuardNegCsv
        GenerateInitialPlanWithAgent = $GenerateInitialPlanWithAgent
        MaxRounds = $MaxRounds
        EvolutionMode = $EvolutionMode
        FinalValidationMode = $FinalValidationMode
        FinalValidationResultSource = $FinalValidationResultSource
        UpdateTarget = $UpdateTarget
        ForceFinalValidationRerun = $ForceFinalValidationRerun
        ReviewCompactMode = $ReviewCompactMode
        MaxReviewPromptChars = $MaxReviewPromptChars
        MaxViewChars = $MaxViewChars
        MaxContextChars = $MaxContextChars
        ForceRebuildPacket = $ForceRebuildPacket
        EnableSourceTools = $EnableSourceTools
        ForceRefreshSource = $ForceRefreshSource
        SaveLlmTranscripts = $SaveLlmTranscripts
        ParallelJudge = $ParallelJudge
        JudgeConcurrency = $JudgeConcurrency
        JudgeMaxTokens = $JudgeMaxTokens
        AggregatorMaxTokens = $AggregatorMaxTokens
        EnableCandidateJudgeReuse = $EnableCandidateJudgeReuse
        EnableRulePartialAcceptance = $EnableRulePartialAcceptance
        EnableCandidateRepair = $EnableCandidateRepair
        EnableRecallRepair = $EnableRecallRepair
        EnableIvStatefulRuntime = $EnableIvStatefulRuntime
        AccessControlBindingMode = $AccessControlBindingMode
        ReentrancyBindingMode = $ReentrancyBindingMode
        DisableStatefulBindings = $DisableStatefulBindings
        EnableDynamicAggregation = $EnableDynamicAggregation
        FollowupContextMode = $FollowupContextMode
        RateLimitSerialFallback = $RateLimitSerialFallback
        RateLimitRetryAttempts = $RateLimitRetryAttempts
        RateLimitRetryDelaySeconds = $RateLimitRetryDelaySeconds
        RunEvaluation = $RunEvaluation
        Remind = $Remind
    }
    if ($DryRun) {
        $TaskArgs["DryRun"] = $true
    }

    $TaskStartedAt = Get-Date
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "$Tag Starting $($Task.Attack), episode $Episode"
    Write-Host "============================================================"
    try {
        & $ChildScript @TaskArgs
        $Results.Add([pscustomobject]@{
            Attack = $Task.Attack
            Episode = $Episode
            Status = "success"
            Elapsed = ((Get-Date) - $TaskStartedAt).ToString()
            Detail = ""
        })
    }
    catch {
        $Results.Add([pscustomobject]@{
            Attack = $Task.Attack
            Episode = $Episode
            Status = "failed"
            Elapsed = ((Get-Date) - $TaskStartedAt).ToString()
            Detail = $_.Exception.Message
        })
        Write-Error "$Tag $($Task.Attack) failed: $($_.Exception.Message)" -ErrorAction Continue
        if (-not $ContinueOnError) {
            break
        }
    }
}

Write-Host ""
Write-Host "$Tag Summary"
$Results | Format-Table -AutoSize
$Failed = @($Results | Where-Object { $_.Status -eq "failed" })
if ($Failed.Count -gt 0) {
    throw "$($Failed.Count) attack-vs-benign task(s) failed."
}
