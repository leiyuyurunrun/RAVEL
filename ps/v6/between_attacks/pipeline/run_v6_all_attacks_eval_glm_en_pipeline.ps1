param(
    [string]$LlmModel = "glm-5.1",
    [string]$LlmProvider = "glm-en",
    [Alias("enable_anthropic")]
    [bool]$EnableAnthropic = $true,
    [ValidateSet("disabled", "adaptive")]
    [string]$JudgeThinking = "disabled",
    [ValidateSet("disabled", "adaptive")]
    [string]$AggregatorThinking = "disabled",
    [ValidateSet("stage-default", "disabled", "adaptive")]
    [string]$MiniMaxThinkingMode = "stage-default",
    [ValidateSet("standard", "hard")]
    [string]$EvaluationSplit = "standard",
    [string]$CsvRoot = "data\csv\v4",
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
    [string[]]$OnlyAttack = @(),
    [string]$OutputPrefix = "",
    [int]$OnlyHead = -1,
    [int]$MaxViewChars = 80000,
    [int]$MaxContextChars = 200000,
    [bool]$ForceRebuildPacket = $true,
    [bool]$EnableSourceTools = $true,
    [bool]$ForceRefreshSource = $false,
    [bool]$UseEnvironment = $false,
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
    [bool]$EnableIvStatefulRuntime = $true,
    [ValidateSet("prompt", "stateful")]
    [string]$AccessControlBindingMode = "stateful",
    [bool]$EnableDynamicAggregation = $false,
    [ValidateSet("legacy", "unified")]
    [string]$FollowupContextMode = "unified",
    [bool]$RateLimitSerialFallback = $true,
    [int]$RateLimitRetryAttempts = 3,
    [double]$RateLimitRetryDelaySeconds = 7.0,
    [switch]$ContinueOnError,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Tag = if ($EvaluationSplit -eq "hard") {
    "[V6-ALL-EVAL-HARD-$($LlmProvider.ToUpperInvariant())]"
}
else {
    "[V6-ALL-EVAL-$($LlmProvider.ToUpperInvariant())]"
}
$PipelineStartedAt = Get-Date
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Set-Location $ProjectRoot

$Tasks = @(
    [ordered]@{
        Attack = "price_manipulation"
        Name = "Price manipulation"
        Episode = 225
    },
    [ordered]@{
        Attack = "access_control"
        Name = "Access control"
        Episode = 275
    },
    [ordered]@{
        Attack = "insufficient_validation"
        Name = "Insufficient validation"
        Episode = 325
    },
    [ordered]@{
        Attack = "flashloans"
        Name = "Flashloans"
        Episode = 375
    },
    [ordered]@{
        Attack = "reentrancy"
        Name = "Reentrancy"
        Episode = 425
    },
    [ordered]@{
        Attack = "token_semantic_exploitation"
        Name = "Token semantic exploitation"
        Episode = 475
    },
    [ordered]@{
        Attack = "protocol_accounting_exploitation"
        Name = "Protocol accounting exploitation"
        Episode = 525
    },
    [ordered]@{
        Attack = "market_manipulation"
        Name = "Market manipulation"
        Episode = 575
    }
)

function Format-Elapsed {
    param([TimeSpan]$Elapsed)
    return "{0:00}:{1:00}:{2:00}.{3:000}" -f `
        [int]$Elapsed.TotalHours, $Elapsed.Minutes, $Elapsed.Seconds, $Elapsed.Milliseconds
}

function New-TaskArgs {
    param([System.Collections.IDictionary]$Task)

    $TaskOutputPrefix = $OutputPrefix.Trim()
    if ([string]::IsNullOrWhiteSpace($TaskOutputPrefix)) {
        $TaskOutputPrefix = if ($EvaluationSplit -eq "hard") {
            "eval_hard_$($Task.Attack)"
        }
        else {
            "eval_$($Task.Attack)"
        }
    }

    $Args = @{
        Episode = [int]$Task.Episode
        AttackLabel = [string]$Task.Attack
        LlmModel = $LlmModel
        LlmProvider = $LlmProvider
        EnableAnthropic = $EnableAnthropic
        JudgeThinking = $JudgeThinking
        AggregatorThinking = $AggregatorThinking
        MiniMaxThinkingMode = $MiniMaxThinkingMode
        EvaluationSplit = $EvaluationSplit
        CsvRoot = $CsvRoot
        OutputPrefix = $TaskOutputPrefix
        OnlyHead = $OnlyHead
        MaxViewChars = $MaxViewChars
        MaxContextChars = $MaxContextChars
        ForceRebuildPacket = $ForceRebuildPacket
        EnableSourceTools = $EnableSourceTools
        ForceRefreshSource = $ForceRefreshSource
        UseEnvironment = $UseEnvironment
        SaveLlmTranscripts = $SaveLlmTranscripts
        AdaptiveEvidence = $AdaptiveEvidence
        AdaptiveEvidenceMode = $AdaptiveEvidenceMode
        AdaptiveEvidenceDebug = $AdaptiveEvidenceDebug
        AdaptiveEvidenceMaxDirectTraceNodes = $AdaptiveEvidenceMaxDirectTraceNodes
        AdaptiveEvidenceMaxDirectTraceChars = $AdaptiveEvidenceMaxDirectTraceChars
        AdaptiveEvidenceMaxMediumTraceNodes = $AdaptiveEvidenceMaxMediumTraceNodes
        ParallelJudge = $ParallelJudge
        JudgeConcurrency = $JudgeConcurrency
        JudgeMaxTokens = $JudgeMaxTokens
        AggregatorMaxTokens = $AggregatorMaxTokens
        StructuredJudgeLogs = $StructuredJudgeLogs
        EnableIvStatefulRuntime = $EnableIvStatefulRuntime
        AccessControlBindingMode = $AccessControlBindingMode
        EnableDynamicAggregation = $EnableDynamicAggregation
        FollowupContextMode = $FollowupContextMode
        RateLimitSerialFallback = $RateLimitSerialFallback
        RateLimitRetryAttempts = $RateLimitRetryAttempts
        RateLimitRetryDelaySeconds = $RateLimitRetryDelaySeconds
    }
    if ($DryRun) {
        $Args["DryRun"] = $true
    }
    return $Args
}

$SelectedTasks = @($Tasks)
if ($OnlyAttack.Count -gt 0) {
    $Wanted = @{}
    foreach ($Attack in $OnlyAttack) {
        $Wanted[$Attack] = $true
    }
    $SelectedTasks = @($Tasks | Where-Object { $Wanted.ContainsKey($_.Attack) })
}

$Runner = (Resolve-Path (Join-Path $PSScriptRoot "..\eval\run_v6_eval_latest.ps1")).Path
if (-not (Test-Path -LiteralPath $Runner)) {
    throw "$Tag Missing eval runner: $Runner"
}

Write-Host "$Tag Project root: $ProjectRoot"
Write-Host "$Tag Model/provider: $LlmModel / $LlmProvider"
Write-Host "$Tag EnableAnthropic: $EnableAnthropic"
Write-Host "$Tag Thinking: judge=$JudgeThinking aggregator=$AggregatorThinking minimax_override=$MiniMaxThinkingMode"
Write-Host "$Tag Evaluation split: $EvaluationSplit"
Write-Host "$Tag CsvRoot: $CsvRoot"
Write-Host "$Tag Task count: $($SelectedTasks.Count)"
Write-Host ""

$Results = New-Object System.Collections.Generic.List[object]

foreach ($Task in $SelectedTasks) {
    $TaskStartedAt = Get-Date
    $TaskArgs = New-TaskArgs -Task $Task

    Write-Host "============================================================"
    Write-Host "$Tag Starting eval $($Task.Name) episode $($Task.Episode)"
    Write-Host "$Tag Runner: $Runner"
    Write-Host "============================================================"

    try {
        & $Runner @TaskArgs
        $FinishedAt = Get-Date
        $Results.Add([pscustomobject]@{
            Attack = $Task.Attack
            Episode = $Task.Episode
            Status = "success"
            Elapsed = Format-Elapsed ($FinishedAt - $TaskStartedAt)
            Detail = ""
        }) | Out-Null
        Write-Host "$Tag Finished $($Task.Name) episode $($Task.Episode) in $(Format-Elapsed ($FinishedAt - $TaskStartedAt))"
    }
    catch {
        $FailedAt = Get-Date
        $Detail = "$_"
        $Results.Add([pscustomobject]@{
            Attack = $Task.Attack
            Episode = $Task.Episode
            Status = "failed"
            Elapsed = Format-Elapsed ($FailedAt - $TaskStartedAt)
            Detail = $Detail
        }) | Out-Null
        Write-Host "$Tag Failed $($Task.Name) episode $($Task.Episode): $Detail"
        if (-not $ContinueOnError) {
            break
        }
    }
    Write-Host ""
}

$PipelineFinishedAt = Get-Date
Write-Host "============================================================"
Write-Host "$Tag Pipeline summary"
Write-Host "============================================================"
$Results | Format-Table -AutoSize
Write-Host "$Tag Total elapsed: $(Format-Elapsed ($PipelineFinishedAt - $PipelineStartedAt))"

$Failures = @($Results | Where-Object { $_.Status -ne "success" })
if ($Failures.Count -gt 0) {
    throw "$Tag Pipeline finished with $($Failures.Count) failed task(s)."
}
