param(
    [string]$LlmModel = "glm-5.1",
    [string]$LlmProvider = "glm-en",
    [Alias("enable_anthropic")]
    [bool]$EnableAnthropic = $true,
    [ValidateSet("disabled", "adaptive")]
    [string]$JudgeThinking = "disabled",
    [ValidateSet("disabled", "adaptive")]
    [string]$AggregatorThinking = "disabled",
    [ValidateSet("disabled", "adaptive")]
    [string]$ReviewThinking = "disabled",
    [ValidateSet("disabled", "adaptive")]
    [string]$UpdateThinking = "disabled",
    [ValidateSet("auto", "reuse", "rerun")]
    [string]$FinalValidationResultSource = "auto",
    [ValidateSet(
        # "price_manipulation",
        "access_control",
        "insufficient_validation",
        "flashloans",
        "reentrancy",
        "token_semantic_exploitation",
        "protocol_accounting_exploitation",
        "market_manipulation"
    )]
    [string[]]$OnlyAttack = @(),
    [string]$Remind = "",
    [switch]$ContinueOnError,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Tag = "[V6-ALL-FEWSHOT-$($LlmProvider.ToUpperInvariant())]"
$PipelineStartedAt = Get-Date
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Set-Location $ProjectRoot

$Tasks = @(
    # [ordered]@{
    #     Attack = "price_manipulation"
    #     Name = "Price manipulation"
    #     Episode = 225
    #     Script = "run_v6_price_manipulation_cold_start_minimax_m3_ep126.ps1"
    #     SupportsEvalEpisode = $true
    #     SupportsRemind = $true
    # },
    [ordered]@{
        Attack = "access_control"
        Name = "Access control"
        Episode = 275
        Script = "run_v6_access_control_cold_start_glm.ps1"
        SupportsEvalEpisode = $true
        SupportsRemind = $false
    },
    [ordered]@{
        Attack = "insufficient_validation"
        Name = "Insufficient validation"
        Episode = 325
        Script = "run_v6_insufficient_validation_fewshot_ep120_glm_explore_v4.ps1"
        SupportsEvalEpisode = $false
        SupportsRemind = $false
    },
    [ordered]@{
        Attack = "flashloans"
        Name = "Flashloans"
        Episode = 375
        Script = "run_v6_flashloans_cold_start_fewshot_ep67_xunfei.ps1"
        SupportsEvalEpisode = $false
        SupportsRemind = $true
    },
    [ordered]@{
        Attack = "reentrancy"
        Name = "Reentrancy"
        Episode = 425
        Script = "run_v6_reentrancy_cold_start_minimax_m3_ep66.ps1"
        SupportsEvalEpisode = $true
        SupportsRemind = $true
    },
    [ordered]@{
        Attack = "token_semantic_exploitation"
        Name = "Token semantic exploitation"
        Episode = 475
        Script = "run_v6_token_semantic_exploitation_cold_start_minimax_m3_ep139.ps1"
        SupportsEvalEpisode = $true
        SupportsRemind = $true
    },
    [ordered]@{
        Attack = "protocol_accounting_exploitation"
        Name = "Protocol accounting exploitation"
        Episode = 525
        Script = "run_v6_protocol_accounting_exploitation_cold_start_minimax_m3_ep146.ps1"
        SupportsEvalEpisode = $true
        SupportsRemind = $true
    },
    [ordered]@{
        Attack = "market_manipulation"
        Name = "Market manipulation"
        Episode = 575
        Script = "run_v6_market_manipulation_cold_start_minimax_m3_ep138.ps1"
        SupportsEvalEpisode = $true
        SupportsRemind = $true
    }
)

function Format-Elapsed {
    param([TimeSpan]$Elapsed)
    return "{0:00}:{1:00}:{2:00}.{3:000}" -f `
        [int]$Elapsed.TotalHours, $Elapsed.Minutes, $Elapsed.Seconds, $Elapsed.Milliseconds
}

function New-TaskArgs {
    param([System.Collections.IDictionary]$Task)

    $Args = @{
        Episode = [int]$Task.Episode
        LlmModel = $LlmModel
        LlmProvider = $LlmProvider
        EnableAnthropic = $EnableAnthropic
        JudgeThinking = $JudgeThinking
        AggregatorThinking = $AggregatorThinking
        ReviewThinking = $ReviewThinking
        UpdateThinking = $UpdateThinking
        FinalValidationResultSource = $FinalValidationResultSource
    }
    if ([bool]$Task.SupportsEvalEpisode) {
        $Args["EvalEpisode"] = [int]$Task.Episode
    }
    if ([bool]$Task.SupportsRemind -and -not [string]::IsNullOrWhiteSpace($Remind)) {
        $Args["Remind"] = $Remind
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

Write-Host "$Tag Project root: $ProjectRoot"
Write-Host "$Tag Model/provider: $LlmModel / $LlmProvider"
Write-Host "$Tag EnableAnthropic: $EnableAnthropic"
Write-Host "$Tag Thinking: judge=$JudgeThinking aggregator=$AggregatorThinking review=$ReviewThinking update=$UpdateThinking"
Write-Host "$Tag Final validation result source: $FinalValidationResultSource"
Write-Host "$Tag Task count: $($SelectedTasks.Count)"
Write-Host ""

$Results = New-Object System.Collections.Generic.List[object]

foreach ($Task in $SelectedTasks) {
    $TaskStartedAt = Get-Date
    $ScriptPath = Join-Path (Join-Path $PSScriptRoot "..\fewshot") $Task.Script
    $TaskArgs = New-TaskArgs -Task $Task

    if (-not (Test-Path -LiteralPath $ScriptPath)) {
        throw "$Tag Missing child script for $($Task.Name): $ScriptPath"
    }

    Write-Host "============================================================"
    Write-Host "$Tag Starting $($Task.Name) episode $($Task.Episode)"
    Write-Host "$Tag Script: $ScriptPath"
    Write-Host "============================================================"

    try {
        & $ScriptPath @TaskArgs
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
