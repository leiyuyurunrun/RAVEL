param(
    [int]$StartEpisode = 319,
    [string]$LlmModel = "MiniMax-M3",
    [string]$MiniMaxApiKey = "",
    [int]$MaxRounds = 4,
    [ValidateSet("stage-default", "disabled", "adaptive")]
    [string]$MiniMaxThinkingMode = "stage-default",
    [ValidateSet("disabled", "adaptive")]
    [string]$ColdStartThinking = "disabled",
    [ValidateSet("disabled", "adaptive")]
    [string]$PlannerThinking = "disabled",
    [ValidateSet("disabled", "adaptive")]
    [string]$JudgeThinking = "disabled",
    [ValidateSet("disabled", "adaptive")]
    [string]$AggregatorThinking = "disabled",
    [ValidateSet("disabled", "adaptive")]
    [string]$ReviewThinking = "disabled",
    [ValidateSet("disabled", "adaptive")]
    [string]$UpdateThinking = "disabled",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Runner = (Resolve-Path (Join-Path $PSScriptRoot "..\fewshot\run_v6_insufficient_validation_cold_start_glm_v4.ps1")).Path
$V8Rule = "data\cold_start\insufficient_validation_rule_stateful_v8.json"

$Common = @{
    LlmModel = $LlmModel
    LlmProvider = "minimax"
    EnableAnthropic = $false
    MiniMaxApiKey = $MiniMaxApiKey
    MiniMaxThinkingMode = $MiniMaxThinkingMode
    ColdStartThinking = $ColdStartThinking
    PlannerThinking = $PlannerThinking
    JudgeThinking = $JudgeThinking
    AggregatorThinking = $AggregatorThinking
    ReviewThinking = $ReviewThinking
    UpdateThinking = $UpdateThinking
    EnableReviewLabelRationale = $true
    MaxRounds = $MaxRounds
    EnableDynamicAggregation = $false
    ParallelJudge = $false
    JudgeConcurrency = 1
    DryRun = $DryRun
}

$Experiments = @(
    @{
        Name = "v8 fixed Rule + agent Plan"
        Episode = $StartEpisode
        RuleFile = $V8Rule
        PlanFile = ""
        GenerateInitialPlanWithAgent = $true
    },
    @{
        Name = "agent Rule + agent Plan"
        Episode = $StartEpisode + 1
        RuleFile = ""
        PlanFile = ""
        GenerateInitialPlanWithAgent = $true
    }
)

foreach ($Experiment in $Experiments) {
    $Run = @{} + $Common
    $Run["Episode"] = $Experiment.Episode
    $Run["RuleFile"] = $Experiment.RuleFile
    $Run["PlanFile"] = $Experiment.PlanFile
    $Run["GenerateInitialPlanWithAgent"] = $Experiment.GenerateInitialPlanWithAgent

    Write-Host ""
    Write-Host (
        "[IV-MiniMax-Agent] Episode {0}: {1}" -f
        $Experiment.Episode,
        $Experiment.Name
    )
    & $Runner @Run
}

Write-Host ""
Write-Host (
    "[IV-MiniMax-Agent] Completed episodes {0}..{1}." -f
    $StartEpisode,
    ($StartEpisode + $Experiments.Count - 1)
)
