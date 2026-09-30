param(
    [int]$StartEpisode = 317,
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
$V7Rule = "data\cold_start\insufficient_validation_rule_stateful_v7.json"
$V7Plan = "data\cold_start\insufficient_validation_plan_stateful_v7.json"
$V8Rule = "data\cold_start\insufficient_validation_rule_stateful_v8.json"
$V8Plan = "data\cold_start\insufficient_validation_plan_stateful_v8.json"

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
        Name = "v7 fixed Rule + fixed Plan"
        Episode = $StartEpisode
        RuleFile = $V7Rule
        PlanFile = $V7Plan
        GenerateInitialPlanWithAgent = $false
    },
    @{
        Name = "v8 fixed Rule + fixed Plan"
        Episode = $StartEpisode + 1
        RuleFile = $V8Rule
        PlanFile = $V8Plan
        GenerateInitialPlanWithAgent = $false
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
        "[IV-MiniMax-Fixed] Episode {0}: {1}" -f
        $Experiment.Episode,
        $Experiment.Name
    )
    & $Runner @Run
}

Write-Host ""
Write-Host (
    "[IV-MiniMax-Fixed] Completed episodes {0}..{1}." -f
    $StartEpisode,
    ($StartEpisode + $Experiments.Count - 1)
)
