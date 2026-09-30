param(
    [int]$StartEpisode = 317,
    [string]$LlmModel = "glm-5.2",
    [string]$ZhipuApiKey = "",
    [int]$MaxRounds = 4,
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
    LlmProvider = "glm-en"
    EnableAnthropic = $true
    ZhipuApiKey = $ZhipuApiKey
    MiniMaxThinkingMode = "stage-default"
    ColdStartThinking = "disabled"
    PlannerThinking = "disabled"
    JudgeThinking = "disabled"
    AggregatorThinking = "disabled"
    ReviewThinking = "disabled"
    UpdateThinking = "disabled"
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
    },
    @{
        Name = "v8 fixed Rule + agent Plan"
        Episode = $StartEpisode + 2
        RuleFile = $V8Rule
        PlanFile = ""
        GenerateInitialPlanWithAgent = $true
    },
    @{
        Name = "agent Rule + agent Plan"
        Episode = $StartEpisode + 3
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
        "[IV-Comparison] Episode {0}: {1}" -f
        $Experiment.Episode,
        $Experiment.Name
    )
    & $Runner @Run
}

Write-Host ""
Write-Host (
    "[IV-Comparison] Completed episodes {0}..{1} with glm-en and all stage thinking disabled." -f
    $StartEpisode,
    ($StartEpisode + $Experiments.Count - 1)
)
