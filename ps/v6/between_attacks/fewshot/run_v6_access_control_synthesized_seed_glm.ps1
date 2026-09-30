param(
    [int]$Episode = 63,
    [int]$EvalEpisode = 63,
    [string]$LlmModel = "MiniMax-M3",
    [string]$ColdStartModel = "",
    [string]$LlmProvider = "minimax",
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
    [string]$SeedRuleFile = "data\cold_start\access_control_rule_synthesized_33_60_62.json",
    [string]$SeedPlanFile = "data\cold_start\access_control_plan_synthesized_33_60_62.json",
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
    [int]$MaxReviewPromptChars = 40000,
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
    [bool]$EnableIvStatefulRuntime = $true,
    [ValidateSet("prompt", "stateful")]
    [string]$AccessControlBindingMode = "stateful",
    [bool]$EnableDynamicAggregation = $true,
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
    [string]$FewshotPosCsv = "data\csv\v2\Access control\access_control_artificial_Fewshot_pos.csv",
    [string]$FewshotNegCsv = "data\csv\v2\Access control\access_control_artificial_Fewshot_neg.csv",
    [string]$GuardNegCsv = "data\csv\v2\Access control\access_control_artificial_guard.csv",
    [string]$EvalPosCsv = "data\csv\v2\Access control\access_control_artificial_evaluation_pos.csv",
    [string]$EvalNegCsv = "data\csv\v2\Access control\access_control_artificial_evaluation_neg.csv",
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
$Runner = (Resolve-Path (Join-Path $PSScriptRoot "run_v6_access_control_cold_start_glm.ps1")).Path
if (-not (Test-Path $Runner)) {
    throw "Shared v6 access-control runner not found: $Runner"
}

$ForwardParameters = @{}
foreach ($ParameterName in $MyInvocation.MyCommand.Parameters.Keys) {
    $ForwardParameters[$ParameterName] = Get-Variable -Name $ParameterName -ValueOnly
}

& $Runner @ForwardParameters

Restore-GlmEnAnthropicOverride
