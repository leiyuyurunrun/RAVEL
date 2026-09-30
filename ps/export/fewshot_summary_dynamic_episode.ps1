param(
    [string[]]$Episodes = @(),
    [string]$ResultsDir = "data\results",
    [string]$LogsDir = "data\log",
    [string]$Out = "data\results\fewshot_summary.csv",
    [switch]$IncludeFinalValidationOnly
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

function Resolve-RepoPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $RepoRoot
    }
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $RepoRoot $Path)
}

function Get-PropValue {
    param(
        $Object,
        [string]$Name,
        $Default = $null
    )
    if ($null -eq $Object) {
        return $Default
    }
    $prop = $Object.PSObject.Properties[$Name]
    if ($null -eq $prop) {
        return $Default
    }
    return $prop.Value
}

function First-NonEmpty {
    param([object[]]$Values)
    foreach ($value in $Values) {
        if ($null -eq $value) {
            continue
        }
        if ($value -is [string] -and [string]::IsNullOrWhiteSpace($value)) {
            continue
        }
        return $value
    }
    return ""
}

function As-Int {
    param($Value)
    if ($null -eq $Value -or [string]$Value -eq "") {
        return 0
    }
    try {
        return [int]$Value
    }
    catch {
        return 0
    }
}

function As-BoolString {
    param($Value)
    if ($null -eq $Value) {
        return ""
    }
    return [string]([bool]$Value)
}

function Get-SummaryField {
    param(
        $Summary,
        [string]$Name
    )
    return As-Int (Get-PropValue $Summary $Name 0)
}

function Get-ReportSummary {
    param($Report)
    $summary = Get-PropValue (Get-PropValue $Report "evolution") "final_summary"
    if ($null -eq $summary) {
        $summary = Get-PropValue $Report "final_summary"
    }
    if ($null -eq $summary) {
        $summary = Get-PropValue $Report "summary"
    }
    return $summary
}

function Get-ArtifactPath {
    param(
        $Object,
        [string]$Name
    )
    return [string](Get-PropValue $Object $Name "")
}

function Get-LatestStageLog {
    param(
        [int]$Episode,
        [string]$Prefix,
        [string]$LogsRoot
    )
    $episodeLogDir = Join-Path $LogsRoot ([string]$Episode)
    if (-not (Test-Path -LiteralPath $episodeLogDir)) {
        return $null
    }
    return Get-ChildItem -LiteralPath $episodeLogDir -File -Filter "${Prefix}*.log" |
        Sort-Object -Property @(
            @{ Expression = {
                if ($_.BaseName -match '__(\d{8}_\d{6})$') {
                    try {
                        return [datetime]::ParseExact(
                            $Matches[1],
                            "yyyyMMdd_HHmmss",
                            [System.Globalization.CultureInfo]::InvariantCulture
                        )
                    }
                    catch {
                        return $_.LastWriteTime
                    }
                }
                return $_.LastWriteTime
            }; Descending = $true },
            @{ Expression = { $_.LastWriteTime }; Descending = $true }
        ) |
        Select-Object -First 1
}

function Get-LatestModelsLine {
    param(
        [System.IO.FileInfo]$LogFile,
        [string]$Stage
    )
    if ($null -eq $LogFile) {
        return ""
    }
    $escapedStage = [regex]::Escape($Stage)
    $match = Select-String -LiteralPath $LogFile.FullName -Pattern "^\[$escapedStage\] Models:" |
        Select-Object -Last 1
    if ($null -eq $match) {
        return ""
    }
    return [string]$match.Line
}

function Get-ModelSummary {
    param(
        [string]$ModelsLine,
        [string[]]$Keys,
        $FallbackModels = $null
    )
    $pairs = @()
    foreach ($key in $Keys) {
        $value = ""
        if (-not [string]::IsNullOrWhiteSpace($ModelsLine)) {
            $match = [regex]::Match(
                $ModelsLine,
                "(?:^|\s)$([regex]::Escape($key))=(?<value>\S+)"
            )
            if ($match.Success) {
                $value = $match.Groups["value"].Value
            }
        }
        if ([string]::IsNullOrWhiteSpace($value) -and $null -ne $FallbackModels) {
            $propertyName = if ($key -eq "llm") { "llm_model" } else { "${key}_model" }
            $value = [string](Get-PropValue $FallbackModels $propertyName "")
        }
        if (-not [string]::IsNullOrWhiteSpace($value) -and $value -ne "unset") {
            $pairs += [PSCustomObject]@{ Key = $key; Value = $value }
        }
    }
    $uniqueModels = @($pairs | ForEach-Object { $_.Value } | Sort-Object -Unique)
    if ($uniqueModels.Count -eq 1) {
        return [string]$uniqueModels[0]
    }
    if ($pairs.Count -gt 0) {
        return ($pairs | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ";"
    }
    return ""
}

function Get-ColdStartStatus {
    param($InitialRule)
    $rulePath = [string](Get-PropValue $InitialRule "rule_path" "")
    if ([string]::IsNullOrWhiteSpace($rulePath)) {
        return ""
    }
    $resolvedRulePath = Resolve-RepoPath $rulePath
    if (-not (Test-Path -LiteralPath $resolvedRulePath)) {
        return ""
    }
    try {
        $rule = Get-Content -LiteralPath $resolvedRulePath -Raw -Encoding UTF8 |
            ConvertFrom-Json
        $source = [string](Get-PropValue (Get-PropValue $rule "metadata") "source" "")
        if ($source -match '^cold_start(?:_|$)') {
            return "True"
        }
        if (-not [string]::IsNullOrWhiteSpace($source)) {
            return "False"
        }
    }
    catch {
        return ""
    }
    return ""
}

function Convert-EpisodeTokens {
    param([string[]]$Tokens)
    $numbers = @()
    foreach ($token in $Tokens) {
        foreach ($part in ([string]$token -split ",")) {
            $text = $part.Trim()
            if ([string]::IsNullOrWhiteSpace($text)) {
                continue
            }
            if ($text -match '^(\d+)\.\.(\d+)$') {
                $start = [int]$Matches[1]
                $end = [int]$Matches[2]
                if ($start -le $end) {
                    $numbers += $start..$end
                }
                else {
                    $numbers += $start..$end
                }
                continue
            }
            if ($text -match '^\d+$') {
                $numbers += [int]$text
                continue
            }
            Write-Warning "Ignoring invalid episode token: $text"
        }
    }
    return @($numbers | Sort-Object -Unique)
}

function Get-EpisodeCsvSuffix {
    param([int[]]$EpisodeNumbers)
    $episodes = @($EpisodeNumbers | Sort-Object -Unique)
    if (-not $episodes -or $episodes.Count -eq 0) {
        return "episodes_none"
    }
    if ($episodes.Count -eq 1) {
        return "episode_$($episodes[0])"
    }

    $isContiguous = $true
    for ($i = 1; $i -lt $episodes.Count; $i++) {
        if ($episodes[$i] -ne ($episodes[$i - 1] + 1)) {
            $isContiguous = $false
            break
        }
    }
    if ($isContiguous) {
        return "episodes_$($episodes[0])-$($episodes[-1])"
    }

    $joined = ($episodes | ForEach-Object { [string]$_ }) -join "_"
    if ($joined.Length -le 80) {
        return "episodes_$joined"
    }
    return "episodes_$($episodes[0])-$($episodes[-1])_n$($episodes.Count)"
}

$resultsRoot = Resolve-RepoPath $ResultsDir
if (-not (Test-Path -LiteralPath $resultsRoot)) {
    throw "ResultsDir does not exist: $resultsRoot"
}
$logsRoot = Resolve-RepoPath $LogsDir
$outExplicitlyProvided = $PSBoundParameters.ContainsKey("Out")

$episodeNumbers = Convert-EpisodeTokens $Episodes

if (-not $episodeNumbers -or $episodeNumbers.Count -eq 0) {
    $episodeNumbers = Get-ChildItem -LiteralPath $resultsRoot -Directory |
        Where-Object {
            $_.Name -match '^\d+$' -and
            (Test-Path -LiteralPath (Join-Path $_.FullName "final_report.json"))
        } |
        ForEach-Object { [int]$_.Name } |
        Sort-Object
}

if (-not $outExplicitlyProvided) {
    $episodeSuffix = Get-EpisodeCsvSuffix -EpisodeNumbers $episodeNumbers
    $Out = "data\results\fewshot_summary_${episodeSuffix}.csv"
}

Write-Host "[FewShotSummary] Repo root: $RepoRoot"
Write-Host "[FewShotSummary] Results dir: $resultsRoot"
Write-Host "[FewShotSummary] Episodes: $($episodeNumbers -join ', ')"
Write-Host "[FewShotSummary] Output: $Out"

$rows = @()

foreach ($episode in $episodeNumbers) {
    $episodeDir = Join-Path $resultsRoot ([string]$episode)
    $reportPath = Join-Path $episodeDir "final_report.json"
    if (-not (Test-Path -LiteralPath $reportPath)) {
        Write-Warning "Missing final_report.json for episode ${episode}: $reportPath"
        continue
    }

    try {
        $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        Write-Warning "Failed to parse final_report.json for episode ${episode}: $($_.Exception.Message)"
        continue
    }

    $task = Get-PropValue $report "task"
    $inputs = Get-PropValue $report "inputs"
    $caseSources = Get-PropValue $inputs "case_sources"
    $models = Get-PropValue $inputs "models"
    $detector = Get-PropValue $report "detector_context"
    $finalRule = Get-PropValue $detector "final_rule"
    $initialRule = Get-PropValue $detector "initial_rule"
    $finalPlan = Get-PropValue $detector "final_plan"
    $initialPlan = Get-PropValue $detector "initial_plan"
    $evolution = Get-PropValue $report "evolution"
    $evolutionPolicy = Get-PropValue $report "evolution_policy"
    $finalValidation = Get-PropValue $report "final_validation"
    $hardGuard = Get-PropValue $report "hard_negative_guard"
    $reviewerPolicy = Get-PropValue $report "reviewer_policy"
    $statefulPolicy = Get-PropValue $report "stateful_runtime_policy"
    $candidatePolicy = Get-PropValue $report "candidate_policy"
    $summary = Get-ReportSummary $report
    $fvSummary = Get-PropValue $finalValidation "final_summary"
    $fvGuardSummary = Get-PropValue $finalValidation "final_guard_summary"
    $latestLog = Get-LatestStageLog -Episode $episode -Prefix "few_shot__" -LogsRoot $logsRoot
    $modelsLine = Get-LatestModelsLine -LogFile $latestLog -Stage "FewShot"

    if ($IncludeFinalValidationOnly -and -not [bool](Get-PropValue $finalValidation "enabled" $false)) {
        continue
    }

    $rows += [PSCustomObject]@{
        episode = $episode
        attack_label = First-NonEmpty @(
            (Get-PropValue $task "attack_label"),
            (Get-PropValue $finalRule "attack_label"),
            (Get-PropValue $initialRule "attack_label")
        )
        raw_attack_label = First-NonEmpty @(
            (Get-PropValue $task "raw_attack_label"),
            (Get-PropValue $finalRule "raw_attack_label"),
            (Get-PropValue $initialRule "raw_attack_label")
        )
        pos_csv = First-NonEmpty @(
            (Get-PropValue $caseSources "pos_csv"),
            (Get-PropValue $caseSources "malicious_csv")
        )
        neg_csv = First-NonEmpty @(
            (Get-PropValue $caseSources "neg_csv"),
            (Get-PropValue $caseSources "benign_csv")
        )
        hard_neg_csv = First-NonEmpty @(
            (Get-PropValue $hardGuard "csv"),
            (Get-PropValue $caseSources "hard_neg_csv")
        )
        llm_model = First-NonEmpty @(
            (Get-PropValue $models "llm_model"),
            (Get-PropValue $models "judge_model"),
            (Get-PropValue $models "review_model")
        )
        llm_provider = Get-PropValue $models "llm_provider" ""
        model = Get-ModelSummary -ModelsLine $modelsLine -Keys @(
            "llm", "cold_start", "planner", "judge", "env", "review", "update"
        ) -FallbackModels $models
        round = @(
            Get-ChildItem -LiteralPath $episodeDir -Directory -Filter "round_*"
        ).Count
        cold_start = Get-ColdStartStatus $initialRule
        rule_source = First-NonEmpty @(
            (Get-ArtifactPath $initialRule "rule_path"),
            (Get-ArtifactPath $finalRule "rule_path")
        )
        plan_source = First-NonEmpty @(
            (Get-ArtifactPath $initialPlan "plan_path"),
            (Get-ArtifactPath $finalPlan "plan_path")
        )
        evolution_mode = First-NonEmpty @(
            (Get-PropValue $evolutionPolicy "evolution_mode"),
            (Get-PropValue $finalValidation "evolution_mode"),
            (Get-PropValue $candidatePolicy "evolution_mode")
        )
        final_validation_mode = First-NonEmpty @(
            (Get-PropValue $evolutionPolicy "final_validation_mode"),
            (Get-PropValue $finalValidation "mode"),
            (Get-PropValue $candidatePolicy "final_validation_mode")
        )
        accepted_rounds = As-Int (Get-PropValue $evolution "accepted_rounds" 0)
        stop_reason = Get-PropValue $evolution "stop_reason" ""
        total = Get-SummaryField $summary "total"
        correct = Get-SummaryField $summary "correct"
        errors = Get-SummaryField $summary "errors"
        fp = Get-SummaryField $summary "fp"
        fn = Get-SummaryField $summary "fn"
        uncertain = Get-SummaryField $summary "uncertain"
        final_validation_enabled = As-BoolString (Get-PropValue $finalValidation "enabled")
        final_validation_passed = As-BoolString (Get-PropValue $finalValidation "passed")
        final_validation_rerun = As-BoolString (Get-PropValue $finalValidation "rerun")
        final_validation_pass_reason = Get-PropValue $finalValidation "pass_reason" ""
        final_validation_reject_reason = Get-PropValue $finalValidation "reject_reason" ""
        fv_total = Get-SummaryField $fvSummary "total"
        fv_correct = Get-SummaryField $fvSummary "correct"
        fv_errors = Get-SummaryField $fvSummary "errors"
        fv_fp = Get-SummaryField $fvSummary "fp"
        fv_fn = Get-SummaryField $fvSummary "fn"
        fv_uncertain = Get-SummaryField $fvSummary "uncertain"
        guard_total = Get-SummaryField $fvGuardSummary "total"
        guard_errors = Get-SummaryField $fvGuardSummary "errors"
        guard_fp = Get-SummaryField $fvGuardSummary "fp"
        guard_fn = Get-SummaryField $fvGuardSummary "fn"
        guard_uncertain = Get-SummaryField $fvGuardSummary "uncertain"
        final_rule_path = First-NonEmpty @(
            (Get-ArtifactPath $finalRule "rule_path"),
            (Get-ArtifactPath $initialRule "rule_path")
        )
        final_plan_path = First-NonEmpty @(
            (Get-ArtifactPath $finalPlan "plan_path"),
            (Get-ArtifactPath $initialPlan "plan_path")
        )
        review_compact_mode = Get-PropValue $reviewerPolicy "review_compact_mode" ""
        iv_stateful_runtime = As-BoolString (Get-PropValue $statefulPolicy "iv_stateful_runtime_enabled_by_cli")
        report_path = "data/results/$episode/final_report.json"
    }
}

$outPath = Resolve-RepoPath $Out
$outDir = Split-Path $outPath -Parent
if (-not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

$rows |
    Sort-Object episode |
    Export-Csv -LiteralPath $outPath -NoTypeInformation -Encoding UTF8

Write-Host "Exported $($rows.Count) few-shot summaries to $outPath"
