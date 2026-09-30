param(
    [string[]]$Episodes = @(10000..12000),
    [string]$ResultsDir = "data\results",
    [string]$LogsDir = "data\log",
    [string]$RunsRoot = "D:\pretended_home\EvoTx-runs",
    [string]$Out = "data\results\fewshot_summary10000-12000.csv",
    [bool]$RecoverHistoricalArtifactInputs = $true,
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

function Convert-ToRepoRelativePath {
    param([string]$Path)
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $rootPrefix = $RepoRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if ($fullPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $fullPath.Substring($rootPrefix.Length)
    }
    return $fullPath
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

function Get-RoundCorrect {
    param(
        [string]$EpisodeResultDir,
        [int]$RoundIndex
    )
    $summaryPath = Join-Path $EpisodeResultDir ("round_{0:d2}\current_summary.json" -f $RoundIndex)
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        return ""
    }
    try {
        $roundSummary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
        return [string](Get-PropValue $roundSummary "correct" "")
    }
    catch {
        return ""
    }
}

function Resolve-EpisodeReportLocation {
    param([string]$EpisodeRootDir)
    $candidateDirs = @(
        $EpisodeRootDir,
        (Join-Path $EpisodeRootDir "evolution")
    )
    foreach ($candidateDir in $candidateDirs) {
        $candidateReport = Join-Path $candidateDir "final_report.json"
        if (Test-Path -LiteralPath $candidateReport) {
            return [PSCustomObject]@{
                ResultDir = $candidateDir
                ReportPath = $candidateReport
                Variant = if ($candidateDir -eq $EpisodeRootDir) { "" } else { Split-Path -Leaf $candidateDir }
            }
        }
    }
    return $null
}

function Get-LatestStageLog {
    param(
        [int]$Episode,
        [string]$Prefix,
        [string]$LogsRoot,
        [string]$Variant = ""
    )
    $episodeLogRoot = Join-Path $LogsRoot ([string]$Episode)
    $candidateLogDirs = @()
    if (-not [string]::IsNullOrWhiteSpace($Variant)) {
        $candidateLogDirs += Join-Path $episodeLogRoot $Variant
    }
    $candidateLogDirs += $episodeLogRoot
    foreach ($episodeLogDir in @($candidateLogDirs | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $episodeLogDir)) {
            continue
        }
        $logFile = Get-ChildItem -LiteralPath $episodeLogDir -File -Filter "${Prefix}*.log" |
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
        if ($null -ne $logFile) {
            return $logFile
        }
    }
    return $null
}

function Get-LogStartedAt {
    param([System.IO.FileInfo]$LogFile)
    if ($null -eq $LogFile) {
        return $null
    }
    # The run stamps its start time at the end of the log; the last match wins
    # in case the string also appears mid-log.
    $match = Select-String -LiteralPath $LogFile.FullName -Pattern '^(?:\[[^\]]+\]\s+)?Started at:\s*(?<ts>\S+)' |
        Select-Object -Last 1
    if ($null -ne $match) {
        try {
            return [DateTimeOffset]::Parse(
                $match.Matches[0].Groups["ts"].Value,
                [System.Globalization.CultureInfo]::InvariantCulture
            )
        }
        catch { }
    }
    # Incomplete reruns may lack the trailing stamp; the log file name encodes
    # the same start timestamp.
    if ($LogFile.BaseName -match '__(\d{8}_\d{6})$') {
        try {
            $startedLocal = [datetime]::ParseExact(
                $Matches[1],
                "yyyyMMdd_HHmmss",
                [System.Globalization.CultureInfo]::InvariantCulture
            )
            return [DateTimeOffset]::new(
                $startedLocal,
                [TimeZoneInfo]::Local.GetUtcOffset($startedLocal)
            )
        }
        catch { }
    }
    return $null
}

function Normalize-DirectoryPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return ""
    }
    try {
        return [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    }
    catch {
        return ""
    }
}

function Get-LogWorktreeRoot {
    param([System.IO.FileInfo]$LogFile)
    if ($null -eq $LogFile) {
        return ""
    }
    $match = Select-String `
        -LiteralPath $LogFile.FullName `
        -Pattern '^(?:\[[^\]]+\]\s+)?(?:Worktree|Project) root:\s*(?<path>.+?)\s*$' |
        Select-Object -Last 1
    if ($null -eq $match) {
        return ""
    }
    return Normalize-DirectoryPath $match.Matches[0].Groups["path"].Value
}

function Get-EpisodeRunName {
    param(
        [System.IO.FileInfo]$LogFile,
        [object[]]$SortedRunDirs
    )
    if (-not $SortedRunDirs -or $SortedRunDirs.Count -eq 0) {
        return ""
    }
    # Prefer the authoritative path in new logs. This remains deterministic
    # when several worktrees have nearby creation timestamps.
    $worktreeRoot = Get-LogWorktreeRoot -LogFile $LogFile
    if (-not [string]::IsNullOrWhiteSpace($worktreeRoot)) {
        foreach ($runDir in $SortedRunDirs) {
            $candidateRoot = Normalize-DirectoryPath $runDir.FullName
            if ($candidateRoot.Equals(
                $worktreeRoot,
                [System.StringComparison]::OrdinalIgnoreCase
            )) {
                return [string]$runDir.Name
            }
        }
    }
    # Historical logs have no path field. Retain creation-time inference as a
    # compatibility fallback, also used when RunsRoot does not contain the
    # recorded worktree anymore.
    $startedAt = Get-LogStartedAt -LogFile $LogFile
    if ($null -eq $startedAt) {
        return ""
    }
    # A run directory can only have produced an episode if it already existed
    # when the episode started; pick the newest such run (ties keep the later
    # entry in the creation-time sort).
    $chosen = $null
    foreach ($runDir in $SortedRunDirs) {
        if ([DateTimeOffset]$runDir.CreationTime -le $startedAt) {
            $chosen = $runDir
        }
        else {
            break
        }
    }
    if ($null -eq $chosen) {
        return ""
    }
    return [string]$chosen.Name
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

function Get-GenerateInitialPlanWithAgentStatus {
    param(
        $PlanTemplatePolicy,
        $InitialPlan
    )
    $mode = [string](Get-PropValue $PlanTemplatePolicy "initial_plan_mode" "")
    if ($mode -eq "agent") {
        return "True"
    }
    if ($mode -in @("baseline", "plan_file")) {
        return "False"
    }

    # Older reports may not contain plan_template_policy. Their initial-plan
    # artifact still records how the plan was produced.
    $source = [string](Get-PropValue $InitialPlan "source" "")
    if ($source -match '^initial_plan_agent(?:_|$)') {
        return "True"
    }
    if (-not [string]::IsNullOrWhiteSpace($source)) {
        return "False"
    }
    return ""
}

function Get-ArtifactHashMatches {
    param(
        [ValidateSet("rule", "plan")][string]$Kind,
        [string]$ArchivePath,
        [int]$Episode
    )
    if (-not $RecoverHistoricalArtifactInputs -or [string]::IsNullOrWhiteSpace($ArchivePath)) {
        return @()
    }
    $resolvedArchive = Resolve-RepoPath $ArchivePath
    if (-not (Test-Path -LiteralPath $resolvedArchive)) {
        return @()
    }
    $root = Join-Path $RepoRoot $(if ($Kind -eq "rule") { "data\rules" } else { "data\plans" })
    if (-not (Test-Path -LiteralPath $root)) {
        return @()
    }
    $versionPattern = if ($Kind -eq "rule") {
        '__v\d+\.json$'
    }
    else {
        '__plan_v\d+\.json$'
    }
    $targetHash = (Get-FileHash -LiteralPath $resolvedArchive -Algorithm SHA256).Hash
    $archiveLeaf = Split-Path -Leaf $resolvedArchive
    return @(
        Get-ChildItem -LiteralPath $root -Recurse -File -Filter "*.json" |
            Where-Object {
                $_.Name -match $versionPattern -and
                $_.Name -eq $archiveLeaf -and
                $_.FullName -ne $resolvedArchive -and
                $_.Directory.Name -ne [string]$Episode
            } |
            ForEach-Object {
                if ((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash -eq $targetHash) {
                    Convert-ToRepoRelativePath $_.FullName
                }
            } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            Sort-Object -Unique
    )
}

function Get-InitialArtifactInputInfo {
    param(
        [ValidateSet("rule", "plan")][string]$Kind,
        $Inputs,
        $Detector,
        $PlanTemplatePolicy,
        [int]$Episode
    )
    $provenance = Get-PropValue $Inputs "initial_artifact_provenance"
    $record = Get-PropValue $provenance $Kind
    $initialRefName = if ($Kind -eq "rule") { "initial_rule" } else { "initial_plan" }
    $initialRef = Get-PropValue $Detector $initialRefName
    $archiveField = if ($Kind -eq "rule") { "rule_path" } else { "plan_path" }
    $archivePath = First-NonEmpty @(
        (Get-PropValue $record "archive_path"),
        (Get-PropValue $initialRef "archive_path"),
        (Get-PropValue $initialRef $archiveField)
    )
    $inputPath = First-NonEmpty @(
        (Get-PropValue $record "cli_path"),
        (Get-PropValue $initialRef "input_path")
    )
    $mode = First-NonEmpty @(
        (Get-PropValue $record "initialization_mode"),
        (Get-PropValue $initialRef "initialization_mode")
    )
    if ([string]::IsNullOrWhiteSpace([string]$mode) -and $Kind -eq "plan") {
        $mode = [string](Get-PropValue $PlanTemplatePolicy "initial_plan_mode" "")
        $mode = @{
            baseline = "baseline_compiled_from_initial_rule"
            agent = "agent_generated"
            plan_file = "plan_file"
        }[$mode]
    }
    if ($null -ne $record -or -not [string]::IsNullOrWhiteSpace([string]$inputPath)) {
        return [PSCustomObject]@{
            InputPath = [string]$inputPath
            InitializationMode = [string]$mode
            ArchivePath = [string]$archivePath
            RecoveryStatus = "recorded"
            RecoveryCandidates = ""
        }
    }
    if (
        $Kind -eq "plan" -and
        [string]$mode -in @("baseline_compiled_from_initial_rule", "agent_generated")
    ) {
        return [PSCustomObject]@{
            InputPath = ""
            InitializationMode = [string]$mode
            ArchivePath = [string]$archivePath
            RecoveryStatus = "inferred_no_cli_plan_file"
            RecoveryCandidates = ""
        }
    }
    $matches = @(Get-ArtifactHashMatches -Kind $Kind -ArchivePath $archivePath -Episode $Episode)
    $status = if ($matches.Count -eq 1) {
        "recovered_unique_exact_hash"
    }
    elseif ($matches.Count -gt 1) {
        "ambiguous_exact_hash_matches"
    }
    else {
        "not_recorded_unrecoverable"
    }
    return [PSCustomObject]@{
        InputPath = if ($matches.Count -eq 1) { [string]$matches[0] } else { "" }
        InitializationMode = [string]$mode
        ArchivePath = [string]$archivePath
        RecoveryStatus = $status
        RecoveryCandidates = ($matches -join ";")
    }
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

$resultsRoot = Resolve-RepoPath $ResultsDir
if (-not (Test-Path -LiteralPath $resultsRoot)) {
    throw "ResultsDir does not exist: $resultsRoot"
}
$logsRoot = Resolve-RepoPath $LogsDir

$runsRootResolved = Resolve-RepoPath $RunsRoot
$runDirs = @()
if (Test-Path -LiteralPath $runsRootResolved) {
    $runDirs = @(Get-ChildItem -LiteralPath $runsRootResolved -Directory | Sort-Object -Property CreationTime)
}
else {
    Write-Warning "RunsRoot does not exist, runs column will be empty: $runsRootResolved"
}

$episodeNumbers = Convert-EpisodeTokens $Episodes

if (-not $episodeNumbers -or $episodeNumbers.Count -eq 0) {
    $episodeNumbers = Get-ChildItem -LiteralPath $resultsRoot -Directory |
        Where-Object {
            $_.Name -match '^\d+$' -and
            $null -ne (Resolve-EpisodeReportLocation -EpisodeRootDir $_.FullName)
        } |
        ForEach-Object { [int]$_.Name } |
        Sort-Object
}

Write-Host "[FewShotSummary] Repo root: $RepoRoot"
Write-Host "[FewShotSummary] Results dir: $resultsRoot"
Write-Host "[FewShotSummary] Episodes: $($episodeNumbers -join ', ')"

$rows = @()

foreach ($episode in $episodeNumbers) {
    $episodeRootDir = Join-Path $resultsRoot ([string]$episode)
    $reportLocation = Resolve-EpisodeReportLocation -EpisodeRootDir $episodeRootDir
    if ($null -eq $reportLocation) {
        Write-Warning "Missing final_report.json for episode ${episode}: checked $episodeRootDir and $(Join-Path $episodeRootDir 'evolution')"
        continue
    }
    $episodeDir = [string]$reportLocation.ResultDir
    $reportPath = [string]$reportLocation.ReportPath

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
    $planTemplatePolicy = Get-PropValue $report "plan_template_policy"
    $summary = Get-ReportSummary $report
    $fvSummary = Get-PropValue $finalValidation "final_summary"
    $fvGuardSummary = Get-PropValue $finalValidation "final_guard_summary"
    $latestLog = Get-LatestStageLog `
        -Episode $episode `
        -Prefix "few_shot__" `
        -LogsRoot $logsRoot `
        -Variant ([string]$reportLocation.Variant)
    $modelsLine = Get-LatestModelsLine -LogFile $latestLog -Stage "FewShot"
    $episodeRun = Get-EpisodeRunName -LogFile $latestLog -SortedRunDirs $runDirs
    $ruleInputInfo = Get-InitialArtifactInputInfo `
        -Kind "rule" `
        -Inputs $inputs `
        -Detector $detector `
        -PlanTemplatePolicy $planTemplatePolicy `
        -Episode $episode
    $planInputInfo = Get-InitialArtifactInputInfo `
        -Kind "plan" `
        -Inputs $inputs `
        -Detector $detector `
        -PlanTemplatePolicy $planTemplatePolicy `
        -Episode $episode

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
        'round0-correct' = Get-RoundCorrect -EpisodeResultDir $episodeDir -RoundIndex 0
        'round1-correct' = Get-RoundCorrect -EpisodeResultDir $episodeDir -RoundIndex 1
        'round2-correct' = Get-RoundCorrect -EpisodeResultDir $episodeDir -RoundIndex 2
        'round3-correct' = Get-RoundCorrect -EpisodeResultDir $episodeDir -RoundIndex 3
        'round4-correct' = Get-RoundCorrect -EpisodeResultDir $episodeDir -RoundIndex 4
        cold_start = if ($ruleInputInfo.InitializationMode -eq "cold_start_agent") {
            "True"
        }
        elseif ($ruleInputInfo.InitializationMode -eq "rule_file") {
            "False"
        }
        else {
            Get-ColdStartStatus $initialRule
        }
        runs = $episodeRun
        GenerateInitialPlanWithAgent = Get-GenerateInitialPlanWithAgentStatus `
            -PlanTemplatePolicy $planTemplatePolicy `
            -InitialPlan $initialPlan
        input_rule_file = $ruleInputInfo.InputPath
        input_plan_file = $planInputInfo.InputPath
        rule_initialization_mode = $ruleInputInfo.InitializationMode
        plan_initialization_mode = $planInputInfo.InitializationMode
        input_rule_recovery_status = $ruleInputInfo.RecoveryStatus
        input_plan_recovery_status = $planInputInfo.RecoveryStatus
        input_rule_recovery_candidates = $ruleInputInfo.RecoveryCandidates
        input_plan_recovery_candidates = $planInputInfo.RecoveryCandidates
        initial_rule_archive = $ruleInputInfo.ArchivePath
        initial_plan_archive = $planInputInfo.ArchivePath
        # Source means the caller-supplied input. Episode-local copies belong
        # in initial_rule_archive / initial_plan_archive instead.
        rule_source = $ruleInputInfo.InputPath
        plan_source = $planInputInfo.InputPath
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
        report_path = (Convert-ToRepoRelativePath $reportPath) -replace '\\', '/'
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
