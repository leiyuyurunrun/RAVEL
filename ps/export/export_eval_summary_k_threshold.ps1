param(
    [string[]]$Episodes = @(),
    [int]$K = 1,
    [string]$ResultsRoot = "data\results",
    [string]$OutputCsv = ""
)
# ps\export\export_eval_summary_k_threshold.ps1 -Episodes 120..137 -K 1
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot

$NegativeRawLabels = @(
    "non_target",
    "benign",
    "normal",
    "safe",
    "legitimate",
    "negative",
    "unknown_other",
    "other",
    "other_attack"
)

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return Join-Path $ProjectRoot $Path
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

function Normalize-Label {
    param([string]$Value)
    return (($Value.Trim().ToLower()) -replace "[-\s]+", "_")
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
                $numbers += $start..$end
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

function Sum-RawCounts {
    param(
        $Counts,
        [bool]$WantTarget
    )

    $sum = 0
    if ($null -eq $Counts) {
        return $sum
    }

    foreach ($prop in $Counts.PSObject.Properties) {
        $name = Normalize-Label $prop.Name
        $isNegative = $NegativeRawLabels -contains $name
        if ($WantTarget -and -not $isNegative) {
            $sum += [int]$prop.Value
        }
        elseif (-not $WantTarget -and $isNegative) {
            $sum += [int]$prop.Value
        }
    }
    return $sum
}

function Get-TargetCategory {
    param($Counts)

    if ($null -eq $Counts) {
        return ""
    }

    $targets = @()
    foreach ($prop in $Counts.PSObject.Properties) {
        $name = Normalize-Label $prop.Name
        if ($NegativeRawLabels -notcontains $name) {
            $targets += $name
        }
    }
    return ($targets | Sort-Object -Unique) -join "|"
}

function Has-NormalizedGroundTruthCounts {
    param($Counts)
    if ($null -eq $Counts) {
        return $false
    }
    return $null -ne $Counts.PSObject.Properties["non_target"]
}

function Get-CountSource {
    param($Summary)

    $groundCounts = Get-PropValue $Summary "ground_truth_counts"
    if (Has-NormalizedGroundTruthCounts $groundCounts) {
        return $groundCounts
    }

    $rawCounts = Get-PropValue $Summary "raw_ground_truth_counts"
    if ($null -ne $rawCounts) {
        return $rawCounts
    }

    return Get-PropValue $Summary "raw_ground_truth_detail_counts"
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

function Safe-Divide {
    param(
        [double]$Numerator,
        [double]$Denominator
    )
    if ($Denominator -le 0) {
        return 0
    }
    return [math]::Round($Numerator / $Denominator, 6)
}

function Get-JudgeMismatchCount {
    param($Case)

    $ids = New-Object 'System.Collections.Generic.HashSet[string]'
    $fallbackCount = 0

    $failed = Get-PropValue $Case "failed_conditions" @()
    foreach ($item in @($failed)) {
        if ($null -eq $item) {
            continue
        }
        $conditionId = [string](Get-PropValue $item "condition_id" "")
        if ([string]::IsNullOrWhiteSpace($conditionId)) {
            $fallbackCount += 1
        }
        else {
            [void]$ids.Add($conditionId)
        }
    }

    $uncertain = Get-PropValue $Case "uncertain_conditions" @()
    foreach ($item in @($uncertain)) {
        if ($null -eq $item) {
            continue
        }
        $conditionId = [string]$item
        if ([string]::IsNullOrWhiteSpace($conditionId)) {
            $fallbackCount += 1
        }
        else {
            [void]$ids.Add($conditionId)
        }
    }

    return ([int]$ids.Count + [int]$fallbackCount)
}

function Is-AttackGroundTruth {
    param($Case)
    return (Normalize-Label ([string](Get-PropValue $Case "ground_truth" ""))) -eq "attack"
}

function Is-BenignGroundTruth {
    param($Case)
    return (Normalize-Label ([string](Get-PropValue $Case "ground_truth" ""))) -eq "benign"
}

$episodeNumbers = Convert-EpisodeTokens $Episodes
if (-not $episodeNumbers -or $episodeNumbers.Count -eq 0) {
    throw "Please provide -Episodes, for example: -Episodes 120..137 or -Episodes 120,121,122"
}

$resultsRootPath = Resolve-RepoPath $ResultsRoot
if (-not (Test-Path -LiteralPath $resultsRootPath)) {
    throw "ResultsRoot does not exist: $resultsRootPath"
}

if ([string]::IsNullOrWhiteSpace($OutputCsv)) {
    $minEpisode = ($episodeNumbers | Measure-Object -Minimum).Minimum
    $maxEpisode = ($episodeNumbers | Measure-Object -Maximum).Maximum
    if ($minEpisode -eq $maxEpisode) {
        $OutputCsv = "data\results\eval_summary_$minEpisode-$($K)_threshold.csv"
    }
    else {
        $OutputCsv = "data\results\eval_summary_${minEpisode}_${maxEpisode}-$($K)_threshold.csv"
    }
}

Write-Host "[EvalKThreshold] Project root: $ProjectRoot"
Write-Host "[EvalKThreshold] Results root: $resultsRootPath"
Write-Host "[EvalKThreshold] Episodes: $($episodeNumbers -join ', ')"
Write-Host "[EvalKThreshold] K threshold: $K"

$rows = @()

foreach ($episode in $episodeNumbers) {
    $summaryPath = Join-Path $resultsRootPath "$episode\eval_summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        Write-Warning "Missing eval_summary.json for episode ${episode}: $summaryPath"
        continue
    }

    try {
        $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        Write-Warning "Skipping episode ${episode}: failed to parse eval_summary.json ($($_.Exception.Message))"
        continue
    }

    $countSource = Get-CountSource $summary
    $predCounts = Get-PropValue $summary "predicted_verdict_counts"

    $target = Sum-RawCounts -Counts $countSource -WantTarget $true
    $nontarget = Sum-RawCounts -Counts $countSource -WantTarget $false
    $predictTarget = As-Int (Get-PropValue $predCounts "attack" 0)
    $predictNontarget = As-Int (Get-PropValue $predCounts "benign" 0)
    $correct = As-Int (Get-PropValue $summary "correct" 0)
    $fp = As-Int (Get-PropValue $summary "fp" 0)
    $fn = As-Int (Get-PropValue $summary "fn" 0)
    $uncertain = As-Int (Get-PropValue $summary "uncertain" 0)

    $errorCases = Get-PropValue $summary "error_cases" @()
    foreach ($case in @($errorCases)) {
        if ($null -eq $case) {
            continue
        }

        $mismatchCount = Get-JudgeMismatchCount $case
        if ($mismatchCount -gt $K) {
            continue
        }

        $predicted = Normalize-Label ([string](Get-PropValue $case "predicted_verdict" ""))

        if (Is-AttackGroundTruth $case) {
            if ($predicted -eq "benign") {
                $fn -= 1
                $predictNontarget -= 1
                $predictTarget += 1
                $correct += 1
            }
            elseif ($predicted -eq "uncertain") {
                $fn -= 1
                $uncertain -= 1
                $predictTarget += 1
                $correct += 1
            }
        }
        elseif (Is-BenignGroundTruth $case) {
            if ($predicted -eq "attack") {
                $fp -= 1
                $predictTarget -= 1
                $predictNontarget += 1
                $correct += 1
            }
            elseif ($predicted -eq "uncertain") {
                $uncertain -= 1
                $predictNontarget += 1
                $correct += 1
            }
        }
    }

    $fp = [math]::Max(0, $fp)
    $fn = [math]::Max(0, $fn)
    $uncertain = [math]::Max(0, $uncertain)
    $predictTarget = [math]::Max(0, $predictTarget)
    $predictNontarget = [math]::Max(0, $predictNontarget)
    $errors = $fp + $fn
    $truePositiveLike = $target - $fn
    if ($truePositiveLike -lt 0) {
        $truePositiveLike = 0
    }

    $rows += [PSCustomObject]@{
        episode = $episode
        category = Get-TargetCategory $countSource
        target = $target
        nontarget = $nontarget
        predict_target = $predictTarget
        predict_nontarget = $predictNontarget
        correct = $correct
        errors = $errors
        fp = $fp
        fn = $fn
        uncertain = $uncertain
        attack_recall = Safe-Divide -Numerator $truePositiveLike -Denominator $target
        attack_precision = Safe-Divide -Numerator $truePositiveLike -Denominator $predictTarget
        negative_specificity = Safe-Divide -Numerator ($nontarget - $fp) -Denominator $nontarget
    }
}

$outPath = Resolve-RepoPath $OutputCsv
$outDir = Split-Path $outPath -Parent
if (-not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

$rows |
    Sort-Object episode |
    Export-Csv -LiteralPath $outPath -NoTypeInformation -Encoding UTF8

Write-Host "Exported $($rows.Count) rows to $outPath"
