param(
    [string[]]$Episodes = @("2000..2060"),
    [string]$ResultsDir = "data\results",
    [string]$LogsDir = "data\log",
    [string]$Out = ""
)

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
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $ProjectRoot
    }
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $ProjectRoot $Path)
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
                $numbers += ([int]$Matches[1])..([int]$Matches[2])
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
        [string[]]$Keys
    )
    $pairs = @()
    foreach ($key in $Keys) {
        $match = [regex]::Match(
            $ModelsLine,
            "(?:^|\s)$([regex]::Escape($key))=(?<value>\S+)"
        )
        if ($match.Success -and $match.Groups["value"].Value -ne "unset") {
            $pairs += [PSCustomObject]@{
                Key = $key
                Value = $match.Groups["value"].Value
            }
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

function Get-FirstDetectorContext {
    param([string]$EvalResultsPath)
    if (-not (Test-Path -LiteralPath $EvalResultsPath)) {
        return $null
    }

    $reader = [System.IO.File]::OpenText($EvalResultsPath)
    $builder = New-Object System.Text.StringBuilder
    $capturing = $false
    $depth = 0
    $inString = $false
    $escaped = $false
    try {
        while (($line = $reader.ReadLine()) -ne $null) {
            $startIndex = 0
            if (-not $capturing) {
                if ($line -notmatch '"detector_context"\s*:') {
                    continue
                }
                $startIndex = $line.IndexOf('{')
                if ($startIndex -lt 0) {
                    continue
                }
                $capturing = $true
            }

            for ($index = $startIndex; $index -lt $line.Length; $index++) {
                $char = $line[$index]
                [void]$builder.Append($char)
                if ($escaped) {
                    $escaped = $false
                    continue
                }
                if ($inString -and $char -eq '\') {
                    $escaped = $true
                    continue
                }
                if ($char -eq '"') {
                    $inString = -not $inString
                    continue
                }
                if (-not $inString) {
                    if ($char -eq '{') {
                        $depth++
                    }
                    elseif ($char -eq '}') {
                        $depth--
                        if ($depth -eq 0) {
                            return ($builder.ToString() | ConvertFrom-Json)
                        }
                    }
                }
            }
            [void]$builder.AppendLine()
            $startIndex = 0
        }
    }
    finally {
        $reader.Dispose()
    }
    return $null
}

function Get-PropValue {
    param(
        [Parameter(Mandatory = $true)]
        [AllowNull()]
        $Object,

        [Parameter(Mandatory = $true)]
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

function Sum-RawCounts {
    param(
        [Parameter(Mandatory = $true)]
        $Counts,

        [Parameter(Mandatory = $true)]
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

function Get-EvalSummaryFiles {
    param([string]$EpisodeDir)
    # 匹配 eval*_summary.json，覆盖通用 eval_summary.json 与带标签的 eval_<tag>_summary.json
    if (-not (Test-Path -LiteralPath $EpisodeDir)) {
        return @()
    }
    return @(Get-ChildItem -LiteralPath $EpisodeDir -File -Filter "eval*_summary.json" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^eval.*_summary\.json$' } |
        Sort-Object Name)
}

function Get-EvalTag {
    param([string]$SummaryFileName)
    # eval_summary.json -> "" ; eval_price_manipulation_summary.json -> "price_manipulation"
    $base = [System.IO.Path]::GetFileNameWithoutExtension($SummaryFileName)
    if ($base -notmatch '^eval_(.*)_summary$') {
        return ""
    }
    return $Matches[1]
}

$resultsRoot = Resolve-RepoPath $ResultsDir
$logsRoot = Resolve-RepoPath $LogsDir
if (-not (Test-Path -LiteralPath $resultsRoot)) {
    throw "ResultsDir does not exist: $resultsRoot"
}

$episodeNumbers = Convert-EpisodeTokens $Episodes
if (-not $episodeNumbers -or $episodeNumbers.Count -eq 0) {
    $episodeNumbers = Get-ChildItem -LiteralPath $resultsRoot -Directory |
        Where-Object {
            $_.Name -match '^\d+$' -and
            (Get-EvalSummaryFiles -EpisodeDir $_.FullName).Count -gt 0
        } |
        ForEach-Object { [int]$_.Name } |
        Sort-Object
}

Write-Host "[EvalSummary] Project root: $ProjectRoot"
Write-Host "[EvalSummary] Results dir: $resultsRoot"
Write-Host "[EvalSummary] Episodes: $($episodeNumbers -join ', ')"

$rows = @()

foreach ($episode in $episodeNumbers) {
    $episodeDir = Join-Path $resultsRoot ([string]$episode)
    $summaryFiles = Get-EvalSummaryFiles -EpisodeDir $episodeDir
    if (-not $summaryFiles -or $summaryFiles.Count -eq 0) {
        Write-Warning "Missing eval*_summary.json for episode ${episode}: $episodeDir"
        continue
    }

    foreach ($summaryFile in $summaryFiles) {
        $summaryPath = $summaryFile.FullName
        $evalTag = Get-EvalTag $summaryFile.Name
        # summary 与 results 共用前缀：把 _summary.json 替换为 _results.json 即为配套文件
        $resultsFileName = $summaryFile.Name -replace '_summary\.json$', '_results.json'
        $evalResultsPath = Join-Path $episodeDir $resultsFileName

        $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 |
            ConvertFrom-Json
        try {
            $detectorContext = Get-FirstDetectorContext $evalResultsPath
        }
        catch {
            Write-Warning "Failed to read detector_context for episode ${episode} ($($summaryFile.Name)): $($_.Exception.Message)"
            $detectorContext = $null
        }
        $latestLog = Get-LatestStageLog -Episode $episode -Prefix "evaluate__" -LogsRoot $logsRoot
        $modelsLine = Get-LatestModelsLine -LogFile $latestLog -Stage "Evaluate"
        $groundCounts = Get-PropValue $summary "ground_truth_counts"
        if (Has-NormalizedGroundTruthCounts $groundCounts) {
            $countSource = $groundCounts
        }
        else {
            $countSource = Get-PropValue $summary "raw_ground_truth_counts"
            if ($null -eq $countSource) {
                $countSource = Get-PropValue $summary "raw_ground_truth_detail_counts"
            }
        }
        $predCounts = Get-PropValue $summary "predicted_verdict_counts"

        $rows += [PSCustomObject]@{
            episode = $episode
            eval_tag = $evalTag
            category = Get-TargetCategory $countSource
            target = Sum-RawCounts -Counts $countSource -WantTarget $true
            nontarget = Sum-RawCounts -Counts $countSource -WantTarget $false
            predict_target = [int](Get-PropValue $predCounts "attack" 0)
            predict_nontarget = [int](Get-PropValue $predCounts "benign" 0)
            correct = Get-PropValue $summary "correct" ""
            errors = Get-PropValue $summary "errors" ""
            fp = Get-PropValue $summary "fp" ""
            fn = Get-PropValue $summary "fn" ""
            uncertain = Get-PropValue $summary "uncertain" ""
            attack_recall = Get-PropValue $summary "attack_recall" ""
            attack_precision = Get-PropValue $summary "attack_precision" ""
            # precision = Get-PropValue $summary "precision" ""
            # benign_specificity = Get-PropValue $summary "benign_specificity" ""
            negative_specificity = Get-PropValue $summary "negative_specificity" ""
            model = Get-ModelSummary -ModelsLine $modelsLine -Keys @(
                "planner", "judge", "env"
            )
            rule_source = Get-PropValue $detectorContext "rule_source" ""
            plan_source = Get-PropValue $detectorContext "plan_source" ""
        }
    }
}

$outputCsv = $Out
if ([string]::IsNullOrWhiteSpace($outputCsv)) {
    if ($episodeNumbers.Count -gt 0) {
        $firstEpisode = ($episodeNumbers | Measure-Object -Minimum).Minimum
        $lastEpisode = ($episodeNumbers | Measure-Object -Maximum).Maximum
        $outputCsv = "data\results\eval_summary_${firstEpisode}_${lastEpisode}.csv"
    }
    else {
        $outputCsv = "data\results\eval_summary.csv"
    }
}
$outPath = Resolve-RepoPath $outputCsv
$outDir = Split-Path $outPath -Parent
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$rows | Sort-Object episode, eval_tag | Export-Csv -LiteralPath $outPath -NoTypeInformation -Encoding UTF8

Write-Host "Exported $($rows.Count) rows to $outPath"
