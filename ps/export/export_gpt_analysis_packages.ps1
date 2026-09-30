param(
    [Parameter(Mandatory = $true)]
    [int]$Episode,

    [string]$ResultsRoot = "data\results",
    [string]$RulesRoot = "data\rules",
    [string]$PlansRoot = "data\plans",
    [string]$LogsRoot = "data\log",
    [string]$ScriptsRoot = "ps",
    [string]$OutputDir = "",
    [int]$LogTailLines = 800,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

function Resolve-RepoPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return ""
    }
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $RepoRoot $Path)
}

function Convert-ToRepoRelativePath {
    param([string]$Path)
    $full = [System.IO.Path]::GetFullPath((Resolve-RepoPath $Path))
    $root = [System.IO.Path]::GetFullPath($RepoRoot)
    if (-not $root.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $root = $root + [System.IO.Path]::DirectorySeparatorChar
    }
    if ($full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $full.Substring($root.Length).Replace("\", "/")
    }
    return (Split-Path -Leaf $full)
}

function New-FileEntry {
    param(
        [string]$Source,
        [string]$RelativePath = ""
    )
    $resolved = Resolve-RepoPath $Source
    if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
        return $null
    }
    if ([string]::IsNullOrWhiteSpace($RelativePath)) {
        $RelativePath = Convert-ToRepoRelativePath $resolved
    }
    [pscustomobject]@{
        Source = (Resolve-Path -LiteralPath $resolved).Path
        RelativePath = $RelativePath.Replace("\", "/")
    }
}

function Add-ExistingFile {
    param(
        [System.Collections.ArrayList]$List,
        [string]$Path,
        [string]$RelativePath = ""
    )
    $entry = New-FileEntry -Source $Path -RelativePath $RelativePath
    if ($null -ne $entry) {
        [void]$List.Add($entry)
    }
}

function Add-GlobFiles {
    param(
        [System.Collections.ArrayList]$List,
        [string]$Pattern
    )
    Get-ChildItem -Path (Resolve-RepoPath $Pattern) -File -ErrorAction SilentlyContinue |
        Sort-Object FullName |
        ForEach-Object {
            Add-ExistingFile -List $List -Path $_.FullName
        }
}

function Copy-EntriesToStage {
    param(
        [System.Collections.IEnumerable]$Entries,
        [string]$StageDir
    )
    foreach ($entry in $Entries) {
        if ($null -eq $entry) {
            continue
        }
        $dest = Join-Path $StageDir $entry.RelativePath
        $destDir = Split-Path -Parent $dest
        if (-not (Test-Path -LiteralPath $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        Copy-Item -LiteralPath $entry.Source -Destination $dest -Force
    }
}

function Write-Readme {
    param(
        [string]$StageDir,
        [string]$Title,
        [string]$Purpose,
        [System.Collections.IEnumerable]$Entries
    )
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# $Title")
    $lines.Add("")
    $lines.Add("Episode: $Episode")
    $lines.Add("Created at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
    $lines.Add("")
    $lines.Add("Purpose: $Purpose")
    $lines.Add("")
    $lines.Add("Files:")
    foreach ($entry in $Entries) {
        $lines.Add("- $($entry.RelativePath)")
    }
    Set-Content -Path (Join-Path $StageDir "README.md") -Value $lines -Encoding UTF8
}

function New-ZipPackage {
    param(
        [string]$Name,
        [string]$Purpose,
        [System.Collections.IEnumerable]$Entries,
        [string]$PackageDir,
        [string]$Timestamp
    )
    $stage = Join-Path $PackageDir "_stage_${Name}_$Timestamp"
    if (Test-Path -LiteralPath $stage) {
        Remove-Item -LiteralPath $stage -Recurse -Force
    }
    New-Item -ItemType Directory -Path $stage -Force | Out-Null

    $unique = @{}
    foreach ($entry in $Entries) {
        if ($null -eq $entry) {
            continue
        }
        $unique[$entry.RelativePath] = $entry
    }
    $deduped = $unique.Values | Sort-Object RelativePath
    Copy-EntriesToStage -Entries $deduped -StageDir $stage
    Write-Readme -StageDir $stage -Title "episode${Episode}_${Name}" -Purpose $Purpose -Entries $deduped

    $zipPath = Join-Path $PackageDir ("episode{0}_{1}_{2}.zip" -f $Episode, $Name, $Timestamp)
    if ((Test-Path -LiteralPath $zipPath) -and -not $Force) {
        throw "Package already exists: $zipPath. Use -Force to overwrite."
    }
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zipPath -Force
    Remove-Item -LiteralPath $stage -Recurse -Force
    return (Resolve-Path -LiteralPath $zipPath).Path
}

function Get-JsonFile {
    param([string]$Path)
    $resolved = Resolve-RepoPath $Path
    if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
        return $null
    }
    return Get-Content -LiteralPath $resolved -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Add-ReportArtifactPaths {
    param(
        [System.Collections.ArrayList]$List,
        $Report
    )
    if ($null -eq $Report) {
        return
    }
    $candidatePaths = @(
        $Report.detector_context.initial_rule.rule_path,
        $Report.detector_context.final_rule.rule_path,
        $Report.detector_context.initial_plan.plan_path,
        $Report.detector_context.final_plan.plan_path
    )
    foreach ($path in $candidatePaths) {
        if (-not [string]::IsNullOrWhiteSpace([string]$path)) {
            Add-ExistingFile -List $List -Path ([string]$path)
        }
    }
}

function Add-LatestRulePlanFiles {
    param(
        [System.Collections.ArrayList]$List,
        [int]$EpisodeId
    )
    Add-GlobFiles -List $List -Pattern (Join-Path $RulesRoot "$EpisodeId\*latest*.json")
    Add-GlobFiles -List $List -Pattern (Join-Path $RulesRoot "$EpisodeId\*_v*.json")
    Add-GlobFiles -List $List -Pattern (Join-Path $PlansRoot "$EpisodeId\*latest*.json")
    Add-GlobFiles -List $List -Pattern (Join-Path $PlansRoot "$EpisodeId\*_v*.json")
}

function Add-EpisodeScripts {
    param(
        [System.Collections.ArrayList]$List,
        [int]$EpisodeId
    )
    Add-GlobFiles -List $List -Pattern (Join-Path $ScriptsRoot "*ep${EpisodeId}*.ps1")
    Add-GlobFiles -List $List -Pattern (Join-Path $ScriptsRoot "*episode${EpisodeId}*.ps1")
}

function New-LogTailFile {
    param(
        [string]$PackageDir,
        [int]$EpisodeId,
        [int]$TailLines
    )
    $logDir = Resolve-RepoPath (Join-Path $LogsRoot "$EpisodeId")
    if (-not (Test-Path -LiteralPath $logDir -PathType Container)) {
        return $null
    }
    $latestLog = Get-ChildItem -LiteralPath $logDir -File -Filter "*.log" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latestLog) {
        return $null
    }
    $generatedDir = Join-Path $PackageDir "_generated"
    New-Item -ItemType Directory -Path $generatedDir -Force | Out-Null
    $tailPath = Join-Path $generatedDir ("episode_{0}_fewshot_tail_{1}.log" -f $EpisodeId, $TailLines)
    Get-Content -LiteralPath $latestLog.FullName -Tail $TailLines -Encoding UTF8 |
        Set-Content -LiteralPath $tailPath -Encoding UTF8
    return New-FileEntry -Source $tailPath -RelativePath ("logs/episode_{0}_fewshot_tail_{1}.log" -f $EpisodeId, $TailLines)
}

$episodeResultsDir = Resolve-RepoPath (Join-Path $ResultsRoot "$Episode")
if (-not (Test-Path -LiteralPath $episodeResultsDir -PathType Container)) {
    throw "Episode results directory not found: $episodeResultsDir"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path (Join-Path $ResultsRoot "$Episode") "gpt_analysis_packages"
}
$packageDir = Resolve-RepoPath $OutputDir
New-Item -ItemType Directory -Path $packageDir -Force | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$finalReportPath = Join-Path (Join-Path $ResultsRoot "$Episode") "final_report.json"
$report = Get-JsonFile -Path $finalReportPath

$overview = New-Object System.Collections.ArrayList
foreach ($name in @(
    "final_report.json",
    "final_validation.json",
    "final_validation_comparison.json",
    "final_validation_guard_summary.json",
    "final_validation_summary.json"
)) {
    Add-ExistingFile -List $overview -Path (Join-Path (Join-Path $ResultsRoot "$Episode") $name)
}
$tailEntry = New-LogTailFile -PackageDir $packageDir -EpisodeId $Episode -TailLines $LogTailLines
if ($null -ne $tailEntry) {
    [void]$overview.Add($tailEntry)
}

$evolution = New-Object System.Collections.ArrayList
$roundFiles = @(
    "reviews.json",
    "review_bundle.json",
    "current_summary.json",
    "current_artifact.json",
    "candidate_artifact.json",
    "candidate_evaluations.json",
    "candidate_rule.json",
    "candidate_plan.json",
    "candidate_summary.json",
    "comparison.json",
    "guard_comparison.json",
    "repair_status.json",
    "repair_reviews.json",
    "repair_review_bundle.json",
    "repair_rule.json",
    "repair_plan.json",
    "repair_summary.json",
    "repair_comparison.json",
    "repair_guard_summary.json",
    "repair_guard_comparison.json"
)
Get-ChildItem -LiteralPath $episodeResultsDir -Directory -Filter "round_*" -ErrorAction SilentlyContinue |
    Sort-Object Name |
    ForEach-Object {
        foreach ($file in $roundFiles) {
            Add-ExistingFile -List $evolution -Path (Join-Path $_.FullName $file)
        }
    }

$finalArtifacts = New-Object System.Collections.ArrayList
foreach ($name in @(
    "final_validation.json",
    "final_validation_summary.json",
    "final_validation_comparison.json",
    "final_validation_slim_results.json",
    "final_validation_guard_summary.json",
    "final_validation_guard_slim_results.json"
)) {
    Add-ExistingFile -List $finalArtifacts -Path (Join-Path (Join-Path $ResultsRoot "$Episode") $name)
}
Add-ReportArtifactPaths -List $finalArtifacts -Report $report
Add-LatestRulePlanFiles -List $finalArtifacts -EpisodeId $Episode
Add-EpisodeScripts -List $finalArtifacts -EpisodeId $Episode

$overviewZip = New-ZipPackage `
    -Name "overview_minimal" `
    -Purpose "Minimal overview: final report, final validation, summaries, comparison, and log tail." `
    -Entries $overview `
    -PackageDir $packageDir `
    -Timestamp $timestamp

$evolutionZip = New-ZipPackage `
    -Name "evolution_process" `
    -Purpose "Evolution process: round reviews, review bundles, candidates, comparisons, guard comparisons, and repair status." `
    -Entries $evolution `
    -PackageDir $packageDir `
    -Timestamp $timestamp

$finalZip = New-ZipPackage `
    -Name "final_validation_artifacts" `
    -Purpose "Final validation artifacts: final validation slim/summary files, final rule/plan, and run scripts." `
    -Entries $finalArtifacts `
    -PackageDir $packageDir `
    -Timestamp $timestamp

$allEntries = New-Object System.Collections.ArrayList
foreach ($entry in @($overview + $evolution + $finalArtifacts)) {
    if ($null -ne $entry) {
        [void]$allEntries.Add($entry)
    }
}
$allZip = New-ZipPackage `
    -Name "gpt_analysis_all_in_one" `
    -Purpose "All-in-one package: main contents from overview, evolution process, and final validation artifacts." `
    -Entries $allEntries `
    -PackageDir $packageDir `
    -Timestamp $timestamp

$manifest = [ordered]@{
    episode = $Episode
    created_at = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    package_dir = Convert-ToRepoRelativePath $packageDir
    log_tail_lines = $LogTailLines
    packages = @(
        [ordered]@{
            name = "overview_minimal"
            path = Convert-ToRepoRelativePath $overviewZip
            purpose = "Overview, stop reason, final validation pass/fail"
        }
        [ordered]@{
            name = "evolution_process"
            path = Convert-ToRepoRelativePath $evolutionZip
            purpose = "Round reviewer/candidate/comparison/guard/repair analysis"
        }
        [ordered]@{
            name = "final_validation_artifacts"
            path = Convert-ToRepoRelativePath $finalZip
            purpose = "final_validation_slim_results + final rule/plan + run scripts"
        }
        [ordered]@{
            name = "all_in_one"
            path = Convert-ToRepoRelativePath $allZip
            purpose = "All-in-one upload package for GPT"
        }
    )
}
$manifestPath = Join-Path $packageDir ("episode{0}_gpt_analysis_manifest_{1}.json" -f $Episode, $timestamp)
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

if (Test-Path -LiteralPath (Join-Path $packageDir "_generated")) {
    Remove-Item -LiteralPath (Join-Path $packageDir "_generated") -Recurse -Force
}

Write-Host "[GPT-Packages] Episode $Episode packages created:"
Write-Host "  overview_minimal:          $overviewZip"
Write-Host "  evolution_process:         $evolutionZip"
Write-Host "  final_validation_artifacts:$finalZip"
Write-Host "  all_in_one:                $allZip"
Write-Host "  manifest:                  $manifestPath"
