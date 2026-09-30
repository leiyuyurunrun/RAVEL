param(
    [Parameter(Mandatory = $true)]
    [string]$WTree
)

$Origin = "D:\pretended_home\EvoTx-2"
$RunsRoot = "D:\pretended_home\EvoTx-runs"

# 如果传入完整路径，直接使用；否则拼接到 RunsRoot 下
if ([System.IO.Path]::IsPathRooted($WTree)) {
    $Worktree = $WTree
} else {
    $Worktree = Join-Path $RunsRoot $WTree
}

Write-Host "Origin   : $Origin"
Write-Host "Worktree : $Worktree"

if (-not (Test-Path -LiteralPath $Worktree)) {
    Write-Warning "Worktree path does not exist: $Worktree"
    Set-Location $Origin
    git worktree prune
    exit 0
}

function Remove-FileLink {
    param(
        [string]$Path
    )

    if (Test-Path -LiteralPath $Path) {
        Write-Host "Remove file link: $Path"
        cmd /c "del /F /Q `"$Path`""
    }
}

function Remove-DirJunction {
    param(
        [string]$Path
    )

    if (Test-Path -LiteralPath $Path) {
        Write-Host "Remove directory junction: $Path"
        # 注意：这里不用 /S，只删除 junction 入口，不递归删除目标内容
        cmd /c "rmdir `"$Path`""
    }
}

# 先安全删除链接入口
Remove-FileLink "$Worktree\.env"
Remove-FileLink "$Worktree\0702.md"
Remove-FileLink "$Worktree\start_standard.ps1"
Remove-DirJunction "$Worktree\.venv"
Remove-DirJunction "$Worktree\data"

# 再让 git 删除 worktree 本体，并清理 .git/worktrees 记录
Set-Location $Origin

Write-Host "`nRemoving git worktree..."
git worktree remove --force "$Worktree"

if ($LASTEXITCODE -ne 0) {
    Write-Warning "git worktree remove failed. Try pruning worktree records."
    git worktree prune
    Write-Warning "If the folder still exists, check it manually before deleting."
    exit 1
}

Write-Host "`nDone."
Write-Host "Removed worktree safely: $Worktree"
Write-Host "Original linked targets are preserved:"
Write-Host "  $Origin\data"
Write-Host "  $Origin\.venv"
Write-Host "  $Origin\.env"
Write-Host "  $Origin\0702.md"