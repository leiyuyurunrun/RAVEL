param(
    [Parameter(Mandatory = $true)]
    [string]$WTree
)

$Origin = "D:\pretended_home\EvoTx-2"
$RunsRoot = "D:\pretended_home\EvoTx-runs"

# 如果传入的是完整路径，就直接使用；否则拼接到 RunsRoot 下
if ([System.IO.Path]::IsPathRooted($WTree)) {
    $Worktree = $WTree
} else {
    $Worktree = Join-Path $RunsRoot $WTree
}

Write-Host "Origin   : $Origin"
Write-Host "Worktree : $Worktree"

# 进入原始 Git 仓库
Set-Location $Origin

# 创建 detached worktree
git worktree add --detach "$Worktree" HEAD

if ($LASTEXITCODE -ne 0) {
    Write-Error "git worktree add failed."
    exit 1
}

# 删除 worktree 中已有的占位文件/目录，然后建立链接
function Remove-IfExists {
    param(
        [string]$Path
    )

    if (Test-Path -LiteralPath $Path) {
        Write-Host "Remove existing: $Path"
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

# 文件软链接：.env
Remove-IfExists "$Worktree\.env"
cmd /c mklink "$Worktree\.env" "$Origin\.env"

# 文件软链接：0702.md
Remove-IfExists "$Worktree\0702.md"
cmd /c mklink "$Worktree\0702.md" "$Origin\0702.md"

# 文件软链接：start_standard.ps1
Remove-IfExists "$Worktree\start_standard.ps1"
cmd /c mklink "$Worktree\start_standard.ps1" "$Origin\start_standard.ps1"
# 目录 Junction：.venv
Remove-IfExists "$Worktree\.venv"
cmd /c mklink /J "$Worktree\.venv" "$Origin\.venv"

# 目录 Junction：data
Remove-IfExists "$Worktree\data"
cmd /c mklink /J "$Worktree\data" "$Origin\data"

# 进入 worktree 并检查 Python 环境
Set-Location $Worktree

Write-Host "`nChecking venv..."
.\.venv\Scripts\python.exe -c "import sys; print('prefix=', sys.prefix); print('base_prefix=', sys.base_prefix)"

Write-Host "`nChecking evotx import path..."
.\.venv\Scripts\python.exe -c "import evotx, inspect; print(inspect.getfile(evotx))"

Write-Host "`nDone."
Write-Host "Now you are in: $Worktree"