$ErrorActionPreference = "Stop"

$PatchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Get-Location).Path

$SourceFile = Join-Path $PatchRoot "src\dlt_analyzer_pro\digit_model.py"
$TargetFile = Join-Path $RepoRoot "src\dlt_analyzer_pro\digit_model.py"
$TestSource = Join-Path $PatchRoot "tests\test_digit_credible_engine.py"
$TestTarget = Join-Path $RepoRoot "tests\test_digit_credible_engine.py"

if (-not (Test-Path $TargetFile)) {
    throw "请在 DLT-Analyzer-Pro-5.0.0-AI 仓库根目录运行此脚本。"
}

$Backup = "$TargetFile.phase1-backup"
Copy-Item $TargetFile $Backup -Force
Copy-Item $SourceFile $TargetFile -Force
Copy-Item $TestSource $TestTarget -Force

Write-Host "第一阶段补丁已应用。原 digit_model.py 备份为：$Backup"
Write-Host "建议运行：python -m pytest tests/test_digit_credible_engine.py"
