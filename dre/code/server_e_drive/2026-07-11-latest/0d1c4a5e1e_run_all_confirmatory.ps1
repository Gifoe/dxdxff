param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$Python = "python",
    [string]$Config = "",
    [string]$Seeds = "42,52,62",
    [switch]$RunPooledCV,
    [switch]$RunLOCO,
    [switch]$RunStatistics,
    [switch]$RunEfficiency,
    [ValidateSet("hup", "lzu", "multicenter", "pediatric")][string]$HeldOutCenter = "",
    [ValidateRange(1,5)][int]$OuterFold = 0,
    [switch]$Resume,
    [switch]$SkipCompleted,
    [switch]$Strict,
    [switch]$DryRun,
    [switch]$Smoke
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepoRoot
$env:PYTHONPATH = "$RepoRoot;$env:PYTHONPATH"
if (-not $Config) { $Config = Join-Path $RepoRoot "configs\task1_confirmatory.yaml" }
$arguments = @(".\scripts\task1_confirmatory\run_all_confirmatory.py", "--config", $Config, "--seeds", $Seeds)
if ($RunPooledCV) { $arguments += "--run_pooled_cv" }
if ($RunLOCO) { $arguments += "--run_loco" }
if ($RunStatistics) { $arguments += "--run_statistics" }
if ($RunEfficiency) { $arguments += "--run_efficiency" }
if ($HeldOutCenter) { $arguments += @("--held_out_center", $HeldOutCenter) }
if ($OuterFold) { $arguments += @("--outer_fold", "$OuterFold") }
if ($Resume) { $arguments += "--resume" }
if ($SkipCompleted) { $arguments += "--skip_completed" }
if ($Strict) { $arguments += "--strict" }
if ($DryRun) { $arguments += "--dry_run" }
if ($Smoke) { $arguments += "--smoke" }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Task 1 confirmatory runner failed with exit code $LASTEXITCODE" }
