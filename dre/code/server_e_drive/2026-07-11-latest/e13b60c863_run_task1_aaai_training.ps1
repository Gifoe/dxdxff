param(
    [string]$RepoRoot = "E:\DRE-nips\new-pipeline\7-11",
    [string]$Python = "E:\DRE-nips\new-pipeline\.venv\Scripts\python.exe",
    [string]$Plan = "E:\DRE-nips\new-pipeline\7-11\configs\task1_aaai_training_plan.json",
    [ValidateSet("audit", "final_bcr", "final_cdel", "loco", "all")]
    [string]$Stage = "all",
    [switch]$DryRun,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepoRoot
$env:PYTHONPATH = "$RepoRoot;$env:PYTHONPATH"

$arguments = @(".\scripts\task1_aaai\run_task1_aaai_training.py", "--plan", $Plan, "--stage", $Stage)
if ($DryRun) { $arguments += "--dry-run" }
if ($NoResume) { $arguments += "--no-resume" }

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Task 1 AAAI training plan failed with exit code $LASTEXITCODE."
}
