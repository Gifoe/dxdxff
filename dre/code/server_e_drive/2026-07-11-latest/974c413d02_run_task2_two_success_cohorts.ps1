param(
    [Parameter(Mandatory = $true)][string]$FeatureCache,
    [Parameter(Mandatory = $true)][string]$Success80Manifest,
    [Parameter(Mandatory = $true)][string]$Success90Manifest,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$Models = "majority,elasticnet,rbf_svm,random_forest,lightgbm",
    [string]$Seeds = "42,52,62"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

function Invoke-Task2Cohort {
    param([int]$SuccessCount, [string]$Manifest)

    $output = Join-Path $OutputRoot "success$SuccessCount`_plus_failure"
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    & python scripts/task2_baselines/run_task2_baseline_suite.py `
        --feature-cache $FeatureCache `
        --success-manifest $Manifest `
        --cohort-name "task2_success$SuccessCount`_plus_failure" `
        --positive-class failure `
        --output-dir $output `
        --models $Models `
        --seeds $Seeds `
        --fixed-baseline `
        --strict
    if ($LASTEXITCODE -ne 0) {
        throw "Task 2 success$SuccessCount + failure cohort failed with exit code $LASTEXITCODE."
    }
}

Invoke-Task2Cohort -SuccessCount 80 -Manifest $Success80Manifest
Invoke-Task2Cohort -SuccessCount 90 -Manifest $Success90Manifest
