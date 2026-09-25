param(
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [Parameter(Mandatory = $true)][string]$Cache,
    [Parameter(Mandatory = $true)][string]$Subjects,
    [Parameter(Mandatory = $true)][string]$Folds,
    [Parameter(Mandatory = $true)][string]$Splits,
    [Parameter(Mandatory = $true)][string]$Reference,
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$Seeds = "42,52,62"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot
& $PythonExe scripts\task1_baselines\run_task1_baseline_suite.py `
    --feature-cache $Cache `
    --cohort-manifest $Subjects `
    --fold-manifest $Folds `
    --split-manifest $Splits `
    --reference-ledger $Reference `
    --expected-patients 80 `
    --protocol-name sensitivity80_p2_q10 `
    --feature-profile p2_matched_simple `
    --selection-protocol fixed_validation `
    --models "logistic_regression,rbf_svm,random_forest,lightgbm" `
    --seeds $Seeds `
    --output-dir $Output `
    --strict
exit $LASTEXITCODE
