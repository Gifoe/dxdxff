$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonExe = if ($env:DRE_PYTHON_EXE) { $env:DRE_PYTHON_EXE } else { "python" }

if (-not (Test-Path -LiteralPath $env:DRE_TASK2_FOLD_MANIFEST) -or -not (Test-Path -LiteralPath $env:DRE_TASK2_OUTCOME_TABLE)) {
  & $PythonExe "$Project\scripts\task2\prepare_sensitivity80_plus_failures.py" `
    --cache $env:DRE_TASK1_FEATURE_CACHE_PATH `
    --success_subjects $env:DRE_TASK2_SUCCESS_SUBJECTS `
    --p2_fold_ledger $env:DRE_TASK1_P2_TRAINING_MANIFEST `
    --fixed_fold_source $env:DRE_TASK2_SOURCE_FOLD_MANIFEST `
    --fallback_failure_folds $env:DRE_TASK2_FAILURE_FOLD_FALLBACK `
    --output_dir $env:DRE_TASK2_PROTOCOL_DIR
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
