param([int[]]$Seeds = @(42))
$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
. "$Project\scripts\task2\ensure_npam_protocol.ps1"
$PythonExe = if ($env:DRE_PYTHON_EXE) { $env:DRE_PYTHON_EXE } else { "python" }
$NvrRoot = if ($env:DRE_TASK2_NVR_OUTPUT_DIR) { $env:DRE_TASK2_NVR_OUTPUT_DIR } else { "$env:DRE_TASK2_OUTPUT_DIR\NVR_OUTCOME" }
$GraphCache = if ($env:DRE_TASK2_GRAPH_CACHE) { $env:DRE_TASK2_GRAPH_CACHE } else { "$env:DRE_TASK2_OUTPUT_DIR\graph_cache\phase_functional_graphs.pkl" }
if (-not (Test-Path -LiteralPath $GraphCache)) {
  & "$Project\scripts\task2\run_npam_graph_cache.ps1"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
foreach ($Control in @("NVR_TARGET_PERMUTATION","NVR_P2_PERMUTATION","NVR_GRAPH_PERMUTATION")) { foreach ($Seed in $Seeds) {
  $Output = "$NvrRoot\negative_controls\$Control\R6_ROBUST_FULL\seed_$Seed"
  Write-Host "`n========== Starting $Control seed $Seed ==========" -ForegroundColor Cyan
  & $PythonExe "$Project\scripts\task2\run_nvr_outcome.py" --profile R6_ROBUST_FULL --negative_control $Control `
    --outcome_table $env:DRE_TASK2_OUTCOME_TABLE --feature_cache $env:DRE_TASK1_FEATURE_CACHE_PATH --graph_cache $GraphCache `
    --p2_checkpoint_root $env:DRE_TASK1_P2_CHECKPOINT_ROOT --p2_runtime_root $env:DRE_TASK1_P2_RUNTIME_ROOT `
    --p2_training_manifest $env:DRE_TASK1_P2_TRAINING_MANIFEST --fold_manifest $env:DRE_TASK2_FOLD_MANIFEST `
    --exclusion_manifest $env:DRE_TASK2_EXCLUSION_MANIFEST --protocol outer_cv --outer_folds 5 --epochs 25 --batch_size 8 --seed $Seed `
    --evidence_cache_dir "$NvrRoot\_frozen_p2_evidence\seed_$Seed" --output_dir $Output --resume --strict
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}}
