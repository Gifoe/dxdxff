param([int[]]$Seeds = @(42), [string[]]$Profiles = @("R0_NEZ_OUTSIDE","R1_NEZ_CONSENSUS","R2_PERSISTENT_RESIDUAL","R3_VIRTUAL_RESECTION","R4_MONOTONE_STRUCTURED","R5_BOUNDED_SET_RESIDUAL","R6_ROBUST_FULL"))
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
foreach ($Profile in $Profiles) { foreach ($Seed in $Seeds) {
  $ProfileOutput = "$NvrRoot\$Profile\seed_$Seed"
  $Audit = "$ProfileOutput\fold_protocol_audit.json"
  if (Test-Path -LiteralPath $Audit) {
    $State = Get-Content -LiteralPath $Audit -Raw | ConvertFrom-Json
    if ($State.n_outer_folds -eq 5 -and $State.paper_valid -eq $true) { Write-Host "Skipping completed $Profile seed $Seed" -ForegroundColor Yellow; continue }
  }
  Write-Host "`n========== Starting NVR $Profile seed $Seed ==========" -ForegroundColor Cyan
  $Arguments = @("$Project\scripts\task2\run_nvr_outcome.py", "--profile", $Profile, "--outcome_table", $env:DRE_TASK2_OUTCOME_TABLE,
    "--feature_cache", $env:DRE_TASK1_FEATURE_CACHE_PATH, "--graph_cache", $GraphCache, "--p2_checkpoint_root", $env:DRE_TASK1_P2_CHECKPOINT_ROOT,
    "--p2_runtime_root", $env:DRE_TASK1_P2_RUNTIME_ROOT, "--p2_training_manifest", $env:DRE_TASK1_P2_TRAINING_MANIFEST,
    "--fold_manifest", $env:DRE_TASK2_FOLD_MANIFEST, "--exclusion_manifest", $env:DRE_TASK2_EXCLUSION_MANIFEST,
    "--protocol", "outer_cv", "--outer_folds", "5", "--epochs", "25", "--batch_size", "8", "--seed", $Seed,
    "--evidence_cache_dir", "$NvrRoot\_frozen_p2_evidence\seed_$Seed", "--output_dir", $ProfileOutput, "--resume", "--strict")
  & $PythonExe @Arguments
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}}
