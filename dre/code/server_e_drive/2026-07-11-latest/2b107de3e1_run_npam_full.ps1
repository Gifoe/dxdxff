param([int[]]$Seeds = @(42))
$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
. "$Project\scripts\task2\ensure_npam_protocol.ps1"
$PythonExe = if ($env:DRE_PYTHON_EXE) { $env:DRE_PYTHON_EXE } else { "python" }
Write-Host "`n========== Strict clinical-target/P2 input audit ==========" -ForegroundColor Cyan
& "$Project\scripts\task2\run_npam_audit.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$GraphCache = "$env:DRE_TASK2_OUTPUT_DIR\graph_cache\phase_functional_graphs.pkl"
if (-not (Test-Path -LiteralPath $GraphCache)) {
  Write-Host "`n========== Building label-blind three-phase graph cache ==========" -ForegroundColor Cyan
  & "$Project\scripts\task2\run_npam_graph_cache.ps1"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$Profiles = @("C0_TARGET_ONLY", "C1_P2_ONLY", "C2_P2_TARGET_CONCORDANCE", "C3_CROSS_SEIZURE_CONCORDANCE", "C4_TARGET_NETWORK", "C5_FULL")
foreach ($Profile in $Profiles) { foreach ($Seed in $Seeds) {
  Write-Host "`n========== Starting $Profile seed $Seed ==========" -ForegroundColor Cyan
  $ProfileOutput = "$env:DRE_TASK2_OUTPUT_DIR\$Profile\seed_$Seed"
  $ProtocolAuditPath = "$ProfileOutput\fold_protocol_audit.json"
  if (Test-Path -LiteralPath $ProtocolAuditPath) {
    $ProtocolAudit = Get-Content -LiteralPath $ProtocolAuditPath -Raw | ConvertFrom-Json
    if ($ProtocolAudit.n_outer_folds -eq 5 -and $ProtocolAudit.paper_valid -eq $true) {
      Write-Host "========== Skipping completed paper-valid $Profile (5 folds) ==========" -ForegroundColor Yellow
      continue
    }
  }
  $Arguments = @("$Project\scripts\task2\run_p2_q10_npam.py", "--profile", $Profile, "--outcome_table", $env:DRE_TASK2_OUTCOME_TABLE, "--feature_cache", $env:DRE_TASK1_FEATURE_CACHE_PATH, "--p2_checkpoint_root", $env:DRE_TASK1_P2_CHECKPOINT_ROOT, "--p2_runtime_root", $env:DRE_TASK1_P2_RUNTIME_ROOT, "--p2_training_manifest", $env:DRE_TASK1_P2_TRAINING_MANIFEST, "--fold_manifest", $env:DRE_TASK2_FOLD_MANIFEST, "--exclusion_manifest", $env:DRE_TASK2_EXCLUSION_MANIFEST, "--protocol", "outer_cv", "--outer_folds", "5", "--batch_size", "8", "--seed", $Seed, "--resume", "--output_dir", $ProfileOutput, "--strict")
  if ($Profile -in @("C4_TARGET_NETWORK", "C5_FULL")) { $Arguments += @("--graph_cache", "$env:DRE_TASK2_OUTPUT_DIR\graph_cache\phase_functional_graphs.pkl") }
  & $PythonExe @Arguments
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  Write-Host "========== Completed $Profile seed $Seed ==========" -ForegroundColor Green
}}
Write-Host "`n========== Summarizing C0-C6 ==========" -ForegroundColor Cyan
& $PythonExe "$Project\scripts\task2\summarize_p2_q10_npam.py" --root_dir $env:DRE_TASK2_OUTPUT_DIR --bootstrap_repeats 2000 --seed 42
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
