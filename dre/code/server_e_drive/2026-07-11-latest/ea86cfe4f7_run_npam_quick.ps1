param([string]$Profile = "C2_P2_TARGET_CONCORDANCE")
$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
. "$Project\scripts\task2\ensure_npam_protocol.ps1"
$PythonExe = if ($env:DRE_PYTHON_EXE) { $env:DRE_PYTHON_EXE } else { "python" }
$Arguments = @("$Project\scripts\task2\run_p2_q10_npam.py", "--profile", $Profile, "--outcome_table", $env:DRE_TASK2_OUTCOME_TABLE, "--feature_cache", $env:DRE_TASK1_FEATURE_CACHE_PATH, "--p2_checkpoint_root", $env:DRE_TASK1_P2_CHECKPOINT_ROOT, "--p2_runtime_root", $env:DRE_TASK1_P2_RUNTIME_ROOT, "--fold_manifest", $env:DRE_TASK2_FOLD_MANIFEST, "--exclusion_manifest", $env:DRE_TASK2_EXCLUSION_MANIFEST, "--protocol", "quick", "--max_outer_folds", "1", "--max_patients", "20", "--stage_a_epochs", "2", "--stage_b_epochs", "1", "--bootstrap_repeats", "50", "--batch_size", "8", "--seed", "42", "--resume", "--output_dir", "$env:DRE_TASK2_OUTPUT_DIR\quick\$Profile")
if ($Profile -in @("C4_TARGET_NETWORK", "C5_FULL", "M1_NETWORK_STATS", "M2_PHASE_NETWORK", "M3_GRAPH_RESIDUAL", "M4_GRAPH_STABILITY", "M5_LIMITED_FINETUNE")) { $Arguments += @("--graph_cache", "$env:DRE_TASK2_OUTPUT_DIR\graph_cache\phase_functional_graphs.pkl") }
& $PythonExe @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
