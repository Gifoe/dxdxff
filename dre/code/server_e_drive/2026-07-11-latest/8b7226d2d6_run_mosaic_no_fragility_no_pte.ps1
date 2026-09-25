$ErrorActionPreference = "Stop"
. "$PSScriptRoot\..\..\configs\paths.server.ps1"

# Fast paired ablation for the current no-Fragility baseline. It cannot isolate
# PTE from Fragility, but it isolates the incremental value of PTE conditional
# on the no-Fragility model.
$Arguments = @(
    "$PSScriptRoot\run_mosaic_outcome.py",
    "--feature_cache", $env:DRE_TASK1_FEATURE_CACHE_PATH,
    "--raw_cache", $env:DRE_TASK1_RAW_CACHE_PATH,
    "--outcome_table", $env:DRE_TASK2_OUTCOME_TABLE,
    "--original_fold_manifest", $env:DRE_TASK2_FOLD_MANIFEST,
    "--exclusion_manifest", $env:DRE_TASK2_EXCLUSION_MANIFEST,
    "--p2_checkpoint_root", $env:DRE_TASK1_P2_CHECKPOINT_ROOT,
    "--p2_runtime_root", $env:DRE_TASK1_P2_RUNTIME_ROOT,
    "--p2_training_manifest", $env:DRE_TASK1_P2_TRAINING_MANIFEST,
    "--cache_dir", $env:DRE_TASK2_MOSAIC_CACHE_DIR,
    "--output_dir", $env:DRE_TASK2_MOSAIC_NO_FRAGILITY_NO_PTE_OUTPUT_DIR,
    "--pilot_size", "16", "--patients_per_center", "4",
    "--batch_size", "32", "--seed", "42", "--bootstrap_repeats", "2000",
    "--n_jobs", "8", "--biomarker_workers", "4", "--strict", "--resume",
    "--disable_fragility", "--disable_pte"
)
if ($env:DRE_TASK1_P2_OOF_CHANNEL_LEDGER) { $Arguments += @("--p2_oof_channel_ledger", $env:DRE_TASK1_P2_OOF_CHANNEL_LEDGER) }
if ($env:DRE_TASK2_MOSAIC_FRAGILITY_CACHE_SOURCE) { $Arguments += @("--fragility_trajectory_cache_dir", $env:DRE_TASK2_MOSAIC_FRAGILITY_CACHE_SOURCE) }
& $PythonExe @Arguments
