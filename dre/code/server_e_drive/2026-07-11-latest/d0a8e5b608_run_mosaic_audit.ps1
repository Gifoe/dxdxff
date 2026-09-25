$ErrorActionPreference = "Stop"
. "$PSScriptRoot\..\..\configs\paths.server.ps1"
& $PythonExe "$PSScriptRoot\run_mosaic_outcome.py" `
  --feature_cache $env:DRE_TASK1_FEATURE_CACHE_PATH --raw_cache $env:DRE_TASK1_RAW_CACHE_PATH `
  --outcome_table $env:DRE_TASK2_OUTCOME_TABLE --original_fold_manifest $env:DRE_TASK2_FOLD_MANIFEST `
  --exclusion_manifest $env:DRE_TASK2_EXCLUSION_MANIFEST --p2_checkpoint_root $env:DRE_TASK1_P2_CHECKPOINT_ROOT `
  --p2_runtime_root $env:DRE_TASK1_P2_RUNTIME_ROOT --p2_training_manifest $env:DRE_TASK1_P2_TRAINING_MANIFEST `
  --cache_dir $env:DRE_TASK2_MOSAIC_CACHE_DIR --output_dir $env:DRE_TASK2_MOSAIC_OUTPUT_DIR `
  --pilot_size 16 --patients_per_center 4 --seed 42 --strict --resume --audit_only
