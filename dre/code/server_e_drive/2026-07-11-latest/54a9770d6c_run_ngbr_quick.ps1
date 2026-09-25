$ErrorActionPreference="Stop"
$Project=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
. "$Project\scripts\task2\ensure_npam_protocol.ps1"
$PythonExe=if($env:DRE_PYTHON_EXE){$env:DRE_PYTHON_EXE}else{"python"}
& $PythonExe "$Project\scripts\task2\run_ngbr_full.py" --feature_cache $env:DRE_TASK1_FEATURE_CACHE_PATH --raw_cache $env:DRE_TASK1_RAW_CACHE_PATH --outcome_table $env:DRE_TASK2_OUTCOME_TABLE --fold_manifest $env:DRE_TASK2_FOLD_MANIFEST --exclusion_manifest $env:DRE_TASK2_EXCLUSION_MANIFEST --p2_checkpoint_root $env:DRE_TASK1_P2_CHECKPOINT_ROOT --p2_runtime_root $env:DRE_TASK1_P2_RUNTIME_ROOT --p2_training_manifest $env:DRE_TASK1_P2_TRAINING_MANIFEST --biomarker_cache_dir $env:DRE_TASK2_NGBR_CACHE_DIR --output_dir "$env:DRE_TASK2_NGBR_ROOT\quick\seed_42" --seed 42 --bootstrap_repeats 100 --max_outer_folds 1 --max_patients 30 --disable_hfo --strict --resume
exit $LASTEXITCODE
