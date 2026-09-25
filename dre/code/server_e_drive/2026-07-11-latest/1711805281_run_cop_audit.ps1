$ErrorActionPreference="Stop"
$Project=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
. "$Project\scripts\task2\ensure_npam_protocol.ps1"
$PythonExe=if($env:DRE_PYTHON_EXE){$env:DRE_PYTHON_EXE}else{"python"}
$FeatureCache=if($env:DRE_TASK2_COP_FEATURE_CACHE){$env:DRE_TASK2_COP_FEATURE_CACHE}else{$env:DRE_TASK1_FEATURE_CACHE_PATH}
& $PythonExe "$Project\scripts\task2\run_cop_outcome.py" --profile O0_FEATURE_PHENOTYPE --feature_cache $FeatureCache --raw_cache $env:DRE_TASK1_RAW_CACHE_PATH --outcome_table $env:DRE_TASK2_OUTCOME_TABLE --fold_manifest $env:DRE_TASK2_FOLD_MANIFEST --exclusion_manifest $env:DRE_TASK2_EXCLUSION_MANIFEST --p2_checkpoint_root $env:DRE_TASK1_P2_CHECKPOINT_ROOT --p2_runtime_root $env:DRE_TASK1_P2_RUNTIME_ROOT --p2_training_manifest $env:DRE_TASK1_P2_TRAINING_MANIFEST --output_dir "$env:DRE_TASK2_COP_OUTPUT_DIR\audit" --phenotype_cache_dir $env:DRE_TASK2_COP_PHENOTYPE_CACHE_DIR --audit_only --strict
exit $LASTEXITCODE
