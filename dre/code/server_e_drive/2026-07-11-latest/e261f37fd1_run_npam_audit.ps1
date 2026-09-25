$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
. "$Project\scripts\task2\ensure_npam_protocol.ps1"
$PythonExe = if ($env:DRE_PYTHON_EXE) { $env:DRE_PYTHON_EXE } else { "python" }
& $PythonExe "$Project\scripts\task2\audit_p2_q10_npam_inputs.py" --outcome_table $env:DRE_TASK2_OUTCOME_TABLE --feature_cache $env:DRE_TASK1_FEATURE_CACHE_PATH --raw_cache $env:DRE_TASK1_RAW_CACHE_PATH --p2_checkpoint_root $env:DRE_TASK1_P2_CHECKPOINT_ROOT --p2_runtime_root $env:DRE_TASK1_P2_RUNTIME_ROOT --fold_manifest $env:DRE_TASK2_FOLD_MANIFEST --exclusion_manifest $env:DRE_TASK2_EXCLUSION_MANIFEST --output_dir "$env:DRE_TASK2_OUTPUT_DIR\audit" --strict
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
