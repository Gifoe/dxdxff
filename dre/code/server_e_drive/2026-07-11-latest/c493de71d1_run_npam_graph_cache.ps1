$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonExe = if ($env:DRE_PYTHON_EXE) { $env:DRE_PYTHON_EXE } else { "python" }
& $PythonExe "$Project\scripts\task2\precompute_phase_functional_graphs.py" --raw_cache $env:DRE_TASK1_RAW_CACHE_PATH --feature_cache $env:DRE_TASK1_FEATURE_CACHE_PATH --exclusion_manifest $env:DRE_TASK2_EXCLUSION_MANIFEST --graph_source raw_aec_spearman --band_low 30 --band_high 80 --topk 8 --output_dir "$env:DRE_TASK2_OUTPUT_DIR\graph_cache" --strict
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
