$ErrorActionPreference = "Stop"
. "$PSScriptRoot\..\..\configs\paths.server.ps1"
& $PythonExe "$PSScriptRoot\build_trace_dre_lite_cache.py" `
  --raw-cache $env:DRE_TASK1_RAW_CACHE_PATH --outcome-table $env:DRE_TASK2_OUTCOME_TABLE `
  --exclusion-manifest $env:DRE_TASK2_EXCLUSION_MANIFEST --cache-dir $env:DRE_TASK2_TRACE_DRE_CACHE_DIR --resume
& $PythonExe "$PSScriptRoot\build_trace_dre_lite_cache.py" --raw-cache $env:DRE_TASK1_RAW_CACHE_PATH `
  --outcome-table $env:DRE_TASK2_OUTCOME_TABLE --exclusion-manifest $env:DRE_TASK2_EXCLUSION_MANIFEST `
  --cache-dir $env:DRE_TASK2_TRACE_DRE_CACHE_DIR --verify-only
