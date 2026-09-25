$ErrorActionPreference = "Stop"
. "$PSScriptRoot\..\..\configs\paths.server.ps1"

& $PythonExe "$PSScriptRoot\run_trace_rawmil.py" `
  --raw_cache $env:DRE_TASK1_RAW_CACHE_PATH `
  --outcome_table $env:DRE_TASK2_OUTCOME_TABLE `
  --fold_manifest $env:DRE_TASK2_FOLD_MANIFEST `
  --exclusion_manifest $env:DRE_TASK2_EXCLUSION_MANIFEST `
  --cache_dir $env:DRE_TASK2_TRACE_CACHE_DIR `
  --output_dir $env:DRE_TASK2_TRACE_OUTPUT_DIR `
  --seed 42 --strict --resume --audit_only
