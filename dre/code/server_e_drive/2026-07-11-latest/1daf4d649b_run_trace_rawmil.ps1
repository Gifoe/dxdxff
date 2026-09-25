$ErrorActionPreference = "Stop"
. "$PSScriptRoot\..\..\configs\paths.server.ps1"

& $PythonExe "$PSScriptRoot\run_trace_rawmil.py" `
  --raw_cache $env:DRE_TASK1_RAW_CACHE_PATH `
  --outcome_table $env:DRE_TASK2_OUTCOME_TABLE `
  --fold_manifest $env:DRE_TASK2_FOLD_MANIFEST `
  --exclusion_manifest $env:DRE_TASK2_EXCLUSION_MANIFEST `
  --cache_dir $env:DRE_TASK2_TRACE_CACHE_DIR `
  --output_dir $env:DRE_TASK2_TRACE_OUTPUT_DIR `
  --seed 42 --folds all --device cuda --batch_size 4 --max_epochs 15 --patience 3 `
  --learning_rate 3e-4 --weight_decay 1e-3 --gradient_clip_norm 1.0 --gradient_accumulation_steps 2 `
  --encoder-window-batch-size-train 1024 --encoder-window-batch-size-eval 4096 `
  --gradient-checkpoint-encoder --auto-window-batch-backoff --min-window-batch-size 128 `
  --vram-budget-fraction 0.75 --no-compile-encoder `
  --strict --recompute-training --no-resume
