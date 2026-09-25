$ErrorActionPreference = "Stop"
. "$PSScriptRoot\..\..\configs\paths.server.ps1"
& $PythonExe "$PSScriptRoot\run_true_ez_nez_dualset.py" `
 --cache-dir $env:DRE_TASK2_TRACE_DRE_CACHE_DIR --fold-manifest $env:DRE_TASK2_FOLD_MANIFEST `
 --output-dir $env:DRE_TASK2_TRUE_DUALSET_OUTPUT_DIR --seed 42 --folds all --device cuda `
 --batch-size 4 --gradient-accumulation-steps 2 --fixed-epochs 15 `
 --learning-rate 3e-4 --weight-decay 1e-3 --gradient-clip 1.0 --ema-decay 0.98 --rank-weight 0.05 `
 --n-deterministic-views 4 --max-seizures-per-view 2 --max-ez-channels-per-view 24 `
 --max-nez-channels-per-view 40 --max-windows-per-phase 4 --strict --no-resume-old-checkpoint
