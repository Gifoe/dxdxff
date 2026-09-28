$ErrorActionPreference = 'Stop'
$source = 'E:\DRE-nips\new-pipeline\7-11'
$experiment = Join-Path $source 'drst_pr_dual_reference_spectral_seed42_v1'
$private = 'D:\nips-temp\drst_pr_dual_reference_spectral_seed42_v1'
$env:R1_HLV_SOURCE_ROOT = $source
$env:R1_HLV_WINDOW_CACHE = 'D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl'
$env:R1_HLV_FIXED_MANIFEST = 'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv'
$env:R1_HLV_RUNTIME = 'D:\nips-temp\r1_hlv_ictal_dynamics_seed42_v1'
$env:A1_A2_RUNTIME = 'D:\nips-temp\a1_a2_patient_equal_objective_seed42_v1'
$env:PYTHONPATH = "$source;$source\neuroez_c;$experiment\code"
$python = 'E:\DRE-nips\new-pipeline\.venv\Scripts\python.exe'
$runner = Join-Path $experiment 'code\run_training_grid.py'
$raw = 'D:\nips-temp\neuroez_c_four_center_caches_success_failure_raw_v1\all_window_cache.pkl'
$lock = Join-Path $experiment 'PROTOCOL_LOCK.json'
$args = @($runner, '--raw-cache', $raw, '--runtime', $private, '--lock', $lock)
$process = Start-Process -FilePath $python -ArgumentList $args -RedirectStandardOutput (Join-Path $private 'training_grid.log') -RedirectStandardError (Join-Path $private 'training_grid.err') -WindowStyle Hidden -PassThru
$process.Id | Set-Content -LiteralPath (Join-Path $private 'training_grid.pid')
Write-Output "DRST_GRID_PID=$($process.Id)"
