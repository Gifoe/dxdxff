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
$raw = 'D:\nips-temp\neuroez_c_four_center_caches_success_failure_raw_v1\all_window_cache.pkl'
$lock = Join-Path $experiment 'PROTOCOL_LOCK.json'
$amendment = Join-Path $experiment 'PROTOCOL_AMENDMENT_02_S4_FEASIBILITY.json'
$freeze = Join-Path $experiment 'S4_SCORE_FREEZE_AUDIT.json'

& $python (Join-Path $experiment 'code\select_s4_feasibility.py') --runtime $private --lock $lock --amendment $amendment --output (Join-Path $experiment 'FIT_HYPERPARAM_SELECTION.csv')
if ($LASTEXITCODE -ne 0) { throw "FIT-only S4 checkpoint selection failed with exit $LASTEXITCODE" }
& $python (Join-Path $experiment 'code\freeze_target_scores.py') --s4-feasibility --raw-cache $raw --runtime $private --lock $lock --amendment $amendment --output $freeze
if ($LASTEXITCODE -ne 0) { throw "S4 target-score freeze failed with exit $LASTEXITCODE" }
$audit = Get-Content -LiteralPath $freeze -Raw | ConvertFrom-Json
if (-not $audit.pass -or $audit.n_score_files -ne 5) { throw 'S4 target-score freeze audit did not pass' }
& $python (Join-Path $experiment 'code\evaluate_s4_feasibility.py') --runtime $private --lock $lock --amendment $amendment --freeze-audit $freeze --output $experiment --srgi-runtime 'D:\nips-temp\seizure_resolved_geometry_identifiability_seed42_v1'
if ($LASTEXITCODE -ne 0) { throw "S4 matched evaluation failed with exit $LASTEXITCODE" }
Write-Output 'S4_FEASIBILITY_FINALIZATION_PASS'
