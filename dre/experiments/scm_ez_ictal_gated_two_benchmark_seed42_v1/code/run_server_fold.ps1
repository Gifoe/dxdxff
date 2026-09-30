param([Parameter(Mandatory = $true)][ValidateRange(1, 5)][int]$Fold)

$python = 'E:\DRE-nips\new-pipeline\.venv\Scripts\python.exe'
$root = 'E:\DRE-nips\new-pipeline\7-11\scm_ez_ictal_gated_two_benchmark_seed42_v1'
$runtime = 'C:\scm_ez_seed42_runtime'
$log = Join-Path $runtime ("fold{0}.log" -f $Fold)
$err = Join-Path $runtime ("fold{0}.err" -f $Fold)

& $python (Join-Path $root 'code\run_scm.py') run-fold `
  --runtime $runtime `
  --protocol (Join-Path $root 'PROTOCOL_LOCK.json') `
  --cache (Join-Path $runtime 'ictal_spectral') `
  --manifest 'E:\DRE-nips\new-pipeline\7-11\task1_confirmatory\audit\fixed_partition_manifest.csv' `
  --fold $Fold 1> $log 2> $err
exit $LASTEXITCODE
