param([Parameter(Mandatory=$true)][ValidateRange(1,5)][int]$Fold)
$ErrorActionPreference = 'Continue'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$python = 'E:\DRE-nips\new-pipeline\.venv\Scripts\python.exe'
$script = 'E:\DRE-nips\new-pipeline\7-11\prcd_ez_ictal_gated_two_benchmark_seed42_v1\code\run_ictal.py'
$runtime = 'C:\prcd_ez_seed42_runtime'
$protocol = 'E:\DRE-nips\new-pipeline\7-11\prcd_ez_ictal_gated_two_benchmark_seed42_v1\PROTOCOL_LOCK.json'
$manifest = 'E:\DRE-nips\new-pipeline\7-11\task1_confirmatory\audit\fixed_partition_manifest.csv'
$log = Join-Path $runtime ("fold{0}_active.log" -f $Fold)
& $python $script run-fold --fold $Fold --runtime $runtime --protocol $protocol --manifest $manifest 1> $log 2> ($log + '.err')
exit $LASTEXITCODE
