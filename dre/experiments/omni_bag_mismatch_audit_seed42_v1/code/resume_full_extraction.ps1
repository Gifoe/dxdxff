param(
    [string]$Python = 'E:\Anaconda\envs\Benchmark_TTA_Win\python.exe',
    [string]$Experiment = 'D:\dxdxff\dre\experiments\omni_bag_mismatch_audit_seed42_v1',
    [string]$Runtime = 'D:\omni_bag_mismatch_audit_seed42_v1',
    [int]$BatchSize = 8,
    [int]$NumShards = 1,
    [int]$ShardIndex = 0,
    [int]$MaximumAttempts = 400
)
$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'E:\Anaconda\pkgs\numpy-1.24.4-py310hd02465a_0\Lib\site-packages'
$output = Join-Path $Runtime 'train_full_predictions'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$status = Join-Path $Runtime ("supervisor_status_{0}_of_{1}.json" -f $ShardIndex,$NumShards)
$completeMarker = Join-Path $output ("SHARD_{0}_OF_{1}.json" -f $ShardIndex,$NumShards)
function Write-Status([hashtable]$Value) {
    $temporary = $status + '.tmp'
    $Value | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $status -Force
}
for ($attempt=1; $attempt -le $MaximumAttempts; $attempt++) {
    $before = @(Get-ChildItem -LiteralPath $output -Filter '*.npz' -File -ErrorAction SilentlyContinue).Count
    if (Test-Path -LiteralPath $completeMarker) { Write-Status @{status='COMPLETE';completed_edfs=$before;attempts=$attempt-1}; exit 0 }
    $stdout = Join-Path $Runtime ('extract_s{0}_{1:D3}.out.log' -f $ShardIndex,$attempt)
    $stderr = Join-Path $Runtime ('extract_s{0}_{1:D3}.err.log' -f $ShardIndex,$attempt)
    Write-Status @{status='RUNNING';attempt=$attempt;completed_edfs=$before}
    $arguments = @(
        '-u',(Join-Path $Experiment 'code\extract_train_full.py'),
        '--train-waveforms','C:\pc_cnn_seed42_runtime\omni_train',
        '--signal-cache','F:\Omni-iEEG\signal_cache',
        '--official-cnn','F:\Omni-iEEG\a1_net_seed42_runtime\official_source\cnn.py',
        '--checkpoint','F:\Omni-iEEG\a1_net_seed42_runtime\official_cnn_training\best_model.pt',
        '--protocol',(Join-Path $Experiment 'PROTOCOL_LOCK.json'),
        '--output',$output,'--batch-size',[string]$BatchSize,
        '--num-shards',[string]$NumShards,'--shard-index',[string]$ShardIndex
    )
    $process = Start-Process -FilePath $Python -ArgumentList $arguments -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
    $after = @(Get-ChildItem -LiteralPath $output -Filter '*.npz' -File -ErrorAction SilentlyContinue).Count
    if ((Test-Path -LiteralPath $completeMarker) -and $process.ExitCode -eq 0) { Write-Status @{status='COMPLETE';completed_edfs=$after;attempts=$attempt}; exit 0 }
    if ($process.ExitCode -ne 0 -and (Get-Item -LiteralPath $stderr).Length -gt 0) {
        Write-Status @{status='PYTHON_ERROR';attempt=$attempt;completed_edfs=$after;exit_code=$process.ExitCode;stderr=$stderr}; exit 2
    }
    if ($after -le $before) { Write-Status @{status='NO_PROGRESS';attempt=$attempt;completed_edfs=$after;exit_code=$process.ExitCode}; exit 3 }
}
Write-Status @{status='ATTEMPT_LIMIT';completed_edfs=@(Get-ChildItem -LiteralPath $output -Filter '*.npz' -File).Count}
exit 4
