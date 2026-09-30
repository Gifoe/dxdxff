param(
    [ValidateSet('train','test')][string]$Mode,
    [int]$BatchSize = 8,
    [int]$MaximumAttempts = 200
)
$ErrorActionPreference = 'Stop'
$python = 'E:\Anaconda\envs\Benchmark_TTA_Win\python.exe'
$experiment = 'E:\DRE-nips\new-pipeline\7-11\omni_pr_residual_seed42_v1'
$runtime = 'C:\omni_pr_residual_seed42_runtime'
$source = if ($Mode -eq 'train') { 'C:\pc_cnn_seed42_runtime\omni_train' } else { 'F:\Omni-iEEG\a1_net_seed42_runtime\features_test_full' }
$output = Join-Path $runtime ($Mode + '_embeddings')
$expected = if ($Mode -eq 'train') { 296 } else { 237 }
$status = Join-Path $runtime ($Mode + '_embedding_supervisor_status.json')
$noProgress = 0

function Write-Status([hashtable]$Value) {
    $temporary = $status + '.tmp'
    $Value | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $status -Force
}

for ($attempt = 1; $attempt -le $MaximumAttempts; $attempt++) {
    $before = @(Get-ChildItem -LiteralPath $output -File -Filter '*.npz' -ErrorAction SilentlyContinue).Count
    if ($before -eq $expected) {
        Write-Status @{status='COMPLETE'; mode=$Mode; completed_files=$before; attempts=$attempt-1}
        exit 0
    }
    $stdout = Join-Path $runtime ('{0}_embedding_ps_attempt_{1:D3}.log' -f $Mode,$attempt)
    $stderr = Join-Path $runtime ('{0}_embedding_ps_attempt_{1:D3}.err' -f $Mode,$attempt)
    Write-Status @{status='RUNNING'; mode=$Mode; attempt=$attempt; completed_files=$before}
    $arguments = @(
        (Join-Path $experiment 'code\extract_embedding_cache.py'), '--mode', $Mode,
        '--source', $source,
        '--official-cnn', 'F:\Omni-iEEG\a1_net_seed42_runtime\official_source\cnn.py',
        '--checkpoint', 'F:\Omni-iEEG\a1_net_seed42_runtime\official_cnn_training\best_model.pt',
        '--protocol', (Join-Path $experiment 'PROTOCOL_LOCK.json'),
        '--output', $output, '--public-audit', $experiment,
        '--batch-size', [string]$BatchSize
    ) -join ' '
    $process = Start-Process -FilePath $python -ArgumentList $arguments -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
    $after = @(Get-ChildItem -LiteralPath $output -File -Filter '*.npz' -ErrorAction SilentlyContinue).Count
    if ($after -eq $expected -and $process.ExitCode -eq 0) {
        Write-Status @{status='COMPLETE'; mode=$Mode; completed_files=$after; attempts=$attempt}
        exit 0
    }
    if ((Get-Item -LiteralPath $stderr).Length -gt 0) {
        Write-Status @{status='PYTHON_ERROR'; mode=$Mode; attempt=$attempt; completed_files=$after; exit_code=$process.ExitCode}
        exit 2
    }
    if ($after -eq $before) { $noProgress++ } else { $noProgress = 0 }
    Write-Status @{status='NATIVE_EXIT_RESUMABLE'; mode=$Mode; attempt=$attempt; before=$before; after=$after; consecutive_no_progress=$noProgress; exit_code=$process.ExitCode}
    if ($noProgress -ge 3) { exit 3 }
    Start-Sleep -Seconds 2
}
Write-Status @{status='ATTEMPT_LIMIT'; mode=$Mode; completed_files=@(Get-ChildItem -LiteralPath $output -File -Filter '*.npz').Count}
exit 4
