param([string]$RunTag='01',[int]$OldWorkerId=40552)
$ErrorActionPreference='Stop'
$taskCode='D:\m8_physiology_feature_discovery_seed42_v1'
$taskRuntime='D:\m8_physiology_feature_discovery_seed42_v1_runtime'
$stdout=Join-Path $taskRuntime ('integrated_'+$RunTag+'.log')
$stderr=Join-Path $taskRuntime ('integrated_'+$RunTag+'.err')
if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) { throw 'New run tag required; preserve previous logs' }
$duplicate=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'train_integrated.py' }
if ($duplicate) { throw 'Existing integrated worker: no duplicate allowed' }
while ($true) {
    $old=Get-CimInstance Win32_Process -Filter "ProcessId=$OldWorkerId"
    if ($old -and $old.CommandLine -notmatch [regex]::Escape($taskCode+'\train_old7_m8.py')) { throw 'PID identity changed; inspect before proceeding' }
    $state=Get-Content -Raw -LiteralPath ($taskRuntime+'\old7_matched_M8\public\STATUS.json') | ConvertFrom-Json
    if ($state.status -eq 'ALL80_MATCHED_OLD7_D1_AND_M8_DEVELOPMENT_CELLS_COMPLETE' -and -not $old) { break }
    if (-not $old) { throw 'Old7 worker missing before complete; diagnose rather than restart' }
    Write-Output ('QUEUE_WAIT_VERIFIED_OLD7_LIVE completed='+$state.completed_model_pairs+' '+(Get-Date -Format o))
    Start-Sleep -Seconds 20
}
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1'
$worker=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u',($taskCode+'\train_integrated.py')) -WorkingDirectory $taskCode -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Write-Output ('INTEGRATED_WORKER_STARTED PID='+$worker.Id)
while (-not $worker.HasExited) {
    Write-Output ('INTEGRATED_WORKER_LIVE PID='+$worker.Id+' '+(Get-Date -Format o))
    Start-Sleep -Seconds 20
    $worker.Refresh()
}
$worker.WaitForExit()
Write-Output ('INTEGRATED_WORKER_TERMINAL exit='+$worker.ExitCode)
if ($worker.ExitCode -ne 0) { throw 'Integrated worker failed: preserve logs and inspect; no automatic retries' }
$final=Get-Content -Raw -LiteralPath ($taskRuntime+'\integrated_M8\public\MODEL_FREEZE_AUDIT.json') | ConvertFrom-Json
if ($final.status -ne 'ALL45_LOGICAL_INTEGRATED_MODELS_FROZEN_BEFORE_OUTER') { throw 'Final freeze not proven' }
Write-Output 'INTEGRATED_TRAIN_VALIDATION_COMPLETE_NO_OUTER'
