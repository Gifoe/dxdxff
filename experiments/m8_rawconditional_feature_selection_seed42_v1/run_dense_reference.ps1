$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$taskCode='D:\m8_rawconditional_feature_selection_seed42_v1'
$taskRoot='D:\m8_rawconditional_feature_selection_seed42_v1_dense_reference_runtime'
$taskStamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Path $taskRoot -Force | Out-Null
$taskLock=[System.IO.File]::Open((Join-Path $taskRoot 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
try {
    $taskOther=Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match '(select_conditional|run_dense_reference)\.py' -and $_.CommandLine -match 'm8_rawconditional_feature_selection_seed42_v1'}
    if($taskOther){throw 'Existing scientific screen worker; do not duplicate'}
    if(Test-Path -LiteralPath "$taskRoot\SELECTED_FEATURES_PRIVATE.json"){throw 'Reference screen exists; verify instead of replaying'}
    $taskProgram="import sys,psutil;psutil.Process().cpu_affinity([16]);import numpy,runpy;sys.path.insert(0,r'$taskCode');runpy.run_path(r'$taskCode\run_dense_reference.py',run_name='__main__')"
    $taskArgs=@('-u','-X','faulthandler','-c',('"'+$taskProgram+'"'))
    $taskWorker=Start-Process -FilePath 'E:\Anaconda\envs\persist_stable_251\python.exe' -ArgumentList $taskArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput "$taskRoot\screen_$taskStamp.log" -RedirectStandardError "$taskRoot\screen_$taskStamp.err"
    $null=$taskWorker.Handle
    Write-Output ('DENSE_REFERENCE_STARTED PID '+$taskWorker.Id+' STAMP '+$taskStamp)
    while(-not $taskWorker.HasExited){Start-Sleep -Seconds 20;$taskWorker.Refresh();Write-Output ('DENSE_REFERENCE_RUNNING '+(Get-Date -Format s))}
    $taskWorker.WaitForExit()
    if($null -ne $taskWorker.ExitCode -and $taskWorker.ExitCode -ne 0){throw 'Reference screen failed; preserve and diagnose'}
    if((Get-Content -LiteralPath "$taskRoot\public\SCREEN_STATUS.json" -Raw|ConvertFrom-Json).status -ne 'COMPLETE_INTERNAL_RAW_CONDITIONAL_SCREEN'){throw 'Reference completion gate absent'}
    Write-Output 'DENSE_REFERENCE_EXECUTION_COMPLETE_REQUIRES_VERIFICATION'
} finally {$taskLock.Dispose()}
