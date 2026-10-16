param([ValidateSet('support','evaluate','report','verify')][string]$Stage)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$taskCode='D:\m8_physiology_discovery_selection_seed42_v2'
$taskRuntime='D:\m8_physiology_discovery_selection_seed42_v2_runtime'
$taskStamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
$taskLock=[System.IO.File]::Open((Join-Path $taskRuntime 'remaining_coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
try {
    $taskExisting=Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match 'm8_physiology_discovery_selection_seed42_v2' -and $_.CommandLine -match '(train|support_control|evaluate|report|verify_results)\.py'}
    if($taskExisting){throw 'Existing scientific worker: do not duplicate'}
    $taskFile=@{support='support_control.py';evaluate='evaluate.py';report='report.py';verify='verify_results.py'}[$Stage]
    $taskExe=if($Stage -in @('report','verify')){'E:\Anaconda\envs\persist_stable_251\python.exe'}else{'E:\Anaconda\python.exe'}
    $taskCommand="import sys,psutil;psutil.Process().cpu_affinity([16]);import numpy,runpy;sys.path.insert(0,r'$taskCode');runpy.run_path(r'$taskCode\$taskFile',run_name='__main__')"
    $taskArgs=@('-u','-X','faulthandler','-c',('"'+$taskCommand+'"'))
    $taskWorker=Start-Process -FilePath $taskExe -ArgumentList $taskArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRuntime ($Stage+'_'+$taskStamp+'.log')) -RedirectStandardError (Join-Path $taskRuntime ($Stage+'_'+$taskStamp+'.err'))
    $null=$taskWorker.Handle
    Write-Output ('STAGE_STARTED '+$Stage+' PID '+$taskWorker.Id+' LOGSTAMP '+$taskStamp)
    # Keep the authenticated foreground session alive and emit ASCII keepalives.
    while(-not $taskWorker.HasExited){Start-Sleep -Seconds 20;$taskWorker.Refresh();Write-Output ('STAGE_RUNNING '+$Stage+' '+(Get-Date -Format s))}
    $taskWorker.WaitForExit()
    if($null -ne $taskWorker.ExitCode -and $taskWorker.ExitCode -ne 0){throw ('Stage failed; preserve files and inspect '+$Stage+' exit '+$taskWorker.ExitCode)}
    $taskEvidence=@{support=@('SUPPORT30_STATUS.json','ALL5_SUPPORT_MATCHED_RAW_MODELS_FROZEN');evaluate=@('OUTER_STATUS.json','ALL105_FROZEN_SCORE_CELLS_SEALED');report=@('REPORT_VERIFICATION.json','PASS');verify=@('INDEPENDENT_RESULTS_VERIFICATION.json','PASS')}[$Stage]
    $taskStatus=Get-Content -LiteralPath (Join-Path "$taskRuntime\public" $taskEvidence[0]) -Raw | ConvertFrom-Json
    if($taskStatus.status -ne $taskEvidence[1]){throw 'Required stage-completion evidence absent'}
    Write-Output ('STAGE_COMPLETE '+$Stage)
} finally {$taskLock.Dispose()}
