$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$taskCode='D:\m8_conditional_readout_validation_seed42_v1'
$taskRoot='D:\m8_conditional_readout_validation_seed42_v1_runtime'
$taskStamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Path $taskRoot -Force | Out-Null
$taskLock=[System.IO.File]::Open((Join-Path $taskRoot 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
try {
    $taskOther=Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match '(run_readouts|readout_report)\.py' -and $_.CommandLine -match 'm8_conditional_readout_validation_seed42_v1'}
    if($taskOther){throw 'A readout worker is already active'}
    foreach($taskStage in @('run_readouts.py','readout_report.py')) {
        $taskProgram="import sys,psutil;psutil.Process().cpu_affinity([16]);sys.path.insert(0,r'$taskCode');import runpy;runpy.run_path(r'$taskCode\$taskStage',run_name='__main__')"
        $taskArgs=@('-u','-X','faulthandler','-c',('"'+$taskProgram+'"'))
        $taskWorker=Start-Process -FilePath 'E:\Anaconda\envs\persist_stable_251\python.exe' -ArgumentList $taskArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput "$taskRoot\${taskStage}_$taskStamp.log" -RedirectStandardError "$taskRoot\${taskStage}_$taskStamp.err"
        $null=$taskWorker.Handle
        Write-Output ('INTERNAL_READOUT_STAGE_STARTED '+$taskStage+' PID '+$taskWorker.Id+' STAMP '+$taskStamp)
        while(-not $taskWorker.HasExited){Start-Sleep -Seconds 20;$taskWorker.Refresh();Write-Output ('INTERNAL_READOUT_STAGE_RUNNING '+$taskStage+' '+(Get-Date -Format s))}
        $taskWorker.WaitForExit()
        if($null -ne $taskWorker.ExitCode -and $taskWorker.ExitCode -ne 0){throw ('Readout stage failed: '+$taskStage+'; preserve and inspect, do not blindly retry')}
    }
    if((Get-Content -LiteralPath "$taskRoot\public\REPORT_STATUS.json" -Raw|ConvertFrom-Json).status -ne 'COMPLETE_VERIFIED_BOUNDED_INTERNAL_READOUT_TRANSFER'){throw 'Readout completion gate absent'}
    Write-Output 'BOUNDED_INTERNAL_READOUT_TRANSFER_COMPLETE_NO_OUTER'
} finally {$taskLock.Dispose()}
