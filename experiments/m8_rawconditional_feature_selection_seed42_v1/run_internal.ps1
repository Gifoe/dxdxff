$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$taskCode='D:\m8_rawconditional_feature_selection_seed42_v1'
$taskRoot='D:\m8_rawconditional_feature_selection_seed42_v1_runtime'
$taskStamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Path $taskRoot -Force | Out-Null
$taskLock=[System.IO.File]::Open((Join-Path $taskRoot 'internal_coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Run-Worker([string]$Script,[bool]$CPU=$false){
    $taskExe=if($CPU){'E:\Anaconda\envs\persist_stable_251\python.exe'}else{'E:\Anaconda\python.exe'}
    $taskAffinity=if($CPU){'import psutil;psutil.Process().cpu_affinity([16]);'}else{''}
    $taskProgram="import sys;${taskAffinity}import numpy,runpy;sys.path.insert(0,r'$taskCode');runpy.run_path(r'$taskCode\$Script.py',run_name='__main__')"
    $taskArgs=@('-u','-X','faulthandler','-c',('"'+$taskProgram+'"'))
    $taskLog=Join-Path $taskRoot ($Script+'_'+$taskStamp+'.log');$taskErr=Join-Path $taskRoot ($Script+'_'+$taskStamp+'.err')
    if((Test-Path -LiteralPath $taskLog) -or (Test-Path -LiteralPath $taskErr)){throw 'Preserve existing logs'}
    $taskWorker=Start-Process -FilePath $taskExe -ArgumentList $taskArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput $taskLog -RedirectStandardError $taskErr
    $null=$taskWorker.Handle
    Write-Output ('INTERNAL_STAGE_STARTED '+$Script+' PID '+$taskWorker.Id+' STAMP '+$taskStamp)
    while(-not $taskWorker.HasExited){Start-Sleep -Seconds 20;$taskWorker.Refresh();Write-Output ('INTERNAL_STAGE_RUNNING '+$Script+' '+(Get-Date -Format s))}
    $taskWorker.WaitForExit()
    if($null -ne $taskWorker.ExitCode -and $taskWorker.ExitCode -ne 0){throw ('Stage failed; inspect preserved logs: '+$Script)}
}
try {
    $taskOther=Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match 'm8_rawconditional_feature_selection_seed42_v1'}
    if($taskOther){throw 'Duplicate internal worker prohibited'}
    Run-Worker 'admit' $true
    if((Get-Content -LiteralPath "$taskRoot\public\ADMISSION_STATUS.json" -Raw|ConvertFrom-Json).status -ne 'PASS'){throw 'Admission absent'}
    if(-not (Test-Path -LiteralPath "$taskRoot\ENCODER_REGISTRY_PRIVATE.pkl")){Run-Worker 'prepare_encoders'}
    if((Get-Content -LiteralPath "$taskRoot\public\ENCODER_STATUS.json" -Raw|ConvertFrom-Json).status -ne 'ALL20_INNER_CROSSFIT_ENCODERS_COMPLETE'){throw 'Encoder gate incomplete'}
    if(-not (Test-Path -LiteralPath "$taskRoot\SELECTED_FEATURES_PRIVATE.json")){Run-Worker 'select_conditional' $true}
    if((Get-Content -LiteralPath "$taskRoot\public\SCREEN_STATUS.json" -Raw|ConvertFrom-Json).status -ne 'COMPLETE_INTERNAL_RAW_CONDITIONAL_SCREEN'){throw 'Screen incomplete'}
    Write-Output 'INTERNAL_SCREEN_COMPLETE_NO_NEW_OUTER_OR_FINAL_MODEL_CLAIM'
} finally {$taskLock.Dispose()}
