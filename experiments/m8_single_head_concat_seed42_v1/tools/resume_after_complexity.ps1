$ErrorActionPreference='Stop'
$taskCode='D:\m8_single_head_concat_seed42_v1'
$taskRuntime='D:\m8_single_head_concat_seed42_v1_runtime'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
$taskPython='E:\Anaconda\python.exe';$taskCPU='E:\Anaconda\envs\persist_stable_251\python.exe'
$taskLog=Join-Path $taskRuntime 'post_complexity01.log';$taskErr=Join-Path $taskRuntime 'post_complexity01.err'
if((Test-Path -LiteralPath $taskLog) -or (Test-Path -LiteralPath $taskErr)){throw 'Preserve previous logs.'}
$taskDup=Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'm8_single_head_concat_seed42_v1' }
if($taskDup){throw 'Active worker exists.'}
foreach($taskScript in @('tools\complexity_recovery.py','evaluate.py')){
    & $taskPython -u -X faulthandler (Join-Path $taskCode $taskScript) 1>> $taskLog 2>> $taskErr
    if($LASTEXITCODE -ne 0){throw ('Frozen stage failed: '+$taskScript)}
}
foreach($taskScript in @('report.py','verify.py')){
    $taskArg=if($taskScript -eq 'report.py'){",'outer'"}else{''}
    $taskCommand="import sys,runpy; import psutil; psutil.Process().cpu_affinity([16]); sys.argv=[r'$taskCode\$taskScript'$taskArg]; sys.path.insert(0,r'$taskCode'); runpy.run_path(sys.argv[0],run_name='__main__')"
    $taskCommand | & $taskCPU -u -X faulthandler - 1>> $taskLog 2>> $taskErr
    if($LASTEXITCODE -ne 0){throw ('Reporting stage failed: '+$taskScript)}
}
Write-Output 'SINGLE_HEAD_RESUMED_AFTER_REPORTING_ONLY_KEY_REPAIR_COMPLETE'
