$ErrorActionPreference='Stop'
$taskCode='D:\m8_single_head_concat_seed42_v1'
$taskRuntime='D:\m8_single_head_concat_seed42_v1_runtime'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONUNBUFFERED='1'
$taskPython='E:\Anaconda\python.exe'
$taskCPU='E:\Anaconda\envs\persist_stable_251\python.exe'
New-Item -ItemType Directory -Path $taskRuntime -Force | Out-Null
$taskLog=Join-Path $taskRuntime 'production01.log'
$taskErr=Join-Path $taskRuntime 'production01.err'
if((Test-Path -LiteralPath $taskLog) -or (Test-Path -LiteralPath $taskErr)){throw 'Use new logs; preserve previous execution.'}
$taskDup=Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'm8_single_head_concat_seed42_v1.+(train|evaluate).py' }
if($taskDup){throw 'Active worker; refuse duplicate.'}
& $taskPython -m pytest (Join-Path $taskCode 'test_protocol.py') (Join-Path $taskCode 'test_model.py') -q 1>> $taskLog 2>> $taskErr
if($LASTEXITCODE -ne 0){throw 'Synthetic admission failed.'}
foreach($taskScript in @('train.py','verify_training.py')){
    & $taskPython -u -X faulthandler (Join-Path $taskCode $taskScript) 1>> $taskLog 2>> $taskErr
    if($LASTEXITCODE -ne 0){throw ('Failed original-runtime stage: '+$taskScript)}
}
$taskCPUCode="import sys,runpy; import psutil; psutil.Process().cpu_affinity([16]); sys.argv=[r'$taskCode\report.py','development']; sys.path.insert(0,r'$taskCode'); runpy.run_path(sys.argv[0],run_name='__main__')"
$taskCPUCode | & $taskCPU -u -X faulthandler - 1>> $taskLog 2>> $taskErr
if($LASTEXITCODE -ne 0){throw 'Internal selection failed; forbid Outer.'}
foreach($taskScript in @('complexity.py','evaluate.py')){
    & $taskPython -u -X faulthandler (Join-Path $taskCode $taskScript) 1>> $taskLog 2>> $taskErr
    if($LASTEXITCODE -ne 0){throw ('Failed frozen stage: '+$taskScript)}
}
foreach($taskScript in @('report.py','verify.py')){
    $taskArg=if($taskScript -eq 'report.py'){",'outer'"}else{''}
    $taskCPUCode="import sys,runpy; import psutil; psutil.Process().cpu_affinity([16]); sys.argv=[r'$taskCode\$taskScript'$taskArg]; sys.path.insert(0,r'$taskCode'); runpy.run_path(sys.argv[0],run_name='__main__')"
    $taskCPUCode | & $taskCPU -u -X faulthandler - 1>> $taskLog 2>> $taskErr
    if($LASTEXITCODE -ne 0){throw ('Failed reporting stage: '+$taskScript)}
}
Write-Output 'SINGLE_HEAD_CONCAT_EXECUTION_AND_INDEPENDENT_VERIFICATION_COMPLETE'
