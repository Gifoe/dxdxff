param([string]$RunTag='remaining01')
$ErrorActionPreference='Stop'
$taskCode='D:\m8_physiology_discovery_selection_seed42_v2'
$taskRuntime='D:\m8_physiology_discovery_selection_seed42_v2_runtime'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
$taskGPU='E:\Anaconda\python.exe';$taskCPU='E:\Anaconda\envs\persist_stable_251\python.exe'
$taskLog=Join-Path $taskRuntime ($RunTag+'.log');$taskErr=Join-Path $taskRuntime ($RunTag+'.err')
if((Test-Path -LiteralPath $taskLog) -or (Test-Path -LiteralPath $taskErr)){throw 'Preserve previous logs; use a fresh RunTag'}
$taskLock=[System.IO.File]::Open((Join-Path $taskRuntime 'remaining_coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
$taskDuplicate=Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match '(support_control|evaluate)\.py' -and $_.CommandLine -match 'm8_physiology_discovery_selection_seed42_v2' }
if($taskDuplicate){throw 'A support/evaluation worker already exists'}
# A saved script, not multi-command stdin, makes any throw terminate this stage.
while($true){
    $taskStatus=Get-Content -LiteralPath (Join-Path $taskRuntime 'public\NEURAL_STATUS.json') -Raw | ConvertFrom-Json
    if($taskStatus.status -eq 'ALL80_LOGICAL_MODELS_FROZEN'){break}
    $taskWorker=Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'm8_physiology_discovery_selection_seed42_v2\\train.py' }
    if(-not $taskWorker){throw 'Main trainer absent before completion; diagnose instead of blind retry'}
    Start-Sleep -Seconds 5
}
& $taskGPU -c "import numpy;import pytest;raise SystemExit(pytest.main([r'$taskCode\test_model.py','-q']))" 1>> $taskLog 2>> $taskErr
if($LASTEXITCODE -ne 0){throw 'Unchanged model tests failed'}
if(-not (Test-Path -LiteralPath (Join-Path $taskRuntime 'SUPPORT_MODEL_FREEZE_PRIVATE.pkl'))){
    & $taskGPU -u -X faulthandler (Join-Path $taskCode 'support_control.py') 1>> $taskLog 2>> $taskErr
    if($LASTEXITCODE -ne 0){throw 'Prospective support control failed'}
}
while(-not (Test-Path -LiteralPath (Join-Path $taskRuntime 'public\INDEPENDENT_SELECTION_VERIFICATION.json'))){
    $taskVerifier=Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'm8_physiology_discovery_selection_seed42_v2' -and $_.CommandLine -match 'verify_selection.py' }
    if(-not $taskVerifier){throw 'Selection verification missing; diagnose before Outer'}
    Start-Sleep -Seconds 5
}
& $taskGPU -u -X faulthandler (Join-Path $taskCode 'evaluate.py') 1>> $taskLog 2>> $taskErr
if($LASTEXITCODE -ne 0){throw 'Frozen validation/Outer admission failed'}
$taskCommand="import sys,psutil,runpy;psutil.Process().cpu_affinity([16]);sys.path.insert(0,r'$taskCode');runpy.run_path(r'$taskCode\report.py',run_name='__main__')"
& $taskCPU -u -X faulthandler -c $taskCommand 1>> $taskLog 2>> $taskErr
if($LASTEXITCODE -ne 0){throw 'Score-only statistics failed'}
Write-Output 'REMAINING_SUPPORT_AND_FROZEN_OUTER_REPORT_COMPLETE'
