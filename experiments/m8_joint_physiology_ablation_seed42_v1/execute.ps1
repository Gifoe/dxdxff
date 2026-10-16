$ErrorActionPreference = 'Stop'
$taskCode = 'D:\m8_joint_physiology_ablation_seed42_v1'
$taskRuntime = 'D:\m8_joint_physiology_ablation_seed42_v1_runtime'
$taskTag = 'production01'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONUNBUFFERED = '1'
$taskOriginalPython = 'E:\Anaconda\python.exe'
$taskReportPython = 'E:\Anaconda\envs\persist_stable_251\python.exe'
New-Item -ItemType Directory -Path $taskRuntime -Force | Out-Null
$taskLog = Join-Path $taskRuntime ($taskTag + '.log')
$taskErr = Join-Path $taskRuntime ($taskTag + '.err')
if ((Test-Path -LiteralPath $taskLog) -or (Test-Path -LiteralPath $taskErr)) { throw 'Use a NEW execution tag; preserve previous logs.' }
$taskDup = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -like '*m8_joint_physiology_ablation_seed42_v1*run.py*' }
if ($taskDup) { throw 'Active experiment worker exists; no duplicate launch.' }
& $taskOriginalPython -m pytest (Join-Path $taskCode 'test_analysis.py') -q 1>> $taskLog 2>> $taskErr
if ($LASTEXITCODE -ne 0) { throw 'Synthetic tests failed.' }
foreach ($taskStage in @('train','verify')) {
    & $taskOriginalPython -u -X faulthandler (Join-Path $taskCode 'run.py') $taskStage 1>> $taskLog 2>> $taskErr
    if ($LASTEXITCODE -ne 0) { throw ('Original-runtime stage failed: ' + $taskStage) }
}
# Existing audited reporting environment; process-only CPU affinity, no package/driver changes.
$taskCPUCode = "import os,sys,runpy; import psutil; psutil.Process().cpu_affinity([16]); sys.argv=[r'$taskCode\report.py','development']; sys.path.insert(0,r'$taskCode'); runpy.run_path(sys.argv[0],run_name='__main__')"
$taskCPUCode | & $taskReportPython -u -X faulthandler - 1>> $taskLog 2>> $taskErr
if ($LASTEXITCODE -ne 0) { throw 'Validation report/selection failed; forbid Outer.' }
& $taskOriginalPython -u -X faulthandler (Join-Path $taskCode 'run.py') outer 1>> $taskLog 2>> $taskErr
if ($LASTEXITCODE -ne 0) { throw 'Frozen Outer inference failed.' }
foreach ($taskScript in @('report.py','verify_results.py')) {
    $taskStageArg = if ($taskScript -eq 'report.py') { ",'outer'" } else { '' }
    $taskCPUCode = "import sys,runpy; import psutil; psutil.Process().cpu_affinity([16]); sys.argv=[r'$taskCode\$taskScript'$taskStageArg]; sys.path.insert(0,r'$taskCode'); runpy.run_path(sys.argv[0],run_name='__main__')"
    $taskCPUCode | & $taskReportPython -u -X faulthandler - 1>> $taskLog 2>> $taskErr
    if ($LASTEXITCODE -ne 0) { throw ('CPU final stage failed: ' + $taskScript) }
}
Write-Output 'JOINT_2CUBED_EXECUTION_COMPLETE_AND_INDEPENDENTLY_VERIFIED'
