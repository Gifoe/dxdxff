param([int]$WorkerId=24904,[string]$RunTag='01')
$ErrorActionPreference='Stop'
$taskCode='D:\m8_physiology_feature_discovery_seed42_v1'
$taskRuntime='D:\m8_physiology_feature_discovery_seed42_v1_runtime'
while ($true) {
    $p=Get-CimInstance Win32_Process -Filter "ProcessId=$WorkerId"
    if ($p -and $p.CommandLine -notmatch [regex]::Escape($taskCode+'\train_integrated_v2.py')) { throw 'PID identity changed' }
    if (-not $p) { break }
    Write-Output ('WAIT_VERIFIED_INTEGRATED_TRAINER PID='+$WorkerId+' '+(Get-Date -Format o))
    Start-Sleep -Seconds 20
}
$freeze=Get-Content -Raw -LiteralPath ($taskRuntime+'\integrated_M8_v2\public\MODEL_FREEZE_AUDIT.json') | ConvertFrom-Json
if ($freeze.status -ne 'ALL45_LOGICAL_INTEGRATED_MODELS_FROZEN_BEFORE_OUTER') { throw 'Model freeze incomplete; diagnose, do not retry' }
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1'
foreach ($item in @(@{Name='old7_independent';Script='tools\verify_old7_training.py'},@{Name='integrated_independent';Script='verify_integrated_training.py'})) {
    $stdout=Join-Path $taskRuntime ($item.Name+'_'+$RunTag+'.log')
    $stderr=Join-Path $taskRuntime ($item.Name+'_'+$RunTag+'.err')
    if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) { throw 'Preserve existing verification logs; new tag required' }
    $worker=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u',($taskCode+'\'+$item.Script)) -WorkingDirectory $taskCode -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Write-Output ('VERIFIER_STARTED '+$item.Name+' PID='+$worker.Id)
    while (-not $worker.HasExited) {
        Write-Output ('VERIFIER_LIVE '+$item.Name+' PID='+$worker.Id+' '+(Get-Date -Format o))
        Start-Sleep -Seconds 20
        $worker.Refresh()
    }
    $worker.WaitForExit()
    if ($worker.ExitCode -ne 0) { throw ('Verifier failed: '+$item.Name+' preserve logs and inspect') }
}
Write-Output 'BOTH_NEURAL_VALIDATION_VERIFIERS_COMPLETE_NO_OUTER'
