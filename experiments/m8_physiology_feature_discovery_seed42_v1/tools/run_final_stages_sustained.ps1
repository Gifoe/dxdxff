param([string]$RunTag='01')
$ErrorActionPreference='Stop'
$taskCode='D:\m8_physiology_feature_discovery_seed42_v1'
$taskRuntime='D:\m8_physiology_feature_discovery_seed42_v1_runtime'
$oldAudit=$taskRuntime+'\old7_matched_M8\public\INDEPENDENT_RETRAIN_VERIFICATION.json'
$newAudit=$taskRuntime+'\integrated_M8_v2\public\INDEPENDENT_TRAINING_VERIFICATION.json'
$quietSince=Get-Date
while ($true) {
    $oldPass=$false;$newPass=$false
    if (Test-Path -LiteralPath $oldAudit) { $oldPass=((Get-Content -Raw -LiteralPath $oldAudit | ConvertFrom-Json).status -eq 'PASS') }
    if (Test-Path -LiteralPath $newAudit) { $newPass=((Get-Content -Raw -LiteralPath $newAudit | ConvertFrom-Json).status -eq 'PASS') }
    $workers=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'train_integrated_v2.py|verify_old7_training.py|verify_integrated_training.py' }
    if ($oldPass -and $newPass -and -not $workers) { break }
    if ($workers) {
        $quietSince=Get-Date
        Write-Output ('FINAL_WAIT_VERIFIED_TRAIN_OR_VALIDATION_WORKER '+($workers.ProcessId -join ',')+' '+(Get-Date -Format o))
    } else {
        Write-Output ('FINAL_WAIT_ASYNC_HANDOFF_GRACE '+(Get-Date -Format o))
        if (((Get-Date)-$quietSince).TotalSeconds -gt 90) { throw 'No live prerequisite worker and audits incomplete: inspect; no automatic retry' }
    }
    Start-Sleep -Seconds 20
}
$duplicates=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'audit_compute.py|evaluate_final' }
if ($duplicates) { throw 'Existing final-stage process: do not duplicate' }
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1'
foreach ($item in @(@{Name='compute';Script='audit_compute.py'},@{Name='final_Outer';Script='evaluate_final_v2.py'})) {
    $stdout=Join-Path $taskRuntime ($item.Name+'_'+$RunTag+'.log')
    $stderr=Join-Path $taskRuntime ($item.Name+'_'+$RunTag+'.err')
    if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) { throw 'Existing logs preserved: new RunTag required' }
    $worker=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u',($taskCode+'\'+$item.Script)) -WorkingDirectory $taskCode -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Write-Output ('FINAL_STAGE_STARTED '+$item.Name+' PID='+$worker.Id)
    while (-not $worker.HasExited) {
        Write-Output ('FINAL_STAGE_VERIFIED_LIVE '+$item.Name+' PID='+$worker.Id+' '+(Get-Date -Format o))
        Start-Sleep -Seconds 20
        $worker.Refresh()
    }
    $worker.WaitForExit()
    if ($worker.ExitCode -ne 0) { throw ('Stage failed '+$item.Name+': preserve all files, inspect, no automatic retry') }
}
$status=Get-Content -Raw -LiteralPath ($taskRuntime+'\final_Outer\public\STATUS.json') | ConvertFrom-Json
if ($status.status -ne 'ALL26_FEATURE_PROCEDURES_OUTER_EVALUATED_AND_METRICS_VERIFIED') { throw 'Final metrics completion not proven' }
Write-Output 'FINAL26_EVALUATION_COMPLETE_REPORT_AND_COMPLETION_AUDIT_STILL_REQUIRED'
