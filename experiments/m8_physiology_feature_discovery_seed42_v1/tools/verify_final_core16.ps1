param([int]$ReportWorkerId=12400)
$ErrorActionPreference='Stop'
$taskRuntime='D:\m8_physiology_feature_discovery_seed42_v1_runtime'
$taskCode='D:\m8_physiology_feature_discovery_seed42_v1'
while ($true) {
    $p=Get-CimInstance Win32_Process -Filter "ProcessId=$ReportWorkerId"
    if (-not $p) { break }
    if ($p.CommandLine -notmatch [regex]::Escape($taskCode+'\report_final_cpu_v2.py')) { throw 'Report PID identity changed' }
    Write-Output ('WAIT_VERIFIED_EXISTING_REPORT '+$ReportWorkerId)
    Start-Sleep -Seconds 10
}
$report=Get-Content -Raw -LiteralPath ($taskRuntime+'\final_Outer\public\REPORTING_RECOVERY_AUDIT.json') | ConvertFrom-Json
if ($report.status -ne 'PASS' -or -not $report.exact_frozen_report_and_classification_AST) { throw 'Report completion not verified' }
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONFAULTHANDLER='1'
foreach ($item in @(@{Name='final_cache_stage_01';Script='verify_final_cache_stage.py';Python='E:\Anaconda\python.exe'},@{Name='final_cpu_metrics_01';Script='verify_final_cpu_metrics.py';Python='E:\Anaconda\envs\persist_stable_251\python.exe'})) {
    $stdout=Join-Path $taskRuntime ($item.Name+'.log');$stderr=Join-Path $taskRuntime ($item.Name+'.err')
    if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) { throw 'New logs required' }
    $worker=Start-Process -FilePath $item.Python -ArgumentList @('-X','faulthandler','-u',($taskCode+'\'+$item.Script)) -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $taskHandle=$worker.Handle
    $worker.ProcessorAffinity=[IntPtr]65536
    if ($worker.ProcessorAffinity.ToInt64() -ne 65536) { throw 'Verifier affinity not applied' }
    Write-Output ('FINAL_VERIFY_STARTED '+$item.Name+' PID='+$worker.Id)
    while (-not $worker.HasExited) { Write-Output ('FINAL_VERIFY_LIVE '+$item.Name);Start-Sleep -Seconds 10;$worker.Refresh() }
    $worker.WaitForExit()
    Write-Output ('FINAL_VERIFY_EXIT '+$item.Name+' '+$worker.ExitCode)
    Get-Content -LiteralPath $stdout -Tail 3
    Get-Content -LiteralPath $stderr -Tail 20
    if ($worker.ExitCode -ne 0) { throw ('Final verification failed '+$item.Name+' inspect, no automatic retry') }
}
$done=Get-Content -Raw -LiteralPath ($taskRuntime+'\final_Outer\public\INDEPENDENT_FINAL_VERIFICATION.json') | ConvertFrom-Json
if ($done.status -ne 'PASS' -or $done.patient_rows -ne 2080) { throw 'Independent final audit not proven' }
Write-Output 'ALL26_FINAL_INDEPENDENT_AUDITS_PASS_REPORT_AND_COMPLETION_REVIEW_PENDING'
