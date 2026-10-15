$ErrorActionPreference='Stop'
$taskRuntime='D:\m8_physiology_feature_discovery_seed42_v1_runtime'
$taskCode='D:\m8_physiology_feature_discovery_seed42_v1'
$duplicates=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'report_final_|test_cpu_statistics_runtime' }
if ($duplicates) { throw 'Existing reporting process; no duplicate' }
if ([Environment]::ProcessorCount -ne 32) { throw 'Processor inventory changed; inspect first' }
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONFAULTHANDLER='1'
foreach ($item in @(@{Name='cpu_parity_core16';Script='test_cpu_statistics_runtime_v2.py';Extra=@('--tag','py310_core16')},@{Name='final_report_core16_07';Script='report_final_cpu_v2.py';Extra=@()})) {
    $stdout=Join-Path $taskRuntime ($item.Name+'.log');$stderr=Join-Path $taskRuntime ($item.Name+'.err')
    if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) { throw 'New logs required' }
    $arguments=@('-X','faulthandler','-u',($taskCode+'\'+$item.Script))+$item.Extra
    $worker=Start-Process -FilePath 'E:\Anaconda\envs\persist_stable_251\python.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $taskHandle=$worker.Handle
    $worker.ProcessorAffinity=[IntPtr]65536
    if ($worker.ProcessorAffinity.ToInt64() -ne 65536) { throw 'Requested process-only affinity not applied' }
    Write-Output ('CORE16_REPORT_STAGE '+$item.Name+' PID='+$worker.Id)
    while (-not $worker.HasExited) { Write-Output ('CORE16_STAGE_LIVE '+$item.Name);Start-Sleep -Seconds 10;$worker.Refresh() }
    $worker.WaitForExit()
    Write-Output ('CORE16_STAGE_EXIT '+$item.Name+' '+$worker.ExitCode)
    Get-Content -LiteralPath $stdout -Tail 3
    Get-Content -LiteralPath $stderr -Tail 12
    if ($worker.ExitCode -ne 0) { throw ('Reporting stage failed '+$item.Name+' preserve files and inspect') }
}
