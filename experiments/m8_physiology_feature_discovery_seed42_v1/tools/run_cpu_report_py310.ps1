$ErrorActionPreference='Stop'
$taskRuntime='D:\m8_physiology_feature_discovery_seed42_v1_runtime'
$parity=Get-Content -Raw -LiteralPath ($taskRuntime+'\public\CPU_STATISTICS_RUNTIME_PARITY.json') | ConvertFrom-Json
if ($parity.status -ne 'PASS' -or $parity.maximum_metric_difference -ne 0) { throw 'Runtime parity not proven' }
$sourceHash=(Get-FileHash -Algorithm SHA256 -LiteralPath 'D:\m8_physiology_feature_discovery_seed42_v1\test_cpu_statistics_runtime_v2.py').Hash.ToLowerInvariant()
if ($sourceHash -ne $parity.source_sha256) { throw 'Parity source changed' }
$duplicates=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'evaluate_final|report_final_' }
if ($duplicates) { throw 'Existing evaluator/reporter; no duplicate' }
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONFAULTHANDLER='1'
$taskOut=$taskRuntime+'\final_report_py310_06.log';$taskErr=$taskRuntime+'\final_report_py310_06.err'
if ((Test-Path -LiteralPath $taskOut) -or (Test-Path -LiteralPath $taskErr)) { throw 'New logs required' }
$worker=Start-Process -FilePath 'E:\Anaconda\envs\persist_stable_251\python.exe' -ArgumentList @('-X','faulthandler','-u','D:\m8_physiology_feature_discovery_seed42_v1\report_final_cpu_v2.py') -WindowStyle Hidden -PassThru -RedirectStandardOutput $taskOut -RedirectStandardError $taskErr
$taskHandle=$worker.Handle
Write-Output ('PY310_REPORT_PID='+$worker.Id)
while (-not $worker.HasExited) { Write-Output 'PY310_REPORT_VERIFIED_LIVE';Start-Sleep -Seconds 10;$worker.Refresh() }
$worker.WaitForExit()
Write-Output ('PY310_REPORT_EXIT='+$worker.ExitCode)
Get-Content -LiteralPath $taskOut -Tail 3
Get-Content -LiteralPath $taskErr -Tail 18
if ($worker.ExitCode -ne 0) { throw 'CPU report failed; inspect diagnostic' }
$final=Get-Content -Raw -LiteralPath ($taskRuntime+'\final_Outer\public\REPORTING_RECOVERY_AUDIT.json') | ConvertFrom-Json
if ($final.status -ne 'PASS' -or -not $final.torch_not_imported) { throw 'Reporting audit failed' }
