param([string]$RunTag='02')
$ErrorActionPreference='Stop'
$code='D:\raw_patient_decision_macro_seed42_v1';$runtime='D:\raw_patient_decision_macro_seed42_v1_runtime\final_stage_v1'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
try {
 if (Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'raw_patient_decision_macro_seed42_v1\\(final_stage|verify_final|verify_final_reporting_v2)\.py'}) {throw 'Owned worker active.'}
 $log=Join-Path $runtime "verify_reporting_${RunTag}.log";$err=Join-Path $runtime "verify_reporting_${RunTag}.err"
 if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $err)) {throw 'Fresh RunTag required.'}
 $w=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u','-X','faulthandler',"$code\verify_final_reporting_v2.py") -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while (-not $w.HasExited) {Write-Output "RAW_DECISION_VERIFY_ALIVE pid=$($w.Id)";Start-Sleep -Seconds 20;$w.Refresh()}
 $w.WaitForExit();if ($null -ne $w.ExitCode -and $w.ExitCode -ne 0) {throw 'Reporting verification failed; preserve logs.'}
 if ((Get-Content -LiteralPath "$runtime\public\INDEPENDENT_FINAL_VERIFICATION.json" -Raw|ConvertFrom-Json).status -ne 'PASS') {throw 'Independent gate missing.'}
 if ((Get-Content -LiteralPath "$runtime\public\REPORTING_VERIFIER_REPAIR.json" -Raw|ConvertFrom-Json).status -ne 'PASS') {throw 'Repair audit missing.'}
 Write-Output 'RAW_DECISION_ALL_NINE_VERIFIED_COMPLETE'
} finally {$guard.Dispose()}
