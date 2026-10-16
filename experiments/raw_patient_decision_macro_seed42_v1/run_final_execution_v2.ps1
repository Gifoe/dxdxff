param([string]$RunTag='03')
# Execution/reporting extension only. All bound scientific sources stay unchanged.
$ErrorActionPreference='Stop'
$code='D:\raw_patient_decision_macro_seed42_v1';$runtime='D:\raw_patient_decision_macro_seed42_v1_runtime\final_stage_v1'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$name,[string]$status) { $p=Join-Path "$runtime\public" $name;return (Test-Path -LiteralPath $p) -and ((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $status) }
function Worker([string]$action) {
 $log=Join-Path $runtime "${action}_${RunTag}.log";$err=Join-Path $runtime "${action}_${RunTag}.err"
 if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $err)) {throw 'Fresh RunTag required.'}
 $argv=@('-u','-X','faulthandler',"$code\final_stage.py",$action)
 if ($action -eq 'verify_reporting') {$argv=@('-u','-X','faulthandler',"$code\verify_final_reporting_v2.py")}
 $w=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList $argv -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while (-not $w.HasExited) {Write-Output "RAW_DECISION_V2_ALIVE action=$action pid=$($w.Id)";Start-Sleep -Seconds 20;$w.Refresh()}
 $w.WaitForExit();if ($null -ne $w.ExitCode -and $w.ExitCode -ne 0) {throw 'Worker failed; inspect preserved logs.'}
}
try {
 if (Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'raw_patient_decision_macro_seed42_v1\\(internal|summarize|final_stage|verify_final|verify_final_reporting_v2)\.py'}) {throw 'Owned worker active; no duplicate.'}
 if (-not (Gate 'CALIBRATION_STATUS.json' 'ALL45_DECISIONS_FROZEN')) {Worker 'calibrate'}
 if (-not (Gate 'CALIBRATION_STATUS.json' 'ALL45_DECISIONS_FROZEN')) {throw 'Decision freeze missing.'}
 if (-not (Gate 'OUTER_STATUS.json' 'ALL45_FROZEN_MARGIN_DECISION_CELLS_SEALED')) {Worker 'test'}
 if (-not (Gate 'OUTER_STATUS.json' 'ALL45_FROZEN_MARGIN_DECISION_CELLS_SEALED')) {throw 'Score seal missing.'}
 if (-not (Gate 'FINAL_TERMINAL.json' 'COMPLETE_NINE_RAW_DECISION_PROCEDURES')) {Worker 'report'}
 if (-not (Gate 'FINAL_TERMINAL.json' 'COMPLETE_NINE_RAW_DECISION_PROCEDURES')) {throw 'Report missing.'}
 if (-not (Gate 'INDEPENDENT_FINAL_VERIFICATION.json' 'PASS')) {Worker 'verify_reporting'}
 if (-not (Gate 'INDEPENDENT_FINAL_VERIFICATION.json' 'PASS')) {throw 'Independent verification missing.'}
 if (-not (Gate 'REPORTING_VERIFIER_REPAIR.json' 'PASS')) {throw 'Reporting repair audit missing.'}
 Write-Output 'RAW_DECISION_ALL_NINE_VERIFIED_COMPLETE'
} finally {$guard.Dispose()}
