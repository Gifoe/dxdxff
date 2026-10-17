param([string]$RunTag='02',[ValidateSet('Training','All')][string]$StopAfter='All')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\raw_joint_moment_macro_seed42_v1';$runtime='D:\raw_joint_moment_macro_seed42_v1_runtime\final_stage_v1'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$name,[string]$status) {$p=Join-Path "$runtime\public" $name;return (Test-Path -LiteralPath $p) -and ((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $status)}
function Worker([string]$action) {
 $log=Join-Path $runtime "${action}_${RunTag}.log";$err=Join-Path $runtime "${action}_${RunTag}.err"
 if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $err)) {throw 'Fresh RunTag required; preserve logs.'}
 $argv=@('-u','-X','faulthandler',"$code\final_stage.py",$action)
 if ($action -eq 'admit') {$argv=@('-u','-X','faulthandler',"$code\admit_exact_roles_v2.py")}
 if ($action -eq 'verify') {$argv=@('-u','-X','faulthandler',"$code\verify_final.py")}
 if ($action -eq 'raw_audit') {$argv=@('-u','-X','faulthandler',"$code\audit_raw_support.py")}
 $w=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList $argv -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while (-not $w.HasExited) {Write-Output "JOINT_RAW_EXACT_ROLE_FINAL_ALIVE action=$action pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 20;$w.Refresh()}
 $w.WaitForExit();if ($null -ne $w.ExitCode -and $w.ExitCode -ne 0) {throw 'Worker failed; inspect preserved logs.'}
}
try {
 if (Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'raw_joint_moment_macro_seed42_v1\\(admit|admit_exact_roles_v2|train_internal|summarize|final_stage|verify_final|audit_raw_support)\.py'}) {throw 'Owned worker active; no duplicate or overlapping stage.'}
 if ((Get-FileHash -LiteralPath "$runtime\INPUT_FREEZE_PRIVATE.json" -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'cd08eeec8272affa4ac7dac448b47b206a725f4ec0c74abe00faf27c25b0d6f2') {throw 'Original final binding changed.'}
 if (-not (Gate 'FINAL_ADMISSION.json' 'PASS')) {Worker 'admit'}
 if (-not (Gate 'FINAL_ADMISSION.json' 'PASS')) {throw 'Actual exact-role FIT admission missing.'}
 if (-not (Gate 'TRAINING_STATUS.json' 'ALL30_JOINT_RAW_FINAL_MODELS_FROZEN')) {Worker 'fit'}
 if (-not (Gate 'TRAINING_STATUS.json' 'ALL30_JOINT_RAW_FINAL_MODELS_FROZEN')) {throw 'Model/threshold freeze missing.'}
 if ($StopAfter -eq 'Training') {return}
 if (-not (Gate 'OUTER_STATUS.json' 'ALL30_FROZEN_JOINT_RAW_SCORE_CELLS_SEALED')) {Worker 'test'}
 if (-not (Gate 'OUTER_STATUS.json' 'ALL30_FROZEN_JOINT_RAW_SCORE_CELLS_SEALED')) {throw 'Score seal missing.'}
 if (-not (Gate 'FINAL_TERMINAL.json' 'COMPLETE_SIX_JOINT_RAW_PROCEDURES')) {Worker 'report'}
 if (-not (Gate 'FINAL_TERMINAL.json' 'COMPLETE_SIX_JOINT_RAW_PROCEDURES')) {throw 'Report missing.'}
 if (-not (Gate 'INDEPENDENT_FINAL_VERIFICATION.json' 'PASS')) {Worker 'verify'}
 if (-not (Gate 'INDEPENDENT_FINAL_VERIFICATION.json' 'PASS')) {throw 'Independent verification missing.'}
 if (-not (Gate 'RAW_SUPPORT_AUDIT.json' 'PASS')) {Worker 'raw_audit'}
 if (-not (Gate 'RAW_SUPPORT_AUDIT.json' 'PASS')) {throw 'Raw support census missing.'}
 Write-Output 'JOINT_RAW_ALL_SIX_FINAL_PROCEDURES_VERIFIED_EXACT_ORIGINAL_ROLES'
} finally {$guard.Dispose()}
