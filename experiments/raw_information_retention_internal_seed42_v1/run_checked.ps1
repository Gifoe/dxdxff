param([string]$RunTag='01',[ValidateSet('Audit','Admission','Train','All')][string]$StopAfter='All')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\raw_information_retention_internal_seed42_v1'
$runtime='D:\raw_information_retention_internal_seed42_v1_runtime\deterministic_execution_v2'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$File,[string[]]$Statuses) {
  $path=Join-Path "$runtime\public" $File
  return (Test-Path -LiteralPath $path) -and ((Get-Content -LiteralPath $path -Raw|ConvertFrom-Json).status -in $Statuses)
}
function Worker([string]$Script,[string]$Action='') {
  $name="${Script}_${Action}_${RunTag}";$log=Join-Path $runtime "$name.log";$err=Join-Path $runtime "$name.err"
  if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $err)) { throw 'Use a new RunTag; preserve existing logs.' }
  $argv=@('-u','-X','faulthandler',"$code\$Script.py")
  if ($Action) { $argv+=$Action }
  $w=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList $argv -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
  $null=$w.Handle
  while (-not $w.HasExited) { Write-Output "RETENTION_ALIVE stage=$Script action=$Action pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 25;$w.Refresh() }
  $w.WaitForExit()
  if ($null -ne $w.ExitCode -and $w.ExitCode -ne 0) { throw "Worker failed: $Script $Action. Inspect preserved logs/checkpoints; no blind retry." }
}
try {
  $owners=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'raw_information_retention_internal_seed42_v1\\(audit_inputs|train_internal|summarize)\.py' }
  if ($owners) { throw 'An owned worker is already active; no duplicates.' }
  if (-not (Gate 'INPUT_ADMISSION.json' @('PASS','PARTIAL_PRE_FAMILY_NOT_IDENTIFIABLE'))) { Worker 'audit_inputs' }
  if (-not (Gate 'INPUT_ADMISSION.json' @('PASS','PARTIAL_PRE_FAMILY_NOT_IDENTIFIABLE'))) { throw 'Input admission absent' }
  if ($StopAfter -eq 'Audit') { Write-Output 'RETENTION_INPUT_AUDIT_FINISHED';return }
  if (-not (Gate 'GRADIENT_ADMISSION.json' @('PASS'))) { Worker 'train_internal' 'admit' }
  if (-not (Gate 'GRADIENT_ADMISSION.json' @('PASS'))) { throw 'Full-patient gradient admission absent' }
  if ($StopAfter -eq 'Admission') { Write-Output 'RETENTION_ADMISSION_FINISHED';return }
  if (-not (Gate 'RUN_STATUS.json' @('ALL_REGISTERED_INTERNAL_CELLS_COMPLETE'))) { Worker 'train_internal' 'train' }
  if (-not (Gate 'RUN_STATUS.json' @('ALL_REGISTERED_INTERNAL_CELLS_COMPLETE'))) { throw 'Internal training completion absent' }
  if ($StopAfter -eq 'Train') { Write-Output 'RETENTION_TRAIN_FINISHED';return }
  if (-not (Gate 'INTERNAL_TERMINAL.json' @('COMPLETE_VERIFIED_INTERNAL_INFORMATION_RETENTION_STUDY'))) { Worker 'summarize' }
  if (-not (Gate 'INTERNAL_TERMINAL.json' @('COMPLETE_VERIFIED_INTERNAL_INFORMATION_RETENTION_STUDY'))) { throw 'Independent internal verification absent' }
  Write-Output 'RETENTION_INTERNAL_PIPELINE_COMPLETE_NO_OUTER'
} finally { $guard.Dispose() }
