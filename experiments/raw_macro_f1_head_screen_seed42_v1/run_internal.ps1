param([string]$RunTag='01',[ValidateSet('Admission','All')][string]$StopAfter='All')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\raw_macro_f1_head_screen_seed42_v1';$runtime='D:\raw_macro_f1_head_screen_seed42_v1_runtime'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$name,[string]$status) { $p=Join-Path "$runtime\public" $name;return (Test-Path -LiteralPath $p) -and ((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $status) }
function Worker([string]$script,[string]$action='') {
  $log=Join-Path $runtime "${script}_${action}_${RunTag}.log";$err=Join-Path $runtime "${script}_${action}_${RunTag}.err"
  if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $err)) { throw 'Use a fresh RunTag; preserve logs.' }
  $argv=@('-u','-X','faulthandler',"$code\$script.py");if ($action) { $argv+=$action }
  $w=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList $argv -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
  $null=$w.Handle
  while (-not $w.HasExited) { Write-Output "MACRO_HEAD_ALIVE stage=$script action=$action pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 25;$w.Refresh() }
  $w.WaitForExit();if ($null -ne $w.ExitCode -and $w.ExitCode -ne 0) { throw 'Worker failure; inspect logs, no blind restart.' }
}
try {
  $active=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'raw_macro_f1_head_screen_seed42_v1\\(train_internal|summarize_internal)\.py' }
  if ($active) { throw 'Owned worker active; no duplicates.' }
  if (-not (Gate 'ADMISSION.json' 'PASS')) { Worker 'train_internal' 'admit' }
  if (-not (Gate 'ADMISSION.json' 'PASS')) { throw 'Admission missing.' }
  if ($StopAfter -eq 'Admission') { Write-Output 'MACRO_HEAD_ADMISSION_FINISHED';return }
  if (-not (Gate 'INTERNAL_STATUS.json' 'ALL120_INTERNAL_HEAD_CELLS_COMPLETE')) { Worker 'train_internal' 'train' }
  if (-not (Gate 'INTERNAL_STATUS.json' 'ALL120_INTERNAL_HEAD_CELLS_COMPLETE')) { throw 'Training terminal missing.' }
  if (-not (Gate 'INTERNAL_TERMINAL.json' 'COMPLETE_VERIFIED_MACRO_HEAD_INTERNAL')) { Worker 'summarize_internal' }
  if (-not (Gate 'INTERNAL_TERMINAL.json' 'COMPLETE_VERIFIED_MACRO_HEAD_INTERNAL')) { throw 'Independent terminal missing.' }
  Write-Output 'MACRO_HEAD_INTERNAL_PIPELINE_COMPLETE_NO_OUTER'
} finally { $guard.Dispose() }
