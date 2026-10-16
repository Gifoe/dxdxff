param([string]$RunTag='01',[ValidateSet('Preparation','Training','All')][string]$StopAfter='All')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\raw_macro_f1_head_screen_seed42_v1';$runtime='D:\raw_macro_f1_head_screen_seed42_v1_runtime\cpu_bridge_execution_v2\final_stage_v1'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$name,[string]$status) { $p=Join-Path "$runtime\public" $name;return (Test-Path -LiteralPath $p) -and ((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $status) }
function Worker([string]$action) {
  $log=Join-Path $runtime "${action}_${RunTag}.log";$err=Join-Path $runtime "${action}_${RunTag}.err"
  if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $err)) { throw 'Use a fresh RunTag; preserve logs.' }
  $w=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u','-X','faulthandler',"$code\final_stage.py",$action) -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
  $null=$w.Handle
  while (-not $w.HasExited) { Write-Output "MACRO_FINAL_ALIVE action=$action pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 25;$w.Refresh() }
  $w.WaitForExit();if ($null -ne $w.ExitCode -and $w.ExitCode -ne 0) { throw 'Worker failed; inspect preserved logs.' }
}
try {
  $active=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'raw_macro_f1_head_screen_seed42_v1\\(final_stage|execute_cpu_bridge_v2)\.py' }
  if ($active) { throw 'Owned worker active; no duplicates.' }
  if (-not (Gate 'FINAL_PREPARATION.json' 'PASS')) { Worker 'prepare' }
  if (-not (Gate 'FINAL_PREPARATION.json' 'PASS')) { throw 'Preparation missing.' }
  if ($StopAfter -eq 'Preparation') { return }
  if (-not (Gate 'FINAL_TRAINING_STATUS.json' 'ALL15_FINAL_LOGICAL_MODELS_FROZEN')) { Worker 'train' }
  if (-not (Gate 'FINAL_TRAINING_STATUS.json' 'ALL15_FINAL_LOGICAL_MODELS_FROZEN')) { throw 'Training freeze missing.' }
  if ($StopAfter -eq 'Training') { return }
  if (-not (Gate 'OUTER_STATUS.json' 'ALL15_FROZEN_SCORE_CELLS_SEALED')) { Worker 'test' }
  if (-not (Gate 'OUTER_STATUS.json' 'ALL15_FROZEN_SCORE_CELLS_SEALED')) { throw 'Score seal missing.' }
  if (-not (Gate 'FINAL_TERMINAL.json' 'COMPLETE_VERIFIED_THREE_RAW_MACRO_PROCEDURES')) { Worker 'summarize' }
  if (-not (Gate 'FINAL_TERMINAL.json' 'COMPLETE_VERIFIED_THREE_RAW_MACRO_PROCEDURES')) { throw 'Verified terminal missing.' }
  Write-Output 'MACRO_FINAL_PIPELINE_COMPLETE'
} finally { $guard.Dispose() }
