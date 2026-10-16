param([string]$RunTag='01')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\raw_macro_f1_optimization_seed42_v1'
$runtime='D:\raw_macro_f1_optimization_seed42_v1_runtime'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$guard=[System.IO.File]::Open((Join-Path $runtime 'admission.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
try {
  $active=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'raw_macro_f1_optimization_seed42_v1\\admit_baseline\.py' }
  if ($active) { throw 'Baseline replay already active; do not duplicate.' }
  $gate=Join-Path $runtime 'public\BASELINE_REFERENCE_PARITY.json'
  if (Test-Path -LiteralPath $gate) { throw 'Completed replay exists; verify its binding rather than repeating.' }
  $log=Join-Path $runtime "baseline_replay_$RunTag.log";$err=Join-Path $runtime "baseline_replay_$RunTag.err"
  if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $err)) { throw 'Use an unused RunTag; retain all logs.' }
  $w=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u','-X','faulthandler',"$code\admit_baseline.py") -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
  $null=$w.Handle
  while (-not $w.HasExited) { Write-Output "MACRO_BASELINE_REPLAY_ALIVE pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 20;$w.Refresh() }
  $w.WaitForExit()
  if ($null -ne $w.ExitCode -and $w.ExitCode -ne 0) { throw 'Replay failed; inspect logs. No scientific changes or blind retry.' }
  if (-not (Test-Path -LiteralPath $gate)) { throw 'No independent replay completion gate.' }
  $q=Get-Content -LiteralPath $gate -Raw | ConvertFrom-Json
  if ($q.status -ne 'PASS' -or $q.folds -ne 5 -or $q.new_Outer_passes -ne 0) { throw 'Unexpected replay completion gate.' }
  Write-Output 'MACRO_BASELINE_REPLAY_COMPLETE_NO_TRAINING_NO_OUTER'
} finally { $guard.Dispose() }
