param([string]$RunTag='01')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\raw_patient_decision_macro_seed42_v1';$runtime='D:\raw_patient_decision_macro_seed42_v1_runtime'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$name,[string]$status) { $p=Join-Path "$runtime\public" $name;return (Test-Path -LiteralPath $p) -and ((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $status) }
function Worker([string]$script) {
 $log=Join-Path $runtime "${script}_${RunTag}.log";$err=Join-Path $runtime "${script}_${RunTag}.err"
 if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $err)) {throw 'Use fresh RunTag; preserve logs.'}
 $w=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u','-X','faulthandler',"$code\$script.py") -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while (-not $w.HasExited) {Write-Output "RAW_DECISION_ALIVE stage=$script pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 20;$w.Refresh()}
 $w.WaitForExit();if ($null -ne $w.ExitCode -and $w.ExitCode -ne 0) {throw 'Worker failed; inspect preserved logs.'}
}
try {
 if (Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'raw_patient_decision_macro_seed42_v1\\(internal|summarize)\.py'}) {throw 'Owned worker active; no duplicate.'}
 if (-not (Gate 'RUN_STATUS.json' 'ALL120_DECODER_CELLS_SEALED')) {Worker 'internal'}
 if (-not (Gate 'RUN_STATUS.json' 'ALL120_DECODER_CELLS_SEALED')) {throw 'Score gate missing.'}
 if (-not (Gate 'INTERNAL_TERMINAL.json' 'COMPLETE_VERIFIED_PATIENT_DECODERS_INTERNAL')) {Worker 'summarize'}
 if (-not (Gate 'INTERNAL_TERMINAL.json' 'COMPLETE_VERIFIED_PATIENT_DECODERS_INTERNAL')) {throw 'Independent verification gate missing.'}
 Write-Output 'RAW_DECISION_INTERNAL_PIPELINE_COMPLETE_NO_OUTER'
} finally {$guard.Dispose()}
