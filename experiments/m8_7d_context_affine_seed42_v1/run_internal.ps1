param([string]$RunTag='01',[ValidateSet('Admission','All')][string]$StopAfter='All')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\m8_7d_context_affine_seed42_v1';$runtime='D:\m8_7d_context_affine_seed42_v1_runtime'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$name,[string]$status) {$p=Join-Path "$runtime\public" $name;return (Test-Path -LiteralPath $p) -and ((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $status)}
function Worker([string]$script) {
 $log=Join-Path $runtime ($script+'_'+$RunTag+'.log');$err=Join-Path $runtime ($script+'_'+$RunTag+'.err')
 if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $err)) {throw 'Fresh RunTag required; preserve logs.'}
 $w=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u','-X','faulthandler',"$code\$script.py") -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while (-not $w.HasExited) {Write-Output "CONTEXT_AFFINE_ALIVE stage=$script pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 20;$w.Refresh()}
 $w.WaitForExit();if ($null -ne $w.ExitCode -and $w.ExitCode -ne 0) {throw 'Worker failed; preserve state and diagnose. No automatic retry.'}
}
try {
 if (Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'm8_7d_context_affine_seed42_v1\\(prepare|admit|train|verify_internal)\.py'}) {throw 'Owned worker active; no duplicate.'}
 if (-not (Gate 'REPRESENTATION_STATUS.json' 'PASS')) {Worker 'prepare'}
 if (-not (Gate 'REPRESENTATION_STATUS.json' 'PASS')) {throw 'Representation seal missing'}
 if (-not (Gate 'ADMISSION.json' 'PASS')) {Worker 'admit'}
 if (-not (Gate 'ADMISSION.json' 'PASS')) {throw 'Admission missing'}
 if ($StopAfter -eq 'Admission') {return}
 if (-not (Gate 'TRAIN_STATUS.json' 'ALL10_NEW_PLUS5_REUSED_MODELS_FROZEN')) {Worker 'train'}
 if (-not (Gate 'TRAIN_STATUS.json' 'ALL10_NEW_PLUS5_REUSED_MODELS_FROZEN')) {throw 'Training terminal missing'}
 if (-not (Gate 'INTERNAL_TERMINAL.json' 'COMPLETE_VERIFIED_CONTEXT_AFFINE_INTERNAL')) {Worker 'verify_internal'}
 if (-not (Gate 'INTERNAL_TERMINAL.json' 'COMPLETE_VERIFIED_CONTEXT_AFFINE_INTERNAL')) {throw 'Independent verification missing'}
 Write-Output 'CONTEXT_AFFINE_INTERNAL_PIPELINE_COMPLETE_NO_OUTER'
} finally {$guard.Dispose()}
