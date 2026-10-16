param([string]$RunTag='02',[ValidateSet('Stage1','Train','Outer')][string]$StopAfter='Stage1')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\raw_latent_contrast_optimization_seed42_v1'
$runtime='D:\raw_latent_contrast_optimization_seed42_v1_runtime'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$Name,[string]$Status){
  $p=Join-Path "$runtime\public" $Name
  return (Test-Path -LiteralPath $p) -and ((Get-Content -LiteralPath $p -Raw | ConvertFrom-Json).status -eq $Status)
}
function Worker([string]$Script,[string]$Action){
  $name="${Script}_${Action}_${RunTag}"
  $log=Join-Path $runtime "$name.log";$err=Join-Path $runtime "$name.err"
  if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $err)) { throw 'Use a new RunTag; preserve logs.' }
  $w=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u','-X','faulthandler',"$code\$Script.py",$Action) -WorkingDirectory $code -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru
  $null=$w.Handle
  while (-not $w.HasExited) {
    Write-Output "RAW_PIPELINE_ALIVE stage=$Script action=$Action pid=$($w.Id) $(Get-Date -Format o)"
    Start-Sleep -Seconds 30;$w.Refresh()
  }
  $w.WaitForExit()
  if ($null -ne $w.ExitCode -and $w.ExitCode -ne 0) { throw "Worker failed stage=$Script action=$Action; inspect preserved logs." }
  # A null PowerShell ExitCode is not success: the caller must require its scientific completion gate.
}
try {
  $owners=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'raw_latent_contrast_optimization_seed42_v1\\(stage1|train|evaluate)\.py' }
  if ($owners) { throw 'An owned worker already exists; inspect it, do not duplicate.' }
  if (-not (Gate 'EXTRACTION_ADMISSION.json' 'PASS')) { Worker 'stage1' 'extract' }
  if (-not (Gate 'EXTRACTION_ADMISSION.json' 'PASS')) { throw 'Extraction gate absent' }
  if (-not (Gate 'STAGE1_STATUS.json' 'ALL16_FACTORIAL_SCREEN_AND_FOLD_LOCAL_RECIPES_COMPLETE')) { Worker 'stage1' 'screen' }
  if (-not (Gate 'STAGE1_STATUS.json' 'ALL16_FACTORIAL_SCREEN_AND_FOLD_LOCAL_RECIPES_COMPLETE')) { throw 'Stage1 completion gate absent' }
  if ($StopAfter -eq 'Stage1') { Write-Output 'RAW_STAGE1_VERIFIED_COMPLETE';return }
  if (-not (Gate 'END_TO_END_ADMISSION.json' 'PASS')) { Worker 'train' 'admit' }
  if (-not (Gate 'END_TO_END_ADMISSION.json' 'PASS')) { throw 'Full-patient gradient admission absent' }
  if (-not (Gate 'TRAIN_STATUS.json' 'ALL20_RAW_MODELS_AND_DEPLOYMENT_FROZEN')) { Worker 'train' 'train' }
  if (-not (Gate 'TRAIN_STATUS.json' 'ALL20_RAW_MODELS_AND_DEPLOYMENT_FROZEN')) { throw 'Training freeze absent' }
  if ($StopAfter -eq 'Train') { Write-Output 'RAW_ALL_MODELS_VERIFIED_FROZEN';return }
  Worker 'evaluate' 'all'
  if (-not (Gate 'OUTER_STATUS.json' 'ONE_FROZEN_FIVEFOLD_OUTER_COMPLETE_VERIFIED')) { throw 'Verified Outer completion absent' }
  Write-Output 'RAW_COMPLETE_PIPELINE_VERIFIED'
} finally { $guard.Dispose() }
