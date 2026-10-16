param([string]$RunTag='01')
$ErrorActionPreference='Stop'
$code='D:\raw_latent_contrast_optimization_seed42_v1'
$runtime='D:\raw_latent_contrast_optimization_seed42_v1_runtime'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$owners=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'raw_latent_contrast_optimization_seed42_v1.*stage1.py' }
if ($owners) { throw 'A stage1 worker already exists; inspect it, do not duplicate.' }
foreach ($action in @('extract','screen')) {
  $log=Join-Path $runtime "stage1_${action}_${RunTag}.log"
  $err=Join-Path $runtime "stage1_${action}_${RunTag}.err"
  if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $err)) { throw 'Use a new RunTag; never overwrite a log.' }
  $worker=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u',"$code\stage1.py",$action) -WorkingDirectory $code -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru
  while (-not $worker.HasExited) {
    Write-Output "RAW_STAGE1_ALIVE action=$action pid=$($worker.Id) $(Get-Date -Format o)"
    Start-Sleep -Seconds 30
    $worker.Refresh()
  }
  $worker.WaitForExit()
  if ($worker.ExitCode -ne 0) { throw "Stage1 $action failed, exit=$($worker.ExitCode); inspect preserved files before resuming." }
}
Write-Output 'RAW_STAGE1_PIPELINE_COMPLETE'
