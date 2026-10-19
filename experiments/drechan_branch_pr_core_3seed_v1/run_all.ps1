param([ValidateSet('Train','Finish')][string]$Mode='Train',[string]$RunTag='production01',[string]$SourceCommit='')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\drechan_branch_pr_core_3seed_v1';$runtime='D:\drechan_branch_pr_core_3seed_v1_runtime'
$python='D:\drechan_temporal_pr_contrast_3seed_v1_env\Scripts\python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1';$env:CORE_SOURCE_COMMIT=$SourceCommit
if($env:KMP_DUPLICATE_LIB_OK){throw 'Unsafe duplicate OpenMP bypass forbidden'}
if($SourceCommit -notmatch '^[a-f0-9]{40}$'){throw 'Verified source push required'}
if(-not(Test-Path -LiteralPath $runtime)){New-Item -ItemType Directory -Path $runtime|Out-Null}
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$name,[string]$value){$p=Join-Path "$runtime\public" ($name+'.json');return(Test-Path -LiteralPath $p)-and((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $value)}
function Worker([string]$script){
 $log=Join-Path $runtime ($script+'_'+$RunTag+'.log');$err=Join-Path $runtime ($script+'_'+$RunTag+'.err')
 if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'Fresh RunTag required; preserve old logs'}
 $w=Start-Process -FilePath $python -ArgumentList @('-u','-X','faulthandler',"$code\affinity_entry.py",($script+'.py')) -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while(-not $w.HasExited){Write-Output "CORE_ALIVE stage=$script pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 15;$w.Refresh()}
 $w.WaitForExit();if($null -eq $w.ExitCode -or $w.ExitCode -ne 0){throw ('Failed '+$script+'; stop for diagnosis, no blind retry')}
}
try{
 if(Get-CimInstance Win32_Process|Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'drechan_branch_pr_core_3seed_v1\\affinity_entry\.py'}){throw 'Owned core worker already active'}
 if($Mode -eq 'Train'){
  $os=Get-CimInstance Win32_OperatingSystem
  if($os.FreePhysicalMemory -lt 12*1024*1024){throw 'Insufficient12GiB free system-memory admission; do not start parallel worker'}
  $gpuFree=& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
  if($LASTEXITCODE -ne 0 -or [int]($gpuFree|Select-Object -First 1) -lt 8192){throw 'Insufficient8GiB free GPU-memory admission'}
  if(-not(Gate 'ADMISSION' 'PASS')){Worker 'admit'}
  if(-not(Gate 'NEW_TRAIN_STATUS' 'ALL75_NEW_COMPLETE')){Worker 'train'}
  Write-Output 'CORE_NEW75_COMPLETE_RUN_FINISH_ONLY_AFTER_PARENT_VERIFIED'
 }else{
  if((-not(Gate 'ADMISSION' 'PASS')) -or (-not(Gate 'NEW_TRAIN_STATUS' 'ALL75_NEW_COMPLETE'))){throw 'Core training not complete'}
  Worker 'finish'
 }
}finally{$guard.Dispose()}
