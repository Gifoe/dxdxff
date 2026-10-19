param([string]$RunTag='production01',[string]$SourceCommit='')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\drechan_branch_pr_residual_3seed_v1';$runtime='D:\drechan_branch_pr_residual_3seed_v1_runtime'
$python='D:\drechan_temporal_pr_contrast_3seed_v1_env\Scripts\python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1';$env:BPR_SOURCE_COMMIT=$SourceCommit
if($env:KMP_DUPLICATE_LIB_OK){throw 'Unsafe duplicate OpenMP bypass forbidden'}
if($SourceCommit -notmatch '^[a-f0-9]{40}$'){throw 'Verified GitHub pre-training source commit required'}
if(-not(Test-Path -LiteralPath $runtime)){New-Item -ItemType Directory -Path $runtime | Out-Null}
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$name,[string]$value){
 $p=Join-Path "$runtime\public" ($name+'.json')
 return (Test-Path -LiteralPath $p)-and((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $value)
}
function Worker([string]$script,[string[]]$words=@()){
 $label=(@($script)+$words)-join '_';$log=Join-Path $runtime ($label+'_'+$RunTag+'.log');$err=Join-Path $runtime ($label+'_'+$RunTag+'.err')
 if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'Fresh RunTag required; preserve prior logs'}
 $w=Start-Process -FilePath $python -ArgumentList (@('-u','-X','faulthandler',"$code\affinity_entry.py",($script+'.py'))+$words) -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while(-not $w.HasExited){Write-Output "BPR_ALIVE stage=$label pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 15;$w.Refresh()}
 $w.WaitForExit();if($null -eq $w.ExitCode -or $w.ExitCode -ne 0){throw ('Worker failed '+$label+'; stop for genuine diagnosis, no blind retry')}
}
try{
 if(Get-CimInstance Win32_Process|Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'drechan_branch_pr_residual_3seed_v1\\affinity_entry\.py'}){throw 'Owned worker already active'}
 if(-not(Gate 'ADMISSION' 'PASS')){Worker 'admit'}
 if(-not(Gate 'TRAIN_STATUS' 'ALL90_COMPLETE')){Worker 'train'}
 if(-not(Gate 'FREEZE_STATUS' 'ALL90_FROZEN')){Worker 'verify' @('selection')}
 if(-not(Gate 'REPORT_STATUS' 'ALL6_ARMS_3SEEDS_REPORTED')){Worker 'assess'}
 if(-not(Gate 'INDEPENDENT_VERIFICATION' 'PASS')){Worker 'verify' @('metrics')}
 Worker 'package_public'
 Write-Output 'BPR_INTERNAL_STAGE_VERIFIED_NO_NEW_OUTER_ACTIVE_GOAL_NOT_YET_ACHIEVED'
}finally{$guard.Dispose()}
