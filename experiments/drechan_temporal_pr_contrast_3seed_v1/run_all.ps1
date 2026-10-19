param([string]$RunTag='01',[ValidateSet('Admission','Internal','All')][string]$StopAfter='All',[string]$SourceCommit='')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\drechan_temporal_pr_contrast_3seed_v1';$runtime='D:\drechan_temporal_pr_contrast_3seed_v1_runtime'
$python='D:\m8_feature_pr_baselines_seed42_v1_env\Scripts\python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1';$env:TPRC_SOURCE_COMMIT=$SourceCommit
New-Item -ItemType Directory -Path $runtime -Force|Out-Null
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$file,[string]$value){$p=Join-Path "$runtime\public" $file;return (Test-Path -LiteralPath $p)-and((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $value)}
function Worker([string]$script,[string[]]$words=@()){
 $label=(@($script)+$words)-join '_';$log=Join-Path $runtime ($label+'_'+$RunTag+'.log');$err=Join-Path $runtime ($label+'_'+$RunTag+'.err')
 if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'Use fresh RunTag; preserve prior logs'}
 $w=Start-Process -FilePath $python -ArgumentList (@('-u','-X','faulthandler',"$code\$script.py")+$words) -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while(-not $w.HasExited){Write-Output "TPRC_ALIVE stage=$label pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 20;$w.Refresh()}
 $w.WaitForExit();if($null -eq $w.ExitCode -or $w.ExitCode -ne 0){throw ('Worker failed '+$label+'; diagnose before any resume')}
}
function Stage([string]$stage){
 if(-not(Gate "$stage/TRAIN_STATUS.json" 'ALL60_MODEL_CELLS_COMPLETE')){Worker 'train' @($stage)}
 if(-not(Gate "$stage/FREEZE_STATUS.json" 'ALL60_FROZEN_BEFORE_SCORES')){Worker 'verify' @('selection',$stage)}
 if(-not(Gate "$stage/REPORT_STATUS.json" 'ALL4_ARMS_3_SEEDS_REPORTED')){
  if($stage -eq 'final'){Worker 'final_outer'}else{Worker 'assess' @($stage)}
 }
 if(-not(Gate "$stage/INDEPENDENT_VERIFICATION.json" 'PASS')){Worker 'verify' @('metrics',$stage)}
}
try{
 if(Get-CimInstance Win32_Process|Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'drechan_temporal_pr_contrast_3seed_v1\\(admit|train|verify|assess|final_outer|package_public)\.py'}){throw 'Owned worker already active'}
 if(-not(Gate 'ADMISSION.json' 'PASS')){Worker 'admit'}
 if(-not(Gate 'ADMISSION.json' 'PASS')){throw 'Admission missing'}
 if($StopAfter -eq 'Admission'){Write-Output 'TPRC_ADMISSION_COMPLETE';exit 0}
 if($SourceCommit -notmatch '^[a-f0-9]{40}$'){throw 'Verified source push required before formal fits'}
 Stage 'internal'
 if($StopAfter -eq 'Internal'){Write-Output 'TPRC_INTERNAL_COMPLETE';exit 0}
 $g=Get-Content -LiteralPath "$runtime\public\internal\GATE.json" -Raw|ConvertFrom-Json
 if($g.continue_final){Stage 'final'}
 Worker 'package_public'
 if($g.continue_final){Write-Output 'TPRC_ALL4_ARMS_3SEED_OUTER_VERIFIED_COMPLETE'}else{Write-Output 'TPRC_INTERNAL_GATE_FAILED_NO_NEW_OUTER'}
}finally{$guard.Dispose()}
