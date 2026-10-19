param([string]$RunTag='production03',[string]$SourceCommit='')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\drechan_temporal_pr_contrast_3seed_v1';$runtime='D:\drechan_temporal_pr_contrast_3seed_v1_runtime'
$python='D:\drechan_temporal_pr_contrast_3seed_v1_env\Scripts\python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1';$env:TPRC_SOURCE_COMMIT=$SourceCommit
if($env:KMP_DUPLICATE_LIB_OK){throw 'Unsafe duplicate OpenMP bypass is forbidden'}
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$file,[string]$value){$p=Join-Path "$runtime\public" $file;return (Test-Path -LiteralPath $p)-and((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $value)}
function Worker([string]$script,[string[]]$words=@()){
 $label=(@($script)+$words)-join '_';$log=Join-Path $runtime ($label+'_'+$RunTag+'.log');$err=Join-Path $runtime ($label+'_'+$RunTag+'.err')
 if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'Fresh RunTag required; preserve all prior production logs'}
 $w=Start-Process -FilePath $python -ArgumentList (@('-u','-X','faulthandler',"$code\affinity_entry_v2.py",($script+'.py'))+$words) -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while(-not $w.HasExited){Write-Output "TPRC_ISOLATED_ALIVE stage=$label pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 20;$w.Refresh()}
 $w.WaitForExit();if($null -eq $w.ExitCode -or $w.ExitCode -ne 0){throw ('Worker failed '+$label+'; stop and diagnose, never blindly retry')}
}
function Stage([string]$stage){
 if(-not(Gate "$stage/TRAIN_STATUS.json" 'ALL60_MODEL_CELLS_COMPLETE')){Worker 'train' @($stage)}
 if(-not(Gate "$stage/FREEZE_STATUS.json" 'ALL60_FROZEN_BEFORE_SCORES')){Worker 'verify' @('selection',$stage)}
 if(-not(Gate "$stage/REPORT_STATUS.json" 'ALL4_ARMS_3_SEEDS_REPORTED')){if($stage -eq 'final'){Worker 'final_outer'}else{Worker 'assess' @($stage)}}
 if(-not(Gate "$stage/INDEPENDENT_VERIFICATION.json" 'PASS')){Worker 'verify' @('metrics',$stage)}
}
try{
 if(Get-CimInstance Win32_Process|Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'drechan_temporal_pr_contrast_3seed_v1\\(affinity_entry_v2|train|verify|assess|final_outer|package_public)\.py'}){throw 'Owned worker already active'}
 if(-not(Gate 'ADMISSION.json' 'PASS')){throw 'Original admission required'}
 if($SourceCommit -notmatch '^[a-f0-9]{40}$'){throw 'Original verified pre-training source commit required'}
 Worker 'admit_execution'
 Stage 'internal'
 $g=Get-Content -LiteralPath "$runtime\public\internal\GATE.json" -Raw|ConvertFrom-Json
 if($g.continue_final){Stage 'final'}
 Worker 'package_public'
 if($g.continue_final){Write-Output 'TPRC_ALL4_ARMS_3SEED_OUTER_VERIFIED_COMPLETE'}else{Write-Output 'TPRC_INTERNAL_GATE_FAILED_NO_NEW_OUTER'}
}finally{$guard.Dispose()}
