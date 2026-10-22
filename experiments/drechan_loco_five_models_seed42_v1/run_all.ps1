param([string]$RunTag='production01',[ValidateSet('Admission','All')][string]$StopAfter='All',[string]$SourceCommit='')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\drechan_loco_five_models_seed42_v1';$runtime='D:\drechan_loco_five_models_seed42_v1_runtime'
$python='D:\drechan_temporal_pr_contrast_3seed_v1_env\Scripts\python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1';$env:LOCO_SOURCE_COMMIT=$SourceCommit
if($env:KMP_DUPLICATE_LIB_OK){throw 'Unsafe OpenMP bypass forbidden'}
if(-not(Test-Path -LiteralPath $runtime)){New-Item -ItemType Directory -Path $runtime | Out-Null}
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$name,[string]$status){
 $p=Join-Path "$runtime\public" ($name+'.json')
 return (Test-Path -LiteralPath $p)-and((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $status)
}
function Worker([string]$stage){
 $log=Join-Path $runtime ($stage+'_'+$RunTag+'.log');$err=Join-Path $runtime ($stage+'_'+$RunTag+'.err')
 if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'Use NEW RunTag; preserve old logs'}
 $args=@('-u','-X','faulthandler',"$code\affinity_entry.py")
 if($stage -eq 'admit'){$args+='admit.py'}else{$args+=@('pipeline.py',$stage)}
 $w=Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while(-not $w.HasExited){Write-Output "LOCO_ALIVE stage=$stage pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 15;$w.Refresh()}
 $w.WaitForExit();if($null -eq $w.ExitCode -or $w.ExitCode -ne 0){throw "Worker failed $stage; diagnose before exact resume"}
}
try{
 if(Get-CimInstance Win32_Process|Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'drechan_loco_five_models_seed42_v1\\affinity_entry\.py'}){throw 'Owned worker already active'}
 if(-not(Gate 'ADMISSION' 'PASS')){Worker 'admit'}
 if(-not(Gate 'ADMISSION' 'PASS')){throw 'Admission missing'}
 if($StopAfter -eq 'Admission'){Write-Output 'LOCO_ADMISSION_COMPLETE';exit 0}
 if($SourceCommit -notmatch '^[a-f0-9]{40}$'){throw 'Verified pre-fit source push required'}
 if(-not(Gate 'TRAIN_STATUS' 'ALL20_SELECTIONS_COMPLETE')){Worker 'train'}
 if(-not(Gate 'FREEZE_STATUS' 'ALL20_FROZEN_BEFORE_TEST')){Worker 'verify_internal'}
 if(-not(Gate 'PREDICTION_STATUS' 'ALL20_TEST_SCORES_SEALED')){Worker 'score_test'}
 if(-not(Gate 'REPORT_STATUS' 'ALL5_MODELS_REPORTED')){Worker 'report'}
 if(-not(Gate 'OUTER_STATUS' 'COMPLETE_VERIFIED_LOCO_5_MODELS_4_SOURCES_SEED42')){Worker 'verify'}
 if(-not(Gate 'OUTER_STATUS' 'COMPLETE_VERIFIED_LOCO_5_MODELS_4_SOURCES_SEED42')){throw 'Independent verification missing'}
 Worker 'package';Write-Output 'LOCO_ALL20_FITS_TEST_AND_INDEPENDENT_VERIFICATION_COMPLETE'
}finally{$guard.Dispose()}
