param([string]$RunTag='production01',[ValidateSet('Admission','All')][string]$StopAfter='All',[string]$SourceCommit='')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\drechan_loco_five_models_3seed_v1';$runtime='D:\drechan_loco_five_models_3seed_v1_runtime'
$python='D:\drechan_temporal_pr_contrast_3seed_v1_env\Scripts\python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1';$env:LOCO_SOURCE_COMMIT=$SourceCommit
if($env:KMP_DUPLICATE_LIB_OK){throw 'Unsafe OpenMP bypass forbidden'}
if(-not(Test-Path -LiteralPath $runtime)){New-Item -ItemType Directory -Path $runtime|Out-Null}
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$path,[string]$status){return (Test-Path -LiteralPath $path)-and((Get-Content -LiteralPath $path -Raw|ConvertFrom-Json).status -eq $status)}
function Worker([int]$seed,[string]$script,[string]$stage){
 $env:LOCO_TRAIN_SEED=[string]$seed
 $prefix=if($script -eq 'admit.py'){'admit'}else{$stage}
 $log=Join-Path $runtime ($prefix+'_seed'+$seed+'_'+$RunTag+'.log');$err=Join-Path $runtime ($prefix+'_seed'+$seed+'_'+$RunTag+'.err')
 if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'Use NEW RunTag; old logs retained'}
 $args=@('-u','-X','faulthandler',"$code\affinity_entry.py",$script)
 if($stage){$args+=$stage}
 $w=Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while(-not $w.HasExited){Write-Output "LOCO3_ALIVE seed=$seed stage=$prefix pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 15;$w.Refresh()}
 $w.WaitForExit()
 if($null -eq $w.ExitCode -or $w.ExitCode -ne 0){throw "Worker failed seed=$seed stage=$prefix; diagnose, do not blindly retry"}
}
try{
 if(Get-CimInstance Win32_Process|Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'drechan_loco_five_models_3seed_v1\\affinity_entry\.py'}){throw 'Own worker already active'}
 foreach($s in @(42,43,44)){
  if(-not(Gate "$runtime\seed$s\public\ADMISSION.json" 'PASS')){Worker $s 'admit.py' ''}
 }
 if($StopAfter -eq 'Admission'){Worker 43 'orchestrate.py' 'admission_package';Write-Output 'LOCO3_ADMISSION_COMPLETE';exit 0}
 if($SourceCommit -notmatch '^[a-f0-9]{40}$'){throw 'Pre-fit source/admission push verification required'}
 foreach($s in @(43,44)){
  if(-not(Gate "$runtime\seed$s\public\TRAIN_STATUS.json" 'ALL20_SELECTIONS_COMPLETE')){Worker $s 'pipeline.py' 'train'}
  if(-not(Gate "$runtime\seed$s\public\FREEZE_STATUS.json" 'ALL20_FROZEN_BEFORE_TEST')){Worker $s 'pipeline.py' 'verify_internal'}
 }
 if(-not(Gate "$runtime\public\FREEZE_STATUS.json" 'ALL60_FROZEN_BEFORE_NEW_TEST')){Worker 43 'orchestrate.py' 'freeze_all'}
 foreach($s in @(43,44)){
  if(-not(Gate "$runtime\seed$s\public\PREDICTION_STATUS.json" 'ALL20_TEST_SCORES_SEALED')){Worker $s 'pipeline.py' 'score_test'}
 }
 if(-not(Gate "$runtime\public\PREDICTION_STATUS.json" 'ALL60_SEALED_BEFORE_METRICS')){Worker 43 'orchestrate.py' 'seal_all'}
 if(-not(Gate "$runtime\public\REPORT_STATUS.json" 'ALL5_MODELS_3SEED_REPORTED')){Worker 43 'orchestrate.py' 'report'}
 if(-not(Gate "$runtime\public\OUTER_STATUS.json" 'COMPLETE_VERIFIED_LOCO_5_MODELS_4_SOURCES_3SEED')){Worker 43 'orchestrate.py' 'verify'}
 if(-not(Gate "$runtime\public\OUTER_STATUS.json" 'COMPLETE_VERIFIED_LOCO_5_MODELS_4_SOURCES_3SEED')){throw 'Independent verification required'}
 Worker 43 'orchestrate.py' 'package'
 Write-Output 'LOCO3_ALL60_CELLS_TEST_AND_INDEPENDENT_VERIFICATION_COMPLETE'
}finally{$guard.Dispose()}
