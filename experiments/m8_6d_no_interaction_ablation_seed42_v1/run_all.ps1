param([string]$RunTag='01',[ValidateSet('Admission','Train','All')][string]$StopAfter='All',[string]$SourceCommit='')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\m8_6d_no_interaction_ablation_seed42_v1';$runtime='D:\m8_6d_no_interaction_ablation_seed42_v1_runtime'
$python='D:\m8_feature_pr_baselines_seed42_v1_env\Scripts\python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1';$env:NI_SOURCE_COMMIT=$SourceCommit
New-Item -ItemType Directory -Path $runtime -Force|Out-Null
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$file,[string]$value){$p=Join-Path "$runtime\public" $file;return (Test-Path -LiteralPath $p)-and((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $value)}
function Worker([string]$script){
 $log=Join-Path $runtime ($script+'_'+$RunTag+'.log');$err=Join-Path $runtime ($script+'_'+$RunTag+'.err')
 if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'Use fresh RunTag; preserve prior logs'}
 $w=Start-Process -FilePath $python -ArgumentList @('-u','-X','faulthandler',"$code\$script.py") -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while(-not $w.HasExited){Write-Output "NI_ALIVE stage=$script pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 20;$w.Refresh()}
 $w.WaitForExit();if($null -eq $w.ExitCode -or $w.ExitCode -ne 0){throw ('Worker failed '+$script+'; diagnose before any resume')}
}
try{
 if(Get-CimInstance Win32_Process|Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'm8_6d_no_interaction_ablation_seed42_v1\\(admit|train|verify_internal|evaluate_outer|report_outer|verify_outer)\.py'}){throw 'Owned worker already active'}
 if(-not(Gate 'ADMISSION.json' 'PASS')){Worker 'admit'}
 if(-not(Gate 'ADMISSION.json' 'PASS')){throw 'Admission missing'}
 if($StopAfter -eq 'Admission'){Write-Output 'NI_ADMISSION_COMPLETE';exit 0}
 if($SourceCommit -notmatch '^[a-f0-9]{40}$'){throw 'Verified source push required before formal fits'}
 if(-not(Gate 'TRAIN_STATUS.json' 'ALL70_MODEL_CELLS_FROZEN')){Worker 'train'}
 if($StopAfter -eq 'Train'){Write-Output 'NI_TRAIN_SELECTIONS_COMPLETE';exit 0}
 if(-not(Gate 'FREEZE_STATUS.json' 'ALL70_FROZEN_BEFORE_NEW_OUTER')){Worker 'verify_internal'}
 if(-not(Gate 'PREDICTION_STATUS.json' 'ALL70_OUTER_CELLS_SEALED')){Worker 'evaluate_outer'}
 if(-not(Gate 'REPORT_STATUS.json' 'ALL14_FIXED_GROUPS_REPORTED')){Worker 'report_outer'}
 if(-not(Gate 'OUTER_STATUS.json' 'COMPLETE_VERIFIED_NI_14_GROUP_FIVE_FOLD_OUTER')){Worker 'verify_outer'}
 if(-not(Gate 'OUTER_STATUS.json' 'COMPLETE_VERIFIED_NI_14_GROUP_FIVE_FOLD_OUTER')){throw 'Final verification missing'}
 Worker 'package_public'
 Write-Output 'NI_14_GROUPS_ALL_FIVE_FOLD_OUTER_VERIFIED_COMPLETE'
}finally{$guard.Dispose()}
