param([string]$RunTag='01',[ValidateSet('Admission','All')][string]$StopAfter='All',[string]$SourceCommit='')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\m8_7d_f0_complete_ablation_seed42_v1';$runtime='D:\m8_7d_f0_complete_ablation_seed42_v1_runtime'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1';$env:F0_ABLATION_SOURCE_COMMIT=$SourceCommit
New-Item -ItemType Directory -Path $runtime -Force|Out-Null
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$name,[string]$status){$p=Join-Path "$runtime\public" $name;return (Test-Path -LiteralPath $p)-and((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $status)}
function Worker([string]$script){
 $log=Join-Path $runtime ($script+'_'+$RunTag+'.log');$err=Join-Path $runtime ($script+'_'+$RunTag+'.err')
 if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'Fresh RunTag required; preserve existing logs'}
 $w=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u','-X','faulthandler',"$code\$script.py") -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while(-not $w.HasExited){Write-Output "F0_ABLATION_ALIVE stage=$script pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 20;$w.Refresh()}
 $w.WaitForExit();if($null -eq $w.ExitCode -or $w.ExitCode -ne 0){throw ('Worker failed '+$script+'; preserve and diagnose, no blind retry')}
}
try{
 if(Get-CimInstance Win32_Process|Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'm8_7d_f0_complete_ablation_seed42_v1\\(admit|train|verify_internal|evaluate_outer|report_outer|verify_outer)\.py'}){throw 'Owned worker active; no duplicate'}
 if(-not(Gate 'ADMISSION.json' 'PASS')){Worker 'admit'}
 if(-not(Gate 'ADMISSION.json' 'PASS')){throw 'Admission missing'}
 if($StopAfter -eq 'Admission'){Write-Output 'F0_COMPLETE_ABLATION_ADMISSION_COMPLETE';exit 0}
 if($SourceCommit -notmatch '^[a-f0-9]{40}$'){throw 'Verified pre-training source commit required'}
 if(-not(Gate 'TRAIN_STATUS.json' 'ALL90_MODEL_CELLS_FROZEN')){Worker 'train'}
 if(-not(Gate 'TRAIN_STATUS.json' 'ALL90_MODEL_CELLS_FROZEN')){throw 'Training terminal missing'}
 if(-not(Gate 'FREEZE_STATUS.json' 'ALL90_FROZEN_BEFORE_NEW_OUTER')){Worker 'verify_internal'}
 if(-not(Gate 'FREEZE_STATUS.json' 'ALL90_FROZEN_BEFORE_NEW_OUTER')){throw 'Full freeze missing'}
 if(-not(Gate 'PREDICTION_STATUS.json' 'ALL90_OUTER_CELLS_SEALED')){Worker 'evaluate_outer'}
 if(-not(Gate 'PREDICTION_STATUS.json' 'ALL90_OUTER_CELLS_SEALED')){throw 'Score seal missing'}
 if(-not(Gate 'REPORT_STATUS.json' 'ALL18_FIXED_ARMS_REPORTED')){Worker 'report_outer'}
 if(-not(Gate 'OUTER_STATUS.json' 'COMPLETE_VERIFIED_F0_18_ARM_FIVE_FOLD_OUTER')){Worker 'verify_outer'}
 if(-not(Gate 'OUTER_STATUS.json' 'COMPLETE_VERIFIED_F0_18_ARM_FIVE_FOLD_OUTER')){throw 'Independent verification missing'}
 Write-Output 'F0_COMPLETE18_FIVE_FOLD_OUTER_VERIFIED_COMPLETE'
}finally{$guard.Dispose()}
