param([string]$RunTag='production01',[ValidateSet('Admission','Train','All')][string]$StopAfter='All',[string]$SourceCommit='')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\drechan_f0_no_lategate_ablation_seed42_v1';$runtime='D:\drechan_f0_no_lategate_ablation_seed42_v1_runtime'
$python='D:\drechan_temporal_pr_contrast_3seed_v1_env\Scripts\python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1';$env:NLG_SOURCE_COMMIT=$SourceCommit
if($env:KMP_DUPLICATE_LIB_OK){throw 'Unsafe duplicate OpenMP bypass forbidden'}
if(-not(Test-Path -LiteralPath $runtime)){New-Item -ItemType Directory -Path $runtime | Out-Null}
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$name,[string]$value){
 $p=Join-Path "$runtime\public" ($name+'.json')
 return (Test-Path -LiteralPath $p)-and((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $value)
}
function Worker([string]$script){
 $log=Join-Path $runtime ($script+'_'+$RunTag+'.log');$err=Join-Path $runtime ($script+'_'+$RunTag+'.err')
 if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'Fresh RunTag required; preserve prior logs'}
 $w=Start-Process -FilePath $python -ArgumentList @('-u','-X','faulthandler',"$code\affinity_entry.py",($script+'.py')) -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while(-not $w.HasExited){Write-Output "NLG_ALIVE stage=$script pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 15;$w.Refresh()}
 $w.WaitForExit();if($null -eq $w.ExitCode -or $w.ExitCode -ne 0){throw ('Worker failed '+$script+'; diagnose engineering failure before exact resume')}
}
try{
 $os=Get-CimInstance Win32_OperatingSystem
 if($os.FreePhysicalMemory -lt 12*1024*1024){throw 'Insufficient12GiB free RAM'}
 $gpuFree=& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
 if($LASTEXITCODE -ne 0 -or [int]($gpuFree|Select-Object -First 1) -lt 8192){throw 'Insufficient8GiB free GPU RAM'}
 if(Get-CimInstance Win32_Process|Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'drechan_f0_no_lategate_ablation_seed42_v1\\affinity_entry\.py'}){throw 'Owned worker already active'}
 if(-not(Gate 'ADMISSION' 'PASS')){Worker 'admit'}
 if(-not(Gate 'ADMISSION' 'PASS')){throw 'Admission missing'}
 if($StopAfter -eq 'Admission'){Write-Output 'NLG_ADMISSION_COMPLETE';exit 0}
 if($SourceCommit -notmatch '^[a-f0-9]{40}$'){throw 'Verified pre-fit source push required'}
 if(-not(Gate 'TRAIN_STATUS' 'ALL90_MODEL_CELLS_FROZEN')){Worker 'train'}
 if($StopAfter -eq 'Train'){Write-Output 'NLG_TRAIN_SELECTIONS_COMPLETE';exit 0}
 if(-not(Gate 'FREEZE_STATUS' 'ALL90_FROZEN_BEFORE_NEW_OUTER')){Worker 'verify_internal'}
 if(-not(Gate 'PREDICTION_STATUS' 'ALL90_OUTER_CELLS_SEALED')){Worker 'evaluate_outer'}
 if(-not(Gate 'REPORT_STATUS' 'ALL18_FIXED_GROUPS_SEED42_REPORTED')){Worker 'report_outer'}
 if(-not(Gate 'OUTER_STATUS' 'COMPLETE_VERIFIED_NLG_18_GROUP_SEED42_FIVE_FOLD_OUTER')){Worker 'verify_outer'}
 if(-not(Gate 'OUTER_STATUS' 'COMPLETE_VERIFIED_NLG_18_GROUP_SEED42_FIVE_FOLD_OUTER')){throw 'Independent final verification missing'}
 Worker 'package_public'
 Write-Output 'NLG_18_GROUPS_SEED42_ORIGINAL80_FIVE_FOLD_OUTER_COMPLETE_NOT_INDEPENDENT_CONFIRMATION'
}finally{$guard.Dispose()}
