param([string]$RunTag='repair01',[string]$SourceCommit='', [switch]$AdmissionOnly)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$runtime='D:\drechan_pooling_capacity_ablation_seed42_v1_runtime';$execution='D:\drechan_poolcap_masked_time_repair_v2'
$python='D:\drechan_temporal_pr_contrast_3seed_v1_env\Scripts\python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1';$env:PCA_SOURCE_COMMIT=$SourceCommit
$guard=[IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
function Gate([string]$name,[string]$status){$p=Join-Path "$runtime\public" ($name+'.json');return (Test-Path -LiteralPath $p)-and((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $status)}
function Worker([string]$name){
 $log=Join-Path $runtime ($name+'_'+$RunTag+'.log');$err=Join-Path $runtime ($name+'_'+$RunTag+'.err')
 if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'Fresh RunTag required'}
 $p=Start-Process -FilePath $python -ArgumentList @('-u','-X','faulthandler',"$execution\entry.py",($name+'.py')) -WorkingDirectory $execution -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$p.Handle
 while(-not $p.HasExited){Write-Output "PCA_REPAIRED_ALIVE stage=$name pid=$($p.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 15;$p.Refresh()}
 $p.WaitForExit();if($null -eq $p.ExitCode -or $p.ExitCode -ne 0){throw ('Failure '+$name+'; preserve and diagnose before resume')}
}
try{
 $os=Get-CimInstance Win32_OperatingSystem
 if($os.FreePhysicalMemory -lt 12*1024*1024){throw 'Insufficient12GiB free RAM'}
 $gpuFree=& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
 if($LASTEXITCODE -ne 0 -or [int]($gpuFree|Select-Object -First 1) -lt 8192){throw 'Insufficient8GiB free GPU RAM'}
 if(Get-CimInstance Win32_Process|Where-Object {$_.Name -eq 'python.exe' -and ($_.CommandLine -match 'drechan_poolcap_masked_time_repair_v1\\entry\.py' -or $_.CommandLine -match 'drechan_pooling_capacity_ablation_seed42_v1\\affinity_entry\.py')}){throw 'Owned worker already active'}
 if(-not(Gate 'ENGINEERING_REPAIR_ADMISSION' 'PASS')){Worker 'admit_repair'}
 if($AdmissionOnly){Write-Output 'PCA_REPAIR_ADMISSION_COMPLETE';exit 0}
 if($SourceCommit -notmatch '^[a-f0-9]{40}$'){throw 'Verified prefit source commit required'}
 if(-not(Gate 'TRAIN_STATUS' 'ALL25_MODEL_CELLS_FROZEN')){Worker 'train'}
 if(-not(Gate 'FREEZE_STATUS' 'ALL25_FROZEN_BEFORE_NEW_OUTER')){Worker 'verify_internal'}
 if(-not(Gate 'PREDICTION_STATUS' 'ALL25_OUTER_CELLS_SEALED')){Worker 'evaluate_outer'}
 if(-not(Gate 'REPORT_STATUS' 'ALL5_FIXED_GROUPS_SEED42_REPORTED')){Worker 'report_outer'}
 if(-not(Gate 'OUTER_STATUS' 'COMPLETE_VERIFIED_PCA_5_GROUP_SEED42_FIVE_FOLD_OUTER')){Worker 'verify_outer'}
 if(-not(Gate 'OUTER_STATUS' 'COMPLETE_VERIFIED_PCA_5_GROUP_SEED42_FIVE_FOLD_OUTER')){throw 'Independent verification missing'}
 Worker 'package_public';Write-Output 'PCA_REPAIRED_FULL_COMPLETION'
}finally{$guard.Dispose()}
