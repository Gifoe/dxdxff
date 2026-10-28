param([string]$RunTag='admission01',[ValidateSet('Admission','StageA','StageB','All')][string]$StopAfter='Admission',[string]$SourceCommit='')
$ErrorActionPreference='Stop'
$taskCode=$PSScriptRoot
$taskRuntime='D:\Codex_Research\experiments\drechan_pr2_quick_ictal_v1_runtime'
$taskPython='D:\Codex_Research\environments\drechan_temporal_pr_contrast_3seed_v1_env\Scripts\python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1';$env:PR2_SOURCE_COMMIT=$SourceCommit
if($env:KMP_DUPLICATE_LIB_OK){throw 'Unsafe OpenMP bypass present'}
New-Item -ItemType Directory -Path $taskRuntime -Force|Out-Null
$taskGuard=[IO.File]::Open((Join-Path $taskRuntime 'coordinator.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
function Done([string]$file){Test-Path -LiteralPath (Join-Path $taskRuntime $file)}
function Worker([string]$script,[string[]]$extra=@()){
 $taskLog=Join-Path $taskRuntime ($script+'_'+$RunTag+'.log');$taskErr=Join-Path $taskRuntime ($script+'_'+$RunTag+'.err')
 if((Test-Path -LiteralPath $taskLog)-or(Test-Path -LiteralPath $taskErr)){throw 'RunTag already used; preserve previous logs'}
 $taskProc=Start-Process -FilePath $taskPython -ArgumentList (@('-u','-X','faulthandler',(Join-Path $taskCode ($script+'.py')))+$extra) -WorkingDirectory $taskCode -WindowStyle Hidden -PassThru -RedirectStandardOutput $taskLog -RedirectStandardError $taskErr
 $null=$taskProc.Handle
 while(-not $taskProc.HasExited){Write-Output "PR2_ALIVE stage=$script pid=$($taskProc.Id) time=$(Get-Date -Format o)";Start-Sleep -Seconds 15;$taskProc.Refresh()}
 $taskProc.WaitForExit();if($null -eq $taskProc.ExitCode -or $taskProc.ExitCode -ne 0){throw "Worker $script failed; diagnose before engineering resume"}
}
try{
 $taskOS=Get-CimInstance Win32_OperatingSystem
 if($taskOS.FreePhysicalMemory -lt 12*1024*1024){throw 'Need12GiB free host RAM'}
 if((Get-PSDrive D).Free -lt 2GB){throw 'Need2GiB free D disk'}
 if(-not(Done 'public\ADMISSION.json')){Worker 'admit'}
 if($StopAfter -eq 'Admission'){Write-Output 'PR2_ADMISSION_COMPLETE';exit 0}
 if($SourceCommit -notmatch '^[a-f0-9]{40}$'){throw 'Source commit required before formal training'}
 if($StopAfter -in @('StageA','All')){
  if(-not(Done 'A_FREEZE.json')){Worker 'train_svm_plugin'}
  if(-not(Done 'A_SCORE_SEAL.json')){Worker 'evaluate' @('A')}
  if(-not(Done 'public\A_VERIFIED.json')){Worker 'verify' @('A')}
  if($StopAfter -eq 'StageA'){Write-Output 'PR2_STAGE_A_VERIFIED';exit 0}
 }
 if(-not(Done 'public\A_VERIFIED.json')){throw 'StageA must finish first'}
 if(-not(Done 'B_FREEZE.json')){Worker 'train_drechan_joint'}
 if(-not(Done 'B_SCORE_SEAL.json')){Worker 'evaluate' @('B')}
 if(-not(Done 'public\B_VERIFIED.json')){Worker 'verify' @('B')}
 Worker 'report'
 Write-Output 'PR2_COMPLETE_VERIFIED'
}finally{$taskGuard.Dispose()}
