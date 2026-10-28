param([Parameter(Mandatory=$true)][string]$SourceCommit,[string]$RunTag='production01')
$ErrorActionPreference='Stop'
if($SourceCommit -notmatch '^[0-9a-f]{40}$' -or $RunTag -notmatch '^[a-zA-Z0-9_]+$'){throw 'Invalid source/run tag'}
$taskSource=$PSScriptRoot
$taskRuntime='D:\Codex_Research\experiments\drechan_pr_representation_vs_score_v1_runtime'
$taskPython='D:\Codex_Research\environments\m8_feature_pr_baselines_seed42_v1_env\Scripts\python.exe'
[IO.Directory]::CreateDirectory($taskRuntime)|Out-Null
$taskOwner=[IO.File]::Open(($taskRuntime+'\COORDINATOR_LOCK_PRIVATE'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
try {
 $existing=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'drechan_pr_representation_vs_score_v1[\\/]entry.py'})
 if($existing.Count -gt 0){throw 'An existing task worker is alive; do not duplicate'}
 [IO.File]::WriteAllText(($taskRuntime+'\COORDINATOR_PRIVATE.json'),(@{pid=$PID;run_tag=$RunTag;source_commit=$SourceCommit;started_utc=(Get-Date).ToUniversalTime().ToString('o')}|ConvertTo-Json))
 foreach($name in @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')){[Environment]::SetEnvironmentVariable($name,'1','Process')}
 $env:PYTHONUNBUFFERED='1'
 $admission=$taskRuntime+'\public\ADMISSION.json'
 if(-not (Test-Path -LiteralPath ($taskRuntime+'\SOURCE_LOCK.json'))){
  Write-Output ('PR_REPR_ADMISSION_START '+(Get-Date).ToString('o'))
  $taskChild=Start-Process -FilePath $taskPython -ArgumentList @('-u',($taskSource+'\entry.py'),'admit','--source-commit',$SourceCommit) -WindowStyle Hidden -PassThru -RedirectStandardOutput ($taskRuntime+'\admit_'+$RunTag+'.log') -RedirectStandardError ($taskRuntime+'\admit_'+$RunTag+'.err')
  Write-Output ('PR_REPR_ADMISSION_WORKER '+$taskChild.Id)
  $taskChild.WaitForExit()
  if($taskChild.ExitCode -ne 0){throw ('Admission failed with exit '+$taskChild.ExitCode+'; preserve logs and diagnose')}
 }
 if(-not (Test-Path -LiteralPath $admission)){throw 'Admission record unavailable'}
 Write-Output ('PR_REPR_PRODUCTION_START '+(Get-Date).ToString('o'))
 $taskChild=Start-Process -FilePath $taskPython -ArgumentList @('-u',($taskSource+'\entry.py'),'all','--source-commit',$SourceCommit) -WindowStyle Hidden -PassThru -RedirectStandardOutput ($taskRuntime+'\train_'+$RunTag+'.log') -RedirectStandardError ($taskRuntime+'\train_'+$RunTag+'.err')
 Write-Output ('PR_REPR_PRODUCTION_WORKER '+$taskChild.Id)
 $taskChild.WaitForExit()
 if($taskChild.ExitCode -ne 0){throw ('Production failed with exit '+$taskChild.ExitCode+'; preserve exact checkpoints and diagnose')}
 Write-Output ('PR_REPR_COORDINATOR_COMPLETE '+(Get-Date).ToString('o'))
} finally {$taskOwner.Dispose()}
