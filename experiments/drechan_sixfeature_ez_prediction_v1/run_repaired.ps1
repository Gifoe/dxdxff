param([Parameter(Mandatory=$true)][string]$SourceCommit,
      [Parameter(Mandatory=$true)][string]$EngineeringCommit,
      [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RunTag)
$ErrorActionPreference='Stop'
$code='D:\drechan_sixfeature_ez_prediction_v1'
$runtime='D:\drechan_sixfeature_ez_prediction_v1_runtime'
$python='D:\drechan_temporal_pr_contrast_3seed_v1_env\Scripts\python.exe'
if($SourceCommit-ne'caceb44597097f0f4baea49d10a5c543e2bb706f'-or$EngineeringCommit-notmatch'^[a-f0-9]{40}$'){throw 'Original and engineering sources must be remote-verified'}
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES='';$env:PYTHONUNBUFFERED='1';$env:PREDICTION_SOURCE_COMMIT=$SourceCommit
$env:PREDICTION_ENGINEERING_COMMIT=$EngineeringCommit
$guard=[IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
function Worker([string]$stage,[string[]]$arguments){
    $log=Join-Path $runtime ($stage+'_'+$RunTag+'.log');$err=Join-Path $runtime ($stage+'_'+$RunTag+'.err')
    if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'Use NEW RunTag, preserve existing logs'}
    $w=Start-Process -FilePath $python -ArgumentList (@('-u')+$arguments) -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
    $null=$w.Handle
    while(-not $w.HasExited){Write-Output ("PREDICTION_STAGE_ALIVE=$stage PID="+$w.Id);Start-Sleep -Seconds 10;$w.Refresh()}
    $w.WaitForExit();if($null-eq$w.ExitCode-or$w.ExitCode-ne0){throw "Stage $stage failed; preserve/diagnose before action"}
}
try{
    if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match 'python' -and $_.CommandLine -like '*D:\drechan_sixfeature_ez_prediction_v1\*'}){throw 'Own worker already active'}
    Worker adapter_test @('-m','unittest','discover','-s',$code,'-p','test_audit_adapter.py','-v')
    Worker train @("$code\audit_adapter.py",'train','--runtime',$runtime)
    foreach($stage in @('freeze','score','report','verify','package')){Worker $stage @("$code\run.py",$stage,'--runtime',$runtime)}
    Get-Content -LiteralPath "$runtime\public\STATUS.json"
    Get-Content -LiteralPath "$runtime\package_$RunTag.log" -Tail 1
}finally{$guard.Dispose()}
