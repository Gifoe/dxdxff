param([Parameter(Mandatory=$true)][string]$SourceCommit,
      [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RunTag,
      [ValidateSet('Admission','All')][string]$StopAfter='All')
$ErrorActionPreference='Stop'
$code='D:\drechan_sixfeature_ez_prediction_v1'
$runtime='D:\drechan_sixfeature_ez_prediction_v1_runtime'
$python='D:\drechan_temporal_pr_contrast_3seed_v1_env\Scripts\python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES='';$env:PYTHONUNBUFFERED='1';$env:PREDICTION_SOURCE_COMMIT=$SourceCommit
if($SourceCommit -notmatch '^[a-f0-9]{40}$'){throw 'Source must be pushed and remote-verified before fit'}
if(-not(Test-Path -LiteralPath $runtime)){New-Item -ItemType Directory -Path $runtime | Out-Null}
$guard=[IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
function Worker([string]$stage,[string[]]$arguments){
    $log=Join-Path $runtime ($stage+'_'+$RunTag+'.log');$err=Join-Path $runtime ($stage+'_'+$RunTag+'.err')
    if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'Use NEW RunTag, preserve existing logs'}
    $w=Start-Process -FilePath $python -ArgumentList (@('-u')+$arguments) -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
    $null=$w.Handle
    while(-not $w.HasExited){Write-Output ("PREDICTION_STAGE_ALIVE=$stage PID="+$w.Id);Start-Sleep -Seconds 10;$w.Refresh()}
    $w.WaitForExit();if($null-eq$w.ExitCode-or$w.ExitCode-ne0){throw "Stage $stage failed; preserve/diagnose logs before resume"}
}
try{
    if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match 'python' -and $_.CommandLine -like '*D:\drechan_sixfeature_ez_prediction_v1\run.py*'}){throw 'Own worker already active'}
    Worker tests @('-m','unittest','discover','-s',$code,'-p','test_math.py','-v')
    if(-not(Test-Path -LiteralPath "$runtime\public\ADMISSION.json")){Worker admission @("$code\run.py",'admission','--runtime',$runtime)}
    if($StopAfter-eq'Admission'){exit 0}
    Worker train @("$code\run.py",'train','--runtime',$runtime)
    Worker freeze @("$code\run.py",'freeze','--runtime',$runtime)
    Worker score @("$code\run.py",'score','--runtime',$runtime)
    Worker report @("$code\run.py",'report','--runtime',$runtime)
    Worker verify @("$code\run.py",'verify','--runtime',$runtime)
    Worker package @("$code\run.py",'package','--runtime',$runtime)
    Get-Content -LiteralPath "$runtime\public\STATUS.json"
    Get-Content -LiteralPath "$runtime\package_$RunTag.log" -Tail 1
}finally{$guard.Dispose()}
