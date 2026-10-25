param(
    [Parameter(Mandatory=$true)][string]$Runtime,
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RunTag,
    [string]$Python='D:\drechan_temporal_pr_contrast_3seed_v1_env\Scripts\python.exe'
)
$ErrorActionPreference='Stop'
$code=$PSScriptRoot
if(-not [IO.Path]::IsPathRooted($Runtime)){throw 'Use an explicit fresh absolute runtime path'}
if(Test-Path -LiteralPath $Runtime){throw 'Preserve existing runtime; use a new directory'}
$active=@(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match 'python' -and $_.CommandLine -like "*$code*" -and
    ($_.CommandLine -like '*run_analysis.py*' -or $_.CommandLine -like '*verify_analysis.py*')
})
if($active.Count -ne 0){throw 'Association worker exists; inspect before starting'}
New-Item -ItemType Directory -Path $Runtime | Out-Null
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES='';$env:PYTHONUNBUFFERED='1'
function Worker([string]$stage,[string[]]$arguments){
    $tag=$stage+'_'+$RunTag
    $log=Join-Path $Runtime ($tag+'.log');$err=Join-Path $Runtime ($tag+'.err')
    if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'Preserve all old logs'}
    $w=Start-Process -FilePath $Python -ArgumentList (@('-u')+$arguments) -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
    $null=$w.Handle
    while(-not $w.HasExited){Write-Output ("ASSOCIATION_STAGE_ALIVE=$tag PID="+$w.Id);Start-Sleep -Seconds 5;$w.Refresh()}
    $w.WaitForExit()
    if($null-eq$w.ExitCode-or$w.ExitCode-ne0){throw "Stage $tag failed; diagnose preserved stderr before retrying"}
}
Worker tests @('-m','unittest','discover','-s',$code,'-p','test_statistics.py','-v')
Worker analysis @("$code\run_analysis.py",'analyze','--runtime',$Runtime)
Worker verification @("$code\verify_analysis.py",'--runtime',$Runtime)
Worker package @("$code\run_analysis.py",'package','--runtime',$Runtime)
Get-Content -LiteralPath (Join-Path $Runtime 'public\STATUS.json')
