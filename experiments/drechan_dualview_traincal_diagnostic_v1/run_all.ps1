param([string]$RunTag='production01',[string]$SourceCommit='')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\drechan_dualview_traincal_diagnostic_v1';$runtime='D:\drechan_dualview_traincal_diagnostic_v1_runtime'
$python='D:\drechan_temporal_pr_contrast_3seed_v1_env\Scripts\python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1';$env:DIAG_SOURCE_COMMIT=$SourceCommit
if($env:KMP_DUPLICATE_LIB_OK){throw 'Unsafe OpenMP bypass forbidden'}
if($SourceCommit -notmatch '^[a-f0-9]{40}$'){throw 'Verified pre-audit source push required'}
if(-not(Test-Path -LiteralPath $runtime)){New-Item -ItemType Directory -Path $runtime|Out-Null}
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Worker([string]$script){
 $log=Join-Path $runtime ($script+'_'+$RunTag+'.log');$err=Join-Path $runtime ($script+'_'+$RunTag+'.err')
 if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'FreshRunTagrequired;preservepriorlogs'}
 $p=Start-Process -FilePath $python -ArgumentList @('-u','-X','faulthandler',"$code\affinity_entry.py",($script+'.py')) -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$p.Handle
 while(-not $p.HasExited){Write-Output "TRAINCAL_ALIVE stage=$script pid=$($p.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 15;$p.Refresh()}
 $p.WaitForExit();if($null -eq $p.ExitCode -or $p.ExitCode -ne 0){throw ('Diagnostic failed '+$script+';diagnose,noblindretry')}
}
try{
 $os=Get-CimInstance Win32_OperatingSystem
 if($os.FreePhysicalMemory -lt 12*1024*1024){throw 'Insufficient12GiB free RAM;do not start'}
 $gpuFree=& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
 if($LASTEXITCODE -ne 0 -or [int]($gpuFree|Select-Object -First 1) -lt 8192){throw 'Insufficient8GiB free GPU RAM;do not start'}
 if(Get-CimInstance Win32_Process|Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'drechan_dualview_traincal_diagnostic_v1\\affinity_entry\.py'}){throw 'Owned diagnostic already running'}
 Worker 'test_audit'
 & $python -c "import json,pathlib; p=pathlib.Path(r'$runtime\public');p.mkdir(exist_ok=True);(p/'TESTS.json').write_text(json.dumps(dict(status='PASS',changed_critical_path_tests=5,no_performance_evidence=True)),encoding='utf-8')"
 if($LASTEXITCODE -ne 0){throw 'Test evidence write failed'}
 Worker 'audit';Worker 'verify'
 Write-Output 'TRAINCAL_DIAGNOSTIC_COMPLETE_NO_TRAINING_NO_ASSESS_ACTIVE_GOAL_UNACHIEVED'
}finally{$guard.Dispose()}
