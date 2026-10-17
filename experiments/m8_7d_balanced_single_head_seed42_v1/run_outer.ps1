param([string]$RunTag='01')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\m8_7d_balanced_single_head_seed42_v1';$runtime='D:\m8_7d_balanced_single_head_seed42_v1_runtime\final_stage_v1'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Path $runtime -Force|Out-Null
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$name,[string]$status){$p=Join-Path "$runtime\public" $name;return (Test-Path -LiteralPath $p) -and ((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $status)}
function Worker([string]$script){
 $log=Join-Path $runtime "${script}_${RunTag}.log";$err=Join-Path $runtime "${script}_${RunTag}.err"
 if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'Fresh RunTag required'}
 $w=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u','-X','faulthandler',"$code\$script.py") -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while(-not $w.HasExited){Write-Output "BALANCED_SINGLE_OUTER_ALIVE stage=$script pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 20;$w.Refresh()}
 $w.WaitForExit();if($null -ne $w.ExitCode -and $w.ExitCode -ne 0){throw 'Worker failed; diagnose preserved state; no automatic retry'}
}
try{
 if(Get-CimInstance Win32_Process|Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'm8_7d_balanced_single_head_seed42_v1\\(evaluate_outer|report_outer|verify_outer)\.py'}){throw 'Owned worker active'}
 if(-not(Gate 'PREDICTION_STATUS.json' 'ALL10_NEW_PLUS5_REUSED_OUTER_SCORES_SEALED')){Worker 'evaluate_outer'}
 if(-not(Gate 'PREDICTION_STATUS.json' 'ALL10_NEW_PLUS5_REUSED_OUTER_SCORES_SEALED')){throw 'Prediction seal missing'}
 if(-not(Test-Path -LiteralPath "$runtime\public\FINAL_DECISION.json")){Worker 'report_outer'}
 if(-not(Gate 'OUTER_STATUS.json' 'COMPLETE_VERIFIED_FULL80_BALANCED_7D')){Worker 'verify_outer'}
 if(-not(Gate 'OUTER_STATUS.json' 'COMPLETE_VERIFIED_FULL80_BALANCED_7D')){throw 'Independent verification missing'}
 Write-Output 'BALANCED_7D_SINGLE_FULL80_OUTER_PIPELINE_VERIFIED_COMPLETE'
}finally{$guard.Dispose()}
