param([string]$RunTag='01',[ValidateSet('Admit','All')][string]$Stage='All')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\m8_7d_frozen_candidate_outer_audit_seed42_v1';$runtime='D:\m8_7d_frozen_candidate_outer_audit_seed42_v1_runtime'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Path $runtime -Force|Out-Null
$guard=[System.IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Gate([string]$name,[string]$status){$p=Join-Path "$runtime\public" $name;return (Test-Path -LiteralPath $p) -and ((Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).status -eq $status)}
function Worker([string]$script){
 $log=Join-Path $runtime ($script+'_'+$RunTag+'.log');$err=Join-Path $runtime ($script+'_'+$RunTag+'.err')
 if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'Fresh RunTag required'}
 $w=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u','-X','faulthandler',"$code\$script.py") -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while(-not $w.HasExited){Write-Output "FROZEN_OUTER_ALIVE stage=$script pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 20;$w.Refresh()}
 $w.WaitForExit();if($null -eq $w.ExitCode -or $w.ExitCode -ne 0){throw ('Worker failed '+$script+'; preserve and diagnose, no automatic retry')}
}
try{
 if(Get-CimInstance Win32_Process|Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'm8_7d_frozen_candidate_outer_audit_seed42_v1\\(admit|evaluate_outer|report_outer|verify_outer)\.py'}){throw 'Owned worker already active'}
 if(-not(Gate 'ADMISSION.json' 'PASS')){Worker 'test_contract';Worker 'admit'}
 if(-not(Gate 'ADMISSION.json' 'PASS')){throw 'Admission failed'}
 if($Stage -eq 'Admit'){Write-Output 'FROZEN_OUTER_ADMISSION_AND_FREEZE_COMPLETE';exit 0}
 if(-not(Gate 'PREDICTION_STATUS.json' 'ALL50_NEW_PLUS5_REUSED_OUTER_SCORES_SEALED')){Worker 'evaluate_outer'}
 if(-not(Gate 'PREDICTION_STATUS.json' 'ALL50_NEW_PLUS5_REUSED_OUTER_SCORES_SEALED')){throw 'Prediction seal missing'}
 if(-not(Test-Path -LiteralPath "$runtime\public\OBSERVED_TARGETS.json")){Worker 'report_outer'}
 if(-not(Gate 'OUTER_STATUS.json' 'COMPLETE_VERIFIED_ALL11_FULL80_FROZEN_OUTER_AUDIT')){Worker 'verify_outer'}
 if(-not(Gate 'OUTER_STATUS.json' 'COMPLETE_VERIFIED_ALL11_FULL80_FROZEN_OUTER_AUDIT')){throw 'Independent verification missing'}
 Write-Output 'ALL11_FROZEN_FULL80_OUTER_PIPELINE_VERIFIED_COMPLETE'
}finally{$guard.Dispose()}
