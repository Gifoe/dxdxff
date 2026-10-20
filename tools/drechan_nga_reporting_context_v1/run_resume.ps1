param([string]$RunTag='reporting01')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\drechan_nga_reporting_context_v1';$runtime='D:\drechan_no_guidance_ablation_42_44_62_v1_runtime'
$python='D:\drechan_temporal_pr_contrast_3seed_v1_env\Scripts\python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
$guard=[IO.File]::Open((Join-Path $runtime 'coordinator.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
function Worker([string]$stage){
 $log=Join-Path $runtime ($stage+'_'+$RunTag+'.log');$err=Join-Path $runtime ($stage+'_'+$RunTag+'.err')
 if((Test-Path -LiteralPath $log)-or(Test-Path -LiteralPath $err)){throw 'New RunTag required; preserve logs'}
 $w=Start-Process -FilePath $python -ArgumentList @('-u','-X','faulthandler',"$code\entry.py",$stage) -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
 $null=$w.Handle
 while(-not $w.HasExited){Write-Output "NGA_REPORTING_ALIVE stage=$stage pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 15;$w.Refresh()}
 $w.WaitForExit();if($null -eq $w.ExitCode -or $w.ExitCode -ne 0){throw ('Reporting stage failed '+$stage+'; diagnose before resume')}
}
try{
 if(Get-CimInstance Win32_Process|Where-Object {$_.Name -eq 'python.exe' -and ($_.CommandLine -match 'drechan_no_guidance_ablation_42_44_62_v1\\affinity_entry' -or $_.CommandLine -match 'drechan_nga_reporting_context_v1\\entry')}){throw 'Existing owned worker active'}
 Worker 'admit'
 Worker 'report_outer'
 Worker 'verify_outer'
 Worker 'package_public'
 Write-Output 'NGA_FINAL_REPORT_VERIFIED_AND_PACKAGED_NO_RETRAIN_OR_REPREDICTION'
}finally{$guard.Dispose()}
