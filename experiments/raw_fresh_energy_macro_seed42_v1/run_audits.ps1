param([string]$RunTag='01')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\raw_fresh_energy_macro_seed42_v1'
$runtime='D:\raw_fresh_energy_macro_seed42_v1_runtime\final_stage_v1'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
$guard=[System.IO.File]::Open((Join-Path $runtime 'audit_coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Worker([string]$script,[string]$tag) {
    $log=Join-Path $runtime "${tag}_${RunTag}.log"
    $err=Join-Path $runtime "${tag}_${RunTag}.err"
    if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $err)) {throw 'Fresh RunTag required; preserve logs.'}
    $w=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u','-X','faulthandler',"$code\$script") -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
    $null=$w.Handle
    while (-not $w.HasExited) {Write-Output "POSTSEAL_AUDIT_ALIVE script=$script pid=$($w.Id) $(Get-Date -Format o)";Start-Sleep -Seconds 10;$w.Refresh()}
    $w.WaitForExit()
    if ($null -ne $w.ExitCode -and $w.ExitCode -ne 0) {throw 'Audit failed; inspect preserved logs.'}
    Get-Content -LiteralPath $log -Tail 2
}
try {
    if ((Get-Content -LiteralPath "$runtime\public\INDEPENDENT_FINAL_VERIFICATION.json" -Raw|ConvertFrom-Json).status -ne 'PASS') {throw 'Independent final verification missing.'}
    if (Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'raw_fresh_energy_macro_seed42_v1\\(audit_raw_support_v[0-9]+|verify_statistics|final_stage|verify_final)\.py'}) {throw 'Owned worker active; no duplicate.'}
    # The deployed v5 name retains earlier audit sources and failed logs; its bytes
    # are identical to the public audit_raw_support.py source.
    Worker 'audit_raw_support_v5.py' 'support_audit'
    Worker 'verify_statistics.py' 'statistics_audit'
    foreach ($name in @('RAW_SUPPORT_AUDIT.json','INDEPENDENT_STATISTICS_VERIFICATION.json')) {
        if ((Get-Content -LiteralPath "$runtime\public\$name" -Raw|ConvertFrom-Json).status -ne 'PASS') {throw 'Post-seal audit gate missing.'}
    }
    Write-Output 'FRESH_RAW_POSTSEAL_AUDITS_PASS'
} finally {$guard.Dispose()}
