param([Parameter(Mandatory=$true)][string]$RunTag)
$ErrorActionPreference='Stop'
if ($RunTag -notmatch '^[A-Za-z0-9_]+$') {throw 'Invalid log tag'}
$taskCode='D:\m8_physiology_feature_discovery_seed42_v1'
$taskRuntime='D:\m8_physiology_feature_discovery_seed42_v1_runtime'
$active=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -like '*python*' -and $_.CommandLine -like '*m8_physiology_feature_discovery_seed42_v1\train_old7_m8.py*'})
if ($active.Count -gt 0) {throw 'Existing live training worker: refuse duplicate'}
$frozen=Get-Content -LiteralPath "$taskRuntime\old7_matched_M8\INPUT_FREEZE_PRIVATE.json" -Raw | ConvertFrom-Json
foreach ($item in $frozen.source.PSObject.Properties) {
    $actual=(Get-FileHash -LiteralPath (Join-Path $taskCode $item.Name) -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $item.Value) {throw "Frozen scientific source changed: $($item.Name)"}
}
$stdout=Join-Path $taskRuntime "old7_M8_$RunTag.log"
$stderr=Join-Path $taskRuntime "old7_M8_$RunTag.err"
if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) {throw 'Preserve logs: choose a new tag'}
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1'
$worker=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u',"$taskCode\train_old7_m8.py") -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Write-Output "OWNED_WORKER_STARTED PID=$($worker.Id) TAG=$RunTag"
$elapsed=[Diagnostics.Stopwatch]::StartNew()
while (-not $worker.HasExited) {
    Start-Sleep -Seconds 5
    $worker.Refresh()
    if ($elapsed.Elapsed.TotalSeconds -ge 30) {
        Write-Output "OWNED_WORKER_ALIVE PID=$($worker.Id) UTC=$([DateTime]::UtcNow.ToString('o'))"
        $elapsed.Restart()
    }
}
$worker.WaitForExit()
Write-Output "OWNED_WORKER_TERMINAL PID=$($worker.Id) EXIT=$($worker.ExitCode)"
exit $worker.ExitCode
