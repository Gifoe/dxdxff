param([Parameter(Mandatory=$true)][int]$WorkerProcessId)
$ErrorActionPreference='Stop'
$row=Get-CimInstance Win32_Process -Filter "ProcessId=$WorkerProcessId"
if ($null -eq $row -or $row.CommandLine -notlike '*m8_physiology_feature_discovery_seed42_v1\train_old7_m8.py*') {throw 'Target is not the existing task worker'}
$worker=Get-Process -Id $WorkerProcessId
Write-Output "ATTACHED_READ_ONLY_EXISTING_WORKER PID=$WorkerProcessId"
while (-not $worker.HasExited) {
    Start-Sleep -Seconds 20
    $worker.Refresh()
    Write-Output "OBSERVED_WORKER_ALIVE PID=$WorkerProcessId LIVE=$(-not $worker.HasExited) UTC=$([DateTime]::UtcNow.ToString('o'))"
}
Write-Output "OBSERVED_WORKER_TERMINAL PID=$WorkerProcessId"
# Observation only: never starts/stops/restarts the scientific process.
