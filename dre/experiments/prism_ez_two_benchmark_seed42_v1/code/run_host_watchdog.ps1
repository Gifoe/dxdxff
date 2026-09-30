param(
    [Parameter(Mandatory = $true)][string]$Runtime,
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [Parameter(Mandatory = $true)][string]$Supervisor,
    [Parameter(Mandatory = $true)][string]$Code,
    [Parameter(Mandatory = $true)][string]$Protocol,
    [Parameter(Mandatory = $true)][string]$FeatureCache,
    [Parameter(Mandatory = $true)][string]$IctalCache,
    [Parameter(Mandatory = $true)][string]$IctalManifest,
    [int]$HostRetries = 96,
    [int]$MaxConsecutiveNoProgress = 3,
    [double]$RetryDelaySeconds = 8.0
)

$ErrorActionPreference = 'Stop'

function Write-AtomicJson([string]$Path, [hashtable]$Value) {
    $parent = Split-Path -Parent $Path
    [void](New-Item -ItemType Directory -Force -Path $parent)
    $temporary = "$Path.tmp"
    $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Get-ResumeFingerprint([string]$Root) {
    $files = @(Get-ChildItem -LiteralPath (Join-Path $Root 'ictal') -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @('in_epoch.pt', 'last.pt') } | Sort-Object FullName)
    $parts = foreach ($file in $files) {
        $relative = $file.FullName.Substring($Root.Length)
        "${relative}:$((Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash)"
    }
    $bytes = [Text.Encoding]::UTF8.GetBytes(($parts -join "`n"))
    return ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)))
}

function Read-InnerStatus([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json)
}

if ($HostRetries -lt 1 -or $MaxConsecutiveNoProgress -lt 1) {
    throw 'host retry limits must be positive'
}

$statusPath = Join-Path $Runtime 'ictal_ps_host_watchdog_status.json'
$innerStatusPath = Join-Path $Runtime 'ictal_native_supervisor_status.json'
$logDirectory = Join-Path $Runtime 'ps_host_watchdog_attempts'
[void](New-Item -ItemType Directory -Force -Path $logDirectory)
$noProgress = 0

for ($hostAttempt = 1; $hostAttempt -le $HostRetries; $hostAttempt++) {
    $before = Get-ResumeFingerprint $Runtime
    Write-AtomicJson $statusPath @{
        status = 'RUNNING'; host_attempt = $hostAttempt; started_utc = [DateTime]::UtcNow.ToString('o'); test_accessed = $false
    }
    $stdout = Join-Path $logDirectory ("host_{0:D3}.supervisor.log" -f $hostAttempt)
    $stderr = Join-Path $logDirectory ("host_{0:D3}.supervisor.err" -f $hostAttempt)
    $development = @(
        $PythonExe, (Join-Path $Code 'run_development.py'), '--benchmark', 'ictal', '--code', $Code,
        '--protocol', $Protocol, '--runtime', $Runtime, '--feature-cache', $FeatureCache,
        '--ictal-cache', $IctalCache, '--ictal-manifest', $IctalManifest
    )
    $supervisorArgs = @(
        $Supervisor, '--runtime', $Runtime, '--max-native-retries', '96', '--retry-delay-seconds', "$RetryDelaySeconds",
        '--attempt-prefix', ("pshost{0:D3}" -f $hostAttempt), '--'
    ) + $development
    $process = Start-Process -FilePath $PythonExe -ArgumentList $supervisorArgs -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru -Wait
    $inner = Read-InnerStatus $innerStatusPath
    if ($process.ExitCode -eq 0 -and $inner -and $inner.status -eq 'COMPLETE') {
        Write-AtomicJson $statusPath @{
            status = 'COMPLETE'; host_attempt = $hostAttempt; completed_utc = [DateTime]::UtcNow.ToString('o'); test_accessed = $false
        }
        exit 0
    }
    if ($inner -and $inner.status -eq 'STOPPED_NON_NATIVE_FAILURE') {
        Write-AtomicJson $statusPath @{
            status = 'STOPPED_NON_NATIVE_FAILURE'; host_attempt = $hostAttempt; returncode = $process.ExitCode; test_accessed = $false
        }
        throw 'inner supervisor recorded a non-native failure'
    }
    $after = Get-ResumeFingerprint $Runtime
    if ($after -eq $before) { $noProgress++ } else { $noProgress = 0 }
    $nextState = @{
        status = 'RETRYING_HOST_INTERRUPTION'; host_attempt = $hostAttempt; returncode = $process.ExitCode;
        progress_observed = ($after -ne $before); consecutive_no_progress = $noProgress;
        updated_utc = [DateTime]::UtcNow.ToString('o'); test_accessed = $false
    }
    if ($noProgress -ge $MaxConsecutiveNoProgress) {
        $nextState.status = 'STOPPED_NO_PROGRESS'
        Write-AtomicJson $statusPath $nextState
        throw 'host recovery stopped after bounded consecutive no-progress interruptions'
    }
    Write-AtomicJson $statusPath $nextState
    if ($hostAttempt -eq $HostRetries) { throw 'host recovery retry budget exhausted' }
    Start-Sleep -Seconds $RetryDelaySeconds
}
