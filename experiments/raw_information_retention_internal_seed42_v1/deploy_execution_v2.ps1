$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\raw_information_retention_internal_seed42_v1'
$archive='D:\raw_information_retention_internal_seed42_v1_runtime\failed_execution_source_v1'
$incoming='D:\raw_information_retention_internal_seed42_v1_runtime\incoming_execution_v2'
$owned=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'raw_information_retention_internal_seed42_v1\\' }
if ($owned) { throw 'Owned study worker active; deployment prohibited' }
foreach ($name in @('common.py','engine.py','run_checked.ps1')) {
    $target=Join-Path $code $name
    if ((Get-FileHash -LiteralPath $target).Hash -ne (Get-FileHash -LiteralPath (Join-Path $archive $name)).Hash) {
        throw "Source $name is not the preserved v1; inspect before deployment"
    }
    Copy-Item -LiteralPath (Join-Path $incoming $name) -Destination $target -Force
    if ((Get-FileHash -LiteralPath $target).Hash -ne (Get-FileHash -LiteralPath (Join-Path $incoming $name)).Hash) {
        throw "Source $name failed deployment hash parity"
    }
}
$amendment=Join-Path $code 'EXECUTION_AMENDMENT.json'
if (Test-Path -LiteralPath $amendment) { throw 'Amendment already exists; inspect before replacing' }
Copy-Item -LiteralPath (Join-Path $incoming 'EXECUTION_AMENDMENT.json') -Destination $amendment
Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $code 'common.py'),(Join-Path $code 'engine.py'),$amendment | Select-Object Hash,@{n='File';e={Split-Path -Leaf $_.Path}} | ConvertTo-Json
if (-not (Select-String -LiteralPath (Join-Path $code 'common.py') -Pattern 'cudnn.deterministic=True' -SimpleMatch)) {
    throw 'Deterministic source setting absent'
}
Write-Output 'EXECUTION_V2_DEPLOYMENT_HASH_VERIFIED'
