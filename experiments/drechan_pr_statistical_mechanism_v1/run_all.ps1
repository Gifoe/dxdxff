param(
    [Parameter(Mandatory=$true)][string]$Runtime,
    [Parameter(Mandatory=$true)][string]$Python
)
$ErrorActionPreference = 'Stop'
if (-not $env:PRM_SOURCE_COMMIT) { throw 'Set the pushed PRM_SOURCE_COMMIT first' }
foreach ($key in @('PRM_RECONSTRUCTION','PRM_LABELS','PRM_FEATURE_PR','PRM_UNIT_PR','PRM_INTERNAL_BANKS')) {
    if (-not [Environment]::GetEnvironmentVariable($key)) { throw "Missing trusted input alias $key" }
}
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = ''
$entry = Join-Path $PSScriptRoot 'run_mechanism.py'
foreach ($stage in @('admission','freeze','evaluate','report','verify','package')) {
    Write-Output "STAGE_BEGIN $stage"
    & $Python -u $entry $stage --runtime $Runtime
    if ($LASTEXITCODE -ne 0) { throw "Mechanism stage failed: $stage; preserve inputs, outputs and logs" }
    Write-Output "STAGE_END $stage"
}
# The frozen package is a TRUSTED intermediate: its80/79 means are complementary.
# Only the filtered archive below may leave the trusted runtime.
& $Python -u (Join-Path $PSScriptRoot 'sanitize_public_export.py') --input (Join-Path $Runtime 'public_aggregate_export_v1.zip') --output (Join-Path $Runtime 'public_aggregate_export_privacy_v1.zip')
if ($LASTEXITCODE -ne 0) { throw 'Privacy export failed; do not transfer the intermediate archive' }
