param([Parameter(Mandatory=$true)][string]$Runtime,[Parameter(Mandatory=$true)][string]$Python)
$ErrorActionPreference='Stop'
foreach($name in @('PRCOMP_SOURCE_COMMIT','PRCOMP_PARENT_RUNTIME','PRM_RECONSTRUCTION','PRM_LABELS','PRM_FEATURE_PR','PRM_UNIT_PR','PRM_INTERNAL_BANKS')){
 if(-not [Environment]::GetEnvironmentVariable($name)){throw "Missing trusted input alias $name"}
}
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES=''
& $Python -m unittest discover -s (Join-Path $PSScriptRoot 'tests') -v
if($LASTEXITCODE -ne 0){throw 'Synthetic tests failed before admission'}
foreach($stage in @('admit','freeze','evaluate','verify','report','package')){
 Write-Output "STAGE_BEGIN $stage"
 & $Python -u (Join-Path $PSScriptRoot 'pipeline.py') $stage --runtime $Runtime
 if($LASTEXITCODE -ne 0){throw "Stage failed: $stage;preserve attempt before diagnosis"}
 Write-Output "STAGE_END $stage"
}
