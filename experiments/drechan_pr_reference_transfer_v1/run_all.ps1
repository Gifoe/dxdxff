param([Parameter(Mandatory=$true)][string]$Runtime,[Parameter(Mandatory=$true)][string]$Python)
$ErrorActionPreference='Stop'
if(-not $env:PRREF_SOURCE_COMMIT){throw 'Set pushed scientific source commit'}
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES=''
& $Python -m unittest discover -s (Join-Path $PSScriptRoot 'tests') -v
if($LASTEXITCODE -ne 0){throw 'Unit tests failed before admission'}
foreach($stage in @('admit','freeze','evaluate','verify','report','package')){
  Write-Output "STAGE_BEGIN $stage"
  & $Python -u (Join-Path $PSScriptRoot 'pipeline.py') $stage --runtime $Runtime
  if($LASTEXITCODE -ne 0){throw "Stage failed: $stage; preserve attempt and diagnose"}
  Write-Output "STAGE_END $stage"
}
