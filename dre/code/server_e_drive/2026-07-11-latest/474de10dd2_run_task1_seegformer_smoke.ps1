param(
    [Parameter(Mandatory = $true)][string]$FeatureCache,
    [Parameter(Mandatory = $true)][string]$RawCache,
    [Parameter(Mandatory = $true)][string]$Ledger,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

foreach ($path in @($FeatureCache, $RawCache, $Ledger)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path does not exist: $path" }
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

& $PythonExe scripts\task1_baselines\run_task1_baseline_suite.py `
  --feature-cache $FeatureCache `
  --raw-cache $RawCache `
  --v3-ledger $Ledger `
  --output-dir $OutputDir `
  --models seegformer `
  --expected-subjects 80 `
  --cohort-name task1_reference_80 `
  --seeds 42 `
  --epochs 1 `
  --max-outer-folds 1 `
  --seegformer-batch-size 4 `
  --seegformer-grad-accum-steps 1 `
  --device cuda `
  --amp `
  --compact `
  --strict

if ($LASTEXITCODE -ne 0) { throw "SEEGformer smoke failed with exit code $LASTEXITCODE" }
