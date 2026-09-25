param(
    [string]$RepoRoot = "E:\DRE-nips\new-pipeline\7-11",
    [string]$Python = "python",
    [string]$Config = "E:\DRE-nips\new-pipeline\7-11\configs\task1_aaai_oof_server.json"
)
$ErrorActionPreference = "Stop"
$cfg = Get-Content -Raw -LiteralPath $Config | ConvertFrom-Json
$out = [string]$cfg.server_output_root
if (Test-Path -LiteralPath $out) { throw "Refusing to overwrite existing result directory: $out" }
Set-Location -LiteralPath $RepoRoot
& $Python .\scripts\task1_aaai\analyze_existing_oof.py `
  --seed-ledger "42=$($cfg.seed_ledgers.'42')" --seed-thresholds "42=$($cfg.seed_thresholds.'42')" `
  --seed-ledger "52=$($cfg.seed_ledgers.'52')" --seed-thresholds "52=$($cfg.seed_thresholds.'52')" `
  --seed-ledger "62=$($cfg.seed_ledgers.'62')" --seed-thresholds "62=$($cfg.seed_thresholds.'62')" `
  --output-dir $out --bootstrap-samples ([int]$cfg.bootstrap_samples) --bootstrap-seed ([int]$cfg.bootstrap_seed)
if ($LASTEXITCODE -ne 0) { throw "Task 1 OOF analysis failed with exit code $LASTEXITCODE" }
