$ErrorActionPreference = 'Stop'
python .\run_outcome_hifos.py audit `
  --config .\configs\outcome_hifos_local_windows.yaml
python .\run_outcome_hifos.py run `
  --config .\configs\outcome_hifos_local_windows.yaml `
  --protocol screening `
  --variants H2_HIER_POOL,H5_ANCHORED_CORE,H6_ANCHORED_UOT_DESC `
  --max-outer-folds 1 `
  --max-patients 12 `
  --seeds 42 `
  --epochs 2 `
  --device cpu
