# ReVA-DLM CPU-only Mini G0/G1

This workspace reproduces a feasibility audit on the public Prophet GSM8K trajectories. It does not run G2/G3 and does not contain LLaDA model weights.

## Frozen result

- G0: `G0_EXPLORATORY_PASS`, not a formal or strong pass. The primary 50% checkpoint has 12 DEV corruptions (`RCR=0.011374`). The already frozen, DEV-selected 60% exploratory checkpoint has 16 (`RCR=0.015166`), with adjacent support at 50%; this evidence is same-DEV, post-diagnostic, and selection-biased.
- G1: `G1_PASS`, not a strong pass. At the frozen 60% checkpoint, the best cheap-feature OOF model has AUROC 0.777847 (95% bootstrap CI 0.749622–0.804846) and AUPRC 0.657636. The critical same-current-wrong control remains difficult (best AUROC 0.604569), although the same-final-correct task is easy (best AUROC 0.864584).
- Cache: `BORDERLINE_REVIEW_REQUIRED`. No `cache/reva_gsm8k_mini_v2/` handoff was created because the prompt permits an ordinary `READY_FOR_G2` cache only for `G0_PASS`/`G0_STRONG_PASS` together with `G1_PASS`/`G1_STRONG_PASS`. The old v1 directory is quarantined as `REBUILD_REQUIRED` and must not be used for G2.

These results use the public GSM8K test set and are not a clean confirmatory benchmark result.

## Exact execution

PowerShell environment:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:NO_PROXY = "hf-mirror.com,.hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "120"
$env:CUDA_VISIBLE_DEVICES = ""
```

Create the isolated Python 3.10 environment:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Download only the frozen data and tokenizer allowlists, then run:

```powershell
.venv\Scripts\python.exe scripts\download_data_mirror.py --workers 8
.venv\Scripts\python.exe scripts\download_tokenizer_mirror.py
.venv\Scripts\python.exe scripts\run_mini_g0_g1.py
.venv\Scripts\python.exe -m pytest -q
```

The pipeline intentionally stops with `BORDERLINE_REVIEW_REQUIRED`. Running the strict verifier on the old v1 directory fails by design because its status is `REBUILD_REQUIRED`; therefore this run does not print `CACHE_VERIFY_OK`.

The data downloader is resumable and skips completed files. It pins dataset revision `91beb881acaa0b6edfccd88e8d19c08ec5e1225b`. The tokenizer downloader uses a closed five-file allowlist at revision `08b83a6feb34df1a6011b80c3c00c7563e963b07`; weight-like suffixes are rejected.

## Main artifacts

- `reports/REVA_MINI_G0_G1_REPORT.md`: complete scientific and provenance report.
- `reports/schema_audit.md`: actual `.pt` schema and 1,319/1,319 reconstruction validation.
- `reports/G0_REPORT.md`, `reports/G1_REPORT.md`: gate-specific evidence.
- `reports/final_terminal_summary.txt`: exact terminal-style stop decision and headline metrics.
- `reports/prompt_index.parquet`, `reports/state_semantics.json`: exact prompt/state provenance needed for a future independently validated cache.
- `figures/`: G0 RCR/RSR and transition-count plots.
- `cache/status.json`: authoritative root status (`BORDERLINE_REVIEW_REQUIRED`).
- `cache/reva_gsm8k_mini_v1/`: obsolete prior artifact retained only for audit; its own status is `REBUILD_REQUIRED`.
- `scripts/verify_cache.py`: strict verifier for a future ordinary `READY_FOR_G2` cache. It correctly rejects the obsolete v1 artifact.

## Next admissible step

Do not start G2 from this workspace state. First validate the frozen 60% G0 checkpoint on trajectories that were not used to select it. Only a qualifying independent result may authorize rebuilding and strictly verifying a versioned handoff cache.
