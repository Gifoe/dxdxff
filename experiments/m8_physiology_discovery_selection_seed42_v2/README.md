# DREChannNet physiological discovery/selection, seed42 v2

Completed study: **5388 low-cost probe fits,85 new neural fits/1108 epochs,
105 sealed Outer score cells, independently verified results.** No replacement
for the old7 is adopted. There is a small exploratory selected-Concat ranking
increment, not a demonstrated broad M8/F1 improvement. Start with
[REPORT.md](REPORT.md) and [RESULT_TABLES.md](RESULT_TABLES.md).

Branch: `codex/m8-physiology-discovery-selection-seed42-v2`.
This is a follow-up study, not a rewrite of the completed prior feature,
factorial or single-head experiments. Old artifacts remain intact.

## Code and stages

| Stage | Entry | Actual deliverable |
|---|---|---|
| Source inventory | `inspect_sources.py` | Read-only operator inventory; never publish its private paths/metadata |
| Label-blind admission/pool | `build_pool.py` | Historical28 audit; exact32-group valid-window pool |
| Nested selection | `select_features.py` |5253 search+135 assessment fits; five own-FIT decisions |
| Independent selector replay | `verify_selection.py` | All search/assessment fits, roles, scalers, thresholds and recipes checked |
| Physiology | `biology.py` |160 primary feature-phase tests; overlapping FIT/source/seizure sensitivity |
| Matched neural fitting | `train.py` |80 actual frozen D1/M8/Concat/Raw fits |
| Label-blind redundancy | `redundancy.py` |496 pairs×2 coordinates×5 FIT sets; numerical quality |
| Prospective support control | `support_control.py` |Five Raw late-pool8–30 fits; no physiology |
| Shared progress repair | `restore_status.py` |Administrative completion field from immutable main freeze |
| Validation replay/Outer seal | `evaluate.py` |All85 validation arrays exact;105 score cells sealed before full labels |
| Statistics | `report.py` |Patient/fold/source/coverage/10000-draw tables, cardinality |
| Independent results | `verify_results.py` |All2785 patient metric rows and all primary aggregate/interval fields |
| Supplementary fixed-score diagnostics | `source_complementarity.py` |Source-paired intervals and post hoc Raw compensation |
| Whitelisted execution export | `export_audit.py` |Source hashes, counts and identical support-control initialization |
| Regression tests | `test_entry.py` |NumPy-first synthetic tests; actual status JSON |
| Local anonymous rendering | `figures.py`, `format_tables.py` |Six editable/vector/raster figures and companion tables |

Scientific locks: `PROTOCOL_LOCK.json`, `BIOLOGY_LOCK.json`,
`IMPLEMENTATION_LOCK.json`, `TIME_SUPPORT_AMENDMENT.json`,
`EVALUATION_LOCK.json`. The time-support amendment preceded new Outer access;
the source/compensation decomposition was added after primary reporting and is
explicitly a post hoc fixed-score diagnostic. No new fit or chosen model follows
Outer results.

## Trusted-server reproduction

The executable code uses the same trusted server artifacts as preceding project
studies. These data/checkpoints cannot be made public. A fresh clone alone does
not contain clinical signals, labels, private roles or trained models. It can
re-render every delivered figure/table from public aggregates, but full training
requires authorized access to the original private artifacts below. Failed
source/hash/role checks must not be bypassed or substituted with another cache.

GPU environment: original Python3.13.9 environment at `E:\Anaconda\python.exe`,
PyTorch2.11.0+cu128, FP32, TF32 disabled, one RTX5090. OMP/MKL/OPENBLAS threads1;
original core routines set Torch CPU threads2. CPU statistics: the existing
stable Python3.10.20 environment with NumPy2.2.6, Pandas2.3.3, SciPy1.15.3,
sklearn1.7.2, Torch2.8.0+cu128; no training/prediction in this CPU environment.
Full statistics versions are recorded in `results/STATISTICS_ENVIRONMENT.json`.
The separately measured GPU environment is in `results/GPU_ENVIRONMENT.json`.

Required project sources, all already versioned in this repository:

- `dre/experiments/raw_delta_encoder_seed42_v1/code/`: `io_utils.py`,
  `dependencies.py`, `raw_store.py`, original physiological extractor.
- Original A0 decoder/UAS core, compact DELTA factory and D2 feature manifest
  referenced by that dependency loader. Their original pinned checks remain.
- `dre/experiments/d1_small_module_exploration_seed42_v1/phase3/code/run.py`:
  original Raw morphology architecture/objective.
- `experiments/m8_physiology_feature_discovery_seed42_v1/features.py`:
  previous11 candidate formulas and summary arithmetic.
- `dre/experiments/d2_graphrole_seed42_v1/`: verified earlier Raw graph rebuild.
- `experiments/m8_raw_backbones_pr_seed42_v1/evaluate.py`: original metric engine.

Required private artifact *types*: exact Raw-dynamic reconstruction/registry and
waveform banks, original FIT/VAL banks and fold/feature manifest, original7
conditioning cache, previous11-candidate/numeric3 window caches, source-verified
Raw-rebuilt graph ledger/windows, and previously sealed single-head reference
scores. `common.py` contains the explicit server roots; no patient path or
credential needs to be added to the public repository.

On the trusted server, place this directory at
`D:\m8_physiology_discovery_selection_seed42_v2`. Results/runtime stay at
`D:\m8_physiology_discovery_selection_seed42_v2_runtime`. Use NumPy-first Python
entry to avoid the documented initialization problem:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
# In a fresh authorized runtime, run each stage once and inspect its gate.
& E:\Anaconda\python.exe -c "import numpy,runpy;runpy.run_path(r'D:\m8_physiology_discovery_selection_seed42_v2\build_pool.py',run_name='__main__')"
# Then select_features.py, independent verify_selection.py, biology.py and train.py.
# All scripts require their original pinned project dependencies on sys.path.
```

For actual complete re-execution, put the experiment directory first on
`sys.path`, as `run_checked.ps1` does. The example above assumes that directory
is the working directory. CPU-only stages should run in the recorded CPU
environment with process-local affinity core16. Synthetic and source gates
precede training. `run_pipeline.ps1` provides the complete guarded sequence on
the authorized server and returns without scientific work when final
verification already exists. It was added for reproduction after the recorded
staged execution. Do not use the superseded `run_remaining.ps1` as an unattended
launcher; its chronology is preserved for audit. Corrected per-stage wrapper:

```powershell
# Keep this authenticated foreground process alive; do not detach and disconnect.
& D:\m8_physiology_discovery_selection_seed42_v2\run_checked.ps1 -Stage support
# After support completes, restore_status.py restores the shared main progress field.
& D:\m8_physiology_discovery_selection_seed42_v2\run_checked.ps1 -Stage evaluate
& D:\m8_physiology_discovery_selection_seed42_v2\run_checked.ps1 -Stage report
& D:\m8_physiology_discovery_selection_seed42_v2\run_checked.ps1 -Stage verify
```

**For the completed runtime, do not restart scientific stages to obtain another
test.** Read the completion/freeze evidence. Only genuinely interrupted work
may use source/input/checkpoint-bound resume with preserved logs. Never
invalidate a freeze, delete a completed score cache or tune a threshold to make
the report match a preferred result. A new scientific experiment needs its own
prospectively fixed protocol and separately reported identity.

Local anonymous reproduction:

```powershell
python experiments/m8_physiology_discovery_selection_seed42_v2/format_tables.py
python experiments/m8_physiology_discovery_selection_seed42_v2/figures.py
python experiments/m8_physiology_discovery_selection_seed42_v2/verify_release.py
```

Local figure requirements: Python3.10+, NumPy, Pandas, Matplotlib, Pillow.
Figures have180 mm width, editable SVG/PDF text and600 dpi PNG/LZW-TIFF. Plotting
does not import any private server reader.

## Public/private boundary

Public: code, scientific locks, source hashes, anonymous aggregate counts,
search configurations, group-level eligibility/correlations/effects, metric
tables/intervals and figure source data. Private: raw/EDF/cache, identifiers,
clinical labels, individual effect/feature/prediction rows, score caches,
embeddings, model/optimizer states, scalers, logs and credentials. No automatic
recursive runtime copy is permitted. All public files pass an allowlisted
schema/extension release audit and are pushed to GitHub, not left at a local
commit.
