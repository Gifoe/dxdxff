# M8 physiology feature audit and discovery — seed42

Final results: **the registered primary simultaneous-improvement gate was not met**. This completed study does not claim that all old7 measurements are necessary or that finite search proves global optimality. Diagnostic new-feature routes improve ranking point estimates,not demonstrated Macro-F1 against the same-width control. Read `REPORT.md` before interpreting a single favorable row.

## Delivered evidence

- `initial_results`:225 original-feature and390 candidate linear screens,full-cohort label-blind extraction and independent checks.
- `phase2_results`:705 patient-isolated and520 source-held-out nonlinear screens,per-FIT frozen feature lists,numerical controls and synthetic model admission.
- `phase3_results`:16×5 old7 model pairs (160 actual fits),development-only summaries and integrated real-FIT admission. Development has65 appearances/47 unique patients,not the80-patient Outer cohort.
- `final_results/old7` and `final_results/integrated`:independent training replay,actual26D1+36M8 integration fits,45 logical cells with9 exact aliases.
- `final_results/outer`:all26 procedures over5 outer folds,80patients/7635channels each;patient-macro,pooled,fold,source,ranking,confusion,missing coverage and10000-patient-bootstrap tables. Includes primary adjusted intervals and compact freeze audit.
- `final_results/compute`:actual parameters,forward-hook affine MAC counts,conditional inference/offline-feature timing and measured/missing training work.
- `final_results/execution`:synthetic reporting-runtime parity,private-cache tamper tests and exact partial/final aggregate parity. No individual predictions,identities,models,scalers,raw data orlogs are included.
- `GOAL_COMPLETION_AUDIT.json`:final requirement-level evidence ledger. Earlier stage files keep their truthful stage-time `full_goal_complete=false`;they are not the final research-state declaration.

## Local verification, no private data required

From the repository root:

```powershell
python experiments/m8_physiology_feature_discovery_seed42_v1/tools/verify_public_initial.py
python experiments/m8_physiology_feature_discovery_seed42_v1/tools/verify_phase2_public.py
python experiments/m8_physiology_feature_discovery_seed42_v1/tools/verify_phase3_public.py
python experiments/m8_physiology_feature_discovery_seed42_v1/tools/verify_final_public.py
python experiments/m8_physiology_feature_discovery_seed42_v1/tools/verify_completion.py
```

The first verifier's historical terminal says the full goal was then active;it verifies only that earlier stage. The completion verifier is authoritative for the whole delivered package. Scores are actual runs,not synthetic performance estimates.

## Trusted-server reproduction and no-repeat rule

Original private source/cache/checkpoint bundles are required;this is not a standalone patient-data release. Scientific hashes and prospective locks pin all formulas,splits,windows,initializers,losses,optimizers,budgets,selection and thresholds. New features never use their own Outer fold labels for screening. No installed research/extraction skills or subagents were used.

Production training sources are `train_old7_m8.py` and **`train_integrated_v2.py` + `integrated_data_v2.py`**. Original optimizers/max30/min6/patience6 and initialization remain unchanged. Do not run the initial integrated v1 entrypoint:its pre-training axis failure is preserved and documented. Do not count discarded admission steps or aliases as extra fits.

The one frozen final predictor is `evaluate_final_v2.py`;all130 outputs are already sealed. **Do not launch it again as a new scientific test or alter its frozen source.** Reporting recovery uses `report_final_cpu_v2.py`,which extracts the unchanged frozen report AST,not a newly fitted classifier. The final adopted reporting process uses the existing Python3.10 CPU environment,one process-only logical-CPU16 affinity,OMP/MKL/OPENBLAS threads1 and no Torch import. `tools/run_cpu_report_core16.ps1` and `tools/verify_final_core16.ps1` record the actually executed launch pattern;their fixed log names intentionally prevent accidental overwrites. Fresh reporting replay needs a new log tag,never removal of old logs.

`verify_final_cache_stage.py` reconstructs inputs/caches/roles in the original model runtime without predictor forwards. `verify_final_cpu_metrics.py` runs the unchanged independent statistic-verifier suffix in the parity-gated CPU runtime. Intermediate failed reporters and the unused safe-ranking experiment are provenance only. See `ENGINEERING_PATCHES.md` for the native failures and recovery;no driver/BIOS,neural math,data,threshold or source was changed to rescue scores.

## Interpretation

Seed42 only;historically viewed Outer is **EXPLORATORY_REPEATED_TEST**,not blind confirmation. Bootstrap is conditional on these fitted models and thresholds. The6 primary intervals are multiplicity-adjusted;the many diagnostic comparisons are not universally corrected. `center` means the original4 source labels,not necessarily4 physical hospitals. The score-tied canonical ranking rule,positive-class convention,unavailable-metric handling and patient-macro versus pooled distinction are explicit in `REPORT.md`.

No models/features were reselected after this Outer evaluation. No external push is claimed:this delivery is committed locally unless a separate push is explicitly performed.
