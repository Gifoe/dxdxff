# Compact A0 sole-view follow-up

Two new matched ablations replace the requested *comparison rows*, not the historical artifacts: DELTA Only and RATIO Only. F0 and ABS Only are frozen historical controls. No Raw-only rerun.

See [FINAL_REPORT.md](FINAL_REPORT.md), [outer metrics](results/OUTER_SUMMARY.csv), [development metrics](results/DEVELOPMENT_SUMMARY.csv), [paired intervals](results/OUTER_BOOTSTRAP.csv), and [independent verification](results/INDEPENDENT_VERIFICATION.json).

Execution order, with the same private dependencies as the preceding experiment:

```powershell
python code/run_study.py prepare
python code/run_study.py train
python code/run_study.py evaluate
python code/run_study.py verify
```

All commands must run on the trusted server with the pinned original source, legal FIT/VAL banks and frozen historical reference files. Private paths are runtime defaults, not public data dependencies to download. The protocol and source hashes are checked at every stage. A completed evaluation is non-rerunnable; unfinished inference can resume only from freeze-bound, hash-verified score files. Never change bound source or protocol to resume.

The sole-view masks keep Patient Z and LayerNorm, zero excluded values after original 88D FIT-only preprocessing, and keep historical duplicate auxiliaries. DELTA is a feature difference view, not the low-frequency delta band. The source-free model, trainer, original patient folds, class orientation and validation-only threshold rules are unchanged.

CSV primary statistics are patient-equal (80 outer patients or 65 validation appearances). Center/fold tables, secondary pooled-channel estimates, equal47-ID development estimates, all metrics, fixed-threshold bootstrap results, selections and resource measurements are included separately. Mean TP/FP/TN/FN counts are per patient, not pooled counts.

No installed skill was used. No raw/cache/weights/private predictions/identities/logs are published. This is exploratory repeated-cohort evidence; no new experiment follows seed42 automatically.

Result files retain exact server-export bytes (including Windows line endings), so the aggregate SHA256 values in the independent verification audit remain directly checkable after publication. Source and protocol files use LF.
