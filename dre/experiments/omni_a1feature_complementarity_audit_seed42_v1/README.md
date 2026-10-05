# Omni A1-feature complementarity audit (seed 42)

This is a completed, fixed-protocol audit on the 124 both-class patients and the exact patient-level five-fold manifest from `omni_patient5fold_a1plugin_seed42_v1`.

The question was deliberately narrow: whether the historical A1 physiological descriptors correct errors made by the frozen TimeConv morphology model strongly enough to justify training an embedding-level feature plugin.  It did **not** train TimeConv, alter labels or folds, add context/attention, or select a fusion weight on a held-out fold.

## Result

The frozen TimeConv OOF replay passed exactly (patient-equal AUROC 0.7621631).  The feature-only models were weaker (F9: 0.5976709; F36: 0.7199595), but F36 made partly different ranking errors: its patient-equal rescue rate on TimeConv-wrong pairs was 0.52615 and its destroy rate on TimeConv-right pairs was 0.23477.  Fixed, predeclared OOF fusion diagnostics for F36 were positive for every beta in `{0.10, 0.25, 0.50}`; the values are reported in `results/LEGAL_FUSION_DIAGNOSTIC.csv`.

This is insufficient to meet the predeclared meaningful gate: rescue is below 0.60 and no beta was selected through a legal nested train-only procedure.  The audit therefore stops without training the proposed feature residual plugin.  See `FINAL_REPORT.md` for the full decision and limitations.

## Public artifact policy

`results/` contains only compact aggregate metrics, checksums, and audit prose.  The following were generated privately but are intentionally excluded from Git: waveform/record caches, checkpoints, feature tensors, per-patient/per-channel predictions, and the private `PATIENT_PAIR_COMPLEMENTARITY.csv`.  This keeps the required pair-level audit available for local reproducibility without publishing patient-level data.

## Contents

- `code/run_audit.py` — fixed-protocol extraction, fold-safe feature-head training, and audit implementation.
- `results/` — compact public aggregate outputs listed in the task specification.
- `FINAL_REPORT.md` — corrected public scientific interpretation of the completed run.
