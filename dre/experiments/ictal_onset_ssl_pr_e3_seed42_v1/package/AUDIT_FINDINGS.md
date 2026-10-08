# Data audit and experimental status, 2026-10-08

## Evidence directly reviewed from `Gifoe/dxdxff`

- [RawTiny IMPLEMENTATION_AUDIT](https://github.com/Gifoe/dxdxff/blob/codex/rawtiny-patient-relative-seed42-v1/dre/experiments/rawtiny_patient_relative_seed42_v1/IMPLEMENTATION_AUDIT.md): exactly 80 A1 patients, 256 matched run/seizure records, 24,995 run-channel incidences, 7,635 unique channels, 1,471,965 aligned window pairs, with alignment validated to raw waveforms.
- [RawTiny RAW_PREPROCESSING_AUDIT](https://github.com/Gifoe/dxdxff/blob/codex/rawtiny-patient-relative-seed42-v1/dre/experiments/rawtiny_patient_relative_seed42_v1/RAW_PREPROCESSING_AUDIT.json): raw cache at 250 Hz and 60 s; source constructor places onset at midpoint; independent EDF-to-cache onset confirmation absent for all 256 records.
- [RawTiny final report](https://github.com/Gifoe/dxdxff/blob/codex/rawtiny-patient-relative-seed42-v1/dre/experiments/rawtiny_patient_relative_seed42_v1/FINAL_REPORT.md): supervised RawTiny without PR EZ-AP .336249, with PR .429936, hybrid .557161, vs exact A1 .576743 on fixed-query development. Raw+PR+Attention was **already tried**; it does not establish added value.
- [Ranking bottleneck source audit](https://github.com/Gifoe/dxdxff/blob/codex/ranking-bottleneck-source-audit-seed42-v1/dre/experiments/ranking_bottleneck_source_audit_seed42_v1/FINAL_REPORT.md): patient/window attention changes not significant under predeclared gates.
- [Original B0 provenance](https://github.com/Gifoe/dxdxff/blob/codex/omni-a1feature-complementarity-audit-seed42-v1/dre/experiments/calibrank_mlp_seed42_v1/BASELINE_AUDIT.md): B0 on same 80 patients/5 outer folds; historical mean patient Macro-F1 0.616167 and EZ-AUPRC 0.531996.

## Checks performed *now* in the local runtime

- Private `raw_window_cache.pkl` and `all_window_cache.pkl` are not mounted. Only the user's summary Markdown is mounted. GitHub code/logs provide past audit evidence, not the raw samples needed for 80-patient revalidation.
- Implemented source-aware strict alignment exporter with duplicate-key, channel-map, patient-label, fold-mismatch, hash, 250Hz/60s, validity and padding checks.
- Implemented four-arm E0-E3 shared-architecture training and SSL; conditional E4 relative attention and E5_SHAM onset-pair ablations; patient-wise metrics, fold/center outputs and patient bootstrap.
- Local synthetic/fixture test suite passes; no real cohort model training has been performed.

## Open items preventing full clinical audit

1. Execute `audit_export.py` on trusted, private 80-patient server caches and compare recorded hashes.
2. Inspect raw EDF and onset sidecar for true seizure onset for all 256 matched recordings (or report audited coverage and nonverified cases). Merely verifying the old midpoint construction does not confirm true onset.
3. Confirm the actual frozen FIT/VAL/TEST patient memberships from the original runtime and prevent any regenerated random fold assignment.
4. Run E0-E3 on the approved 80-patient dataset, select checkpoints/thresholds solely on validation data, and run gated E4/E5_SHAM only if E3-vs-E1 validation gate passes.
5. Lock results and do a final external/patient-held-out confirmation on untouched data before a 0.70 publication claim.

**Current terminal status:** `IMPLEMENTATION_TESTED; HISTORICAL_ALIGNMENT_AUDITED; REAL_CACHE_RECHECK_NOT_RUN; EDF_ONSET_PROVENANCE_UNVERIFIED; REAL_E0_E3_TRAINING_NOT_RUN`.
