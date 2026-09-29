# CRST-iEEG seed-42: stopped partial result

Run stopped at the user's request on 2026-09-30. This is **not** a completed two-benchmark evaluation. Only aggregate results are published; private checkpoints, per-patient/per-channel records, caches, and runtime logs remain on the server.

## Completed historical ictal comparison

The frozen historical 47-ID, 65-cell, 20-query-repeat VLOO comparison was completed with the exact A1 matched-query replay. These are development-side results, **not** a new outer test.

| Model | AP | AUROC | MRR | Top-1 | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 0.576743 | 0.746382 | 0.740038 | 0.654771 | 0.620810 |
| CRST-0 | 0.410613 | 0.619257 | 0.575096 | 0.449185 | 0.538492 |
| CRST-FULL | 0.421082 | 0.625983 | 0.555936 | 0.417378 | 0.532666 |

For CRST-FULL minus A1, the paired 10,000-draw patient-cluster bootstrap (seed 42) found AUROC Δ = -0.120399, 95% CI [-0.185515, -0.058519], and AP Δ = -0.155662, 95% CI [-0.205246, -0.108751]. The observed ictal evidence is against the claimed gain. Full precision and other metrics are in `ICTAL_PRIMARY_METRICS.csv` and `ICTAL_BOOTSTRAP.csv`.

## Omni-iEEG progress at stop

The train-side spectral cache passed its aggregate audit: 141 official train patients, 296 train EDFs, 13,350 labeled EDF-channel units, 1,449 spectral clips, and no cross-EDF patient-channel label conflicts. The official test split was not accessed.

CRST-0 finished inner training and validation. Its selected epoch was 6. On the 26 inner-validation patients, patient-equal AUROC was 0.830241 (13 estimable patients), AP 0.755156 (26 estimable), MRR 0.817321, and Top-1 0.769231. A train-side frozen threshold of 0.198242 gave inner-validation Macro-F1 0.673796. **These are validation metrics, not official test scores and not directly comparable to the previously published test AUROC.**

Omni CRST-FULL was interrupted during SSL after its first logged epoch; its saved resume checkpoint and logs were retained privately. It has no completed validation or test result. No official Omni test, ictal outer test, refit, or ablation was run. The intended two-benchmark comparison and any 0.85-AUROC target remain unverified.

## Stop and provenance

- The four matching Omni CRST Python processes were stopped and a fresh process query found zero matching processes.
- Existing source, frozen protocol/training locks, completed ictal result CSVs, Omni CRST-0 checkpoints, CRST-FULL SSL resume checkpoint, and logs were preserved.
- No scientific rule, checkpoint, prediction, or test outcome was changed to improve a result.
- Exact status: `USER_STOPPED_PARTIAL_RUN`; `FINAL_TEST_ACCESSED = NO`.
