# PaReSet-EZ v1: frozen seed-42 outer-fold result

The five disjoint outer folds cover all 80 patients once. This is the **seed-42 outer test**, not a three-seed confirmation or an additional sealed holdout. Protocol lock SHA256: `734ccb3b185e1c6b660df71fec7e0e510fdfb470cae81fd9cc9da91198d13dc5`. All checkpoints and decision thresholds were selected by their existing inner-validation partitions before outer access. Validation replay passed 20/20 cells with zero threshold-prediction mismatches. `OUTER_TEST_USED_FOR_TUNING = NO`.

Patient-equal means:

| Method | Macro-F1 | EZ-F1 | Balanced accuracy | EZ-AUPRC |
| --- | ---: | ---: | ---: | ---: |
| PaReSet full | 0.5935 | 0.3733 | 0.6323 | 0.5003 |
| Retained base | 0.5949 | 0.3850 | 0.6351 | 0.4921 |
| Repaired PRQ | 0.5923 | 0.3779 | 0.6376 | 0.4844 |
| Repaired BCR | **0.6085** | **0.4119** | **0.6563** | 0.4837 |
| Repaired CDEL 0.8/0.2 | 0.6029 | 0.3835 | 0.6492 | **0.5121** |

Patient-paired delta (PaReSet full minus comparator), 10,000 patient bootstrap resamples, 95% percentile interval:

| Comparator | Metric | Delta, percentage points | 95% CI, percentage points |
| --- | --- | ---: | ---: |
| Retained base | Macro-F1 | -0.14 | [-1.21, +0.90] |
| Retained base | EZ-F1 | -1.16 | [-2.86, +0.51] |
| Retained base | Balanced accuracy | -0.28 | [-1.30, +0.75] |
| Retained base | EZ-AUPRC | +0.82 | [-0.14, +1.88] |
| Repaired BCR | Macro-F1 | -1.50 | [-3.61, +0.62] |
| Repaired BCR | EZ-F1 | -3.86 | [-7.33, -0.33] |
| Repaired BCR | Balanced accuracy | -2.41 | [-4.84, -0.10] |
| Repaired CDEL | Macro-F1 | -0.94 | [-3.61, +1.76] |

On outer Macro-F1, full is below retained base, BCR and CDEL. Its apparent five-fold inner-validation gain over base (+1.07 pp) did not transfer: full exceeded base on only 3/5 outer folds. The primary PaReSet-EZ full-model improvement is **not supported by this seed-42 outer test**. In particular, it would be invalid to choose the validation-winning BCE-only/no-reference ablation after inspecting this outer result and call a subsequent outer comparison confirmatory.

Limitations: one seed only; no extra final sealed cohort was accessed. These controls were newly trained on the same adapted 36-D input with a finite-gradient variance repair. They are **not** literal historical 28-D Table II checkpoints. The source cache and canonical label discrepancy are documented in `AUDIT.md` and `ADAPTER_DIFF.md`. Private per-channel and per-patient ledgers remain server-side; only aggregate numbers appear here.
