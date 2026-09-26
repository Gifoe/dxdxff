# Retrospective diagnostics, not deployable scores

All values are patient-equal over the 80 historical outer-test patients. D1/D2 use each patient's test labels and are **oracles**, not valid inference procedures. They were run before training B1–B3 solely to assess headroom. The predeclared continuation gate passed; no oracle-derived threshold, count or shift enters adapter training or inference.

| Analysis | Macro-F1 | Difference from B0 | EZ-F1 | BA |
|---|---:|---:|---:|---:|
| B0 | 0.616167 | — | 0.423866 | 0.664179 |
| D1 oracle per-patient constant logit shift | 0.702602 | +0.086435 | 0.544341 | 0.717443 |
| D2 oracle true-K top-K | 0.649248 | +0.033081 | 0.469383 | 0.649248 |

For D1, 79 patients improve, one is unchanged and none degrades under a label-optimized shift. The shift median is `-0.0302`; its 10th/90th percentiles are `-1.0348/+0.7471`. This indicates substantial *retrospective* calibration headroom, not that a label-blind head can recover it.

| Aligned historical score | EZ-AUPRC | EZ-AUROC | EZ-MRR | Top-1 EZ | True-K recall |
|---|---:|---:|---:|---:|---:|
| B0 | 0.531996 | 0.726844 | 0.733031 | 0.6500 | 0.469383 |
| PRQ | 0.521175 | 0.740943 | 0.688629 | 0.5625 | 0.464977 |
| BCR | 0.562311 | 0.735460 | 0.774794 | 0.6875 | 0.498163 |

BCR exceeds B0's patient-equal EZ-AUPRC by `0.030315` and MRR by `0.041763`; this is compatible with a ranking contribution but does not isolate it causally because the branches use different representations. Mean within-patient Spearman agreement is B0/BCR `0.7454`, B0/PRQ `0.6916`, PRQ/BCR `0.7852`; respective top-10%-channel disagreement rates are `0.5529`, `0.6473`, `0.5892`.

The exact compact diagnostic values and hashes are in `results/DIAGNOSTICS_SUMMARY.json`. Private patient/channel-level diagnostic tables remain only on the server.
