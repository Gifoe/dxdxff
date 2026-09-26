# Part A — five-fold validation gate

The exact historical B0 checkpoints and 88-feature table replayed (80 patients, 7,635 channels; maximum channel-score error `1.48e-7`). Five patient-level cross-fit cells were completed inside each outer-fit partition: 25/25 cells, 255 outer-fit patient-fold prediction episodes and 24,267 out-of-fold channel predictions. Every meta-training patient was excluded from its predictor's training and nested validation sets. The feature imputer/scaler were refit only on the appropriate inner-training patients. See `B0_REPLAY_AUDIT.json` and `CROSS_FIT_AUDIT.json`.

The OOF logit standard deviation exceeded the prior in-sample B0 standard deviation in every fold (five-fold average difference `+0.0924` logit units). This establishes a score-distribution mismatch, although the cross-fit models were also trained on fewer patients; the comparison does not isolate an in-sample effect causally.

| Outer fold | B0 validation Macro-F1 | A1 direct-threshold Macro-F1 | A1−B0 (pp) | A1 fit-OOF Macro-F1, optimistic |
|---:|---:|---:|---:|---:|
| 1 | 0.64119 | 0.59418 | −4.70 | 0.64471 |
| 2 | 0.64198 | 0.60629 | −3.57 | 0.65154 |
| 3 | 0.62728 | 0.56231 | −6.50 | 0.59915 |
| 4 | 0.63990 | 0.62139 | −1.85 | 0.64205 |
| 5 | 0.59155 | 0.57517 | −1.64 | 0.62716 |
| **Mean of folds** | **0.62838** | **0.59187** | **−3.65** | **0.63292** |

Patient-equal validation EZ-F1 fell by `4.87 pp` on average. The label-blind A1 predicted-threshold distance to the retrospective validation-optimal interval averaged `0.710` logit units, worse than the frozen B0 anchor's `0.575`. Mean within-fold correlation between predicted threshold and the diagnostic optimal-interval midpoint was `−0.141`. These oracle calculations used validation labels only for diagnosis, not model fitting or adjustment.

The fit-OOF target oracle averaged `0.69950` Macro-F1; the validation threshold oracle averaged `0.70284`. Thus the large oracle ceiling persists, but the fixed linear descriptor did not recover any of it on validation. The fit targets had **zero** patients with disconnected optimal intervals in all five folds, so the failure is not explained by accidentally collapsing separated optimum regions. These are diagnostics, not deployable results.

Locked gate: mean A1 validation gain ≥ `+0.01`, at least `3/5` positive folds, and mean EZ-F1 decline no worse than `−0.02`. Observed: `−0.03651`, `0/5`, and `−0.04875`. **Gate status: STOP.** No A1 outer-test arrays were read; no `OUTER_RESULTS.csv` or `OUTER_SUMMARY.md` is generated. The prior prevalence-head B1 had different validation threshold selection and was evaluated on already viewed outer outcomes, so its number is not a clean direct-threshold head-to-head comparator.
