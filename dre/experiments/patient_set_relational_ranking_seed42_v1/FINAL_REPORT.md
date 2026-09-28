# Patient-set Relational Ranking Zero-shot Study

Exact A1/B0 fixed-query AP: 0.576743463; 150 source checkpoints, 65 cells, 47 IDs, 20 fixed repetitions.
All readout selection used FIT patients. All new target/control scores were hash-frozen before target-label metrics.
Inference is transductive B=0: full context uses all patient channels' unlabeled R4/A1 margins, with no target labels or updates.
Strict target-label sequencing is false: legacy VLOO source selection used cross-patient labels, and its loader materializes target labels. No target labels were indexed for the new readout selection or score freeze.

## Frozen R4 results

| Variant | Context | AP | Delta AP vs A1 | Patient-ID CI | MRR | Top1 |
|---|---|---:|---:|---|---:|---:|
| R0_A1 | full | 0.576743 | +0.000000 | [+0.000000, +0.000000] | 0.740038 | 0.654771 |
| R1_DEEPSET_RESIDUAL__BCE_ONLY | full | 0.569814 | -0.006930 | [-0.013151, -0.001515] | 0.742557 | 0.665632 |
| R1_DEEPSET_RESIDUAL__BCE_ONLY | query_only | 0.569367 | -0.007377 | [-0.014122, -0.001678] | 0.741748 | 0.664856 |
| R1_DEEPSET_RESIDUAL__RANK_SELECTED | full | 0.571490 | -0.005253 | [-0.009538, -0.000984] | 0.743176 | 0.667184 |
| R1_DEEPSET_RESIDUAL__RANK_SELECTED | query_only | 0.571219 | -0.005525 | [-0.009924, -0.001139] | 0.742496 | 0.666408 |
| R2_PAIRWISE_RELRANK__BCE_ONLY | full | 0.569008 | -0.007736 | [-0.016198, +0.000391] | 0.742061 | 0.664856 |
| R2_PAIRWISE_RELRANK__BCE_ONLY | query_only | 0.568816 | -0.007927 | [-0.016478, +0.000336] | 0.743953 | 0.667184 |
| R2_PAIRWISE_RELRANK__RANK_SELECTED | full | 0.570962 | -0.005781 | [-0.013848, +0.001937] | 0.741639 | 0.662529 |
| R2_PAIRWISE_RELRANK__RANK_SELECTED | query_only | 0.570362 | -0.006382 | [-0.014858, +0.001540] | 0.741573 | 0.662529 |

## Context controls

### FULL_VS_QUERY_CONTEXT
- R1_DEEPSET_RESIDUAL__BCE_ONLY: delta AP +0.000447, 95% CI [+0.000002, +0.001127].
- R1_DEEPSET_RESIDUAL__RANK_SELECTED: delta AP +0.000272, 95% CI [+0.000068, +0.000526].
- R2_PAIRWISE_RELRANK__BCE_ONLY: delta AP +0.000192, 95% CI [-0.000199, +0.000589].
- R2_PAIRWISE_RELRANK__RANK_SELECTED: delta AP +0.000600, 95% CI [+0.000037, +0.001309].

### CORRECT_VS_WRONG_CONTEXT
- R1_DEEPSET_RESIDUAL__BCE_ONLY: delta AP +0.004310, 95% CI [-0.000901, +0.012602].
- R1_DEEPSET_RESIDUAL__RANK_SELECTED: delta AP +0.001029, 95% CI [-0.001066, +0.003734].
- R2_PAIRWISE_RELRANK__BCE_ONLY: delta AP +0.000792, 95% CI [-0.001840, +0.003671].
- R2_PAIRWISE_RELRANK__RANK_SELECTED: delta AP +0.000960, 95% CI [-0.001991, +0.004120].

### CORRECT_CONTEXT_VS_SHUFFLED_RELATION
- R1_DEEPSET_RESIDUAL__BCE_ONLY: delta AP +0.000000, 95% CI [+0.000000, +0.000000].
- R1_DEEPSET_RESIDUAL__RANK_SELECTED: delta AP +0.000000, 95% CI [+0.000000, +0.000000].
- R2_PAIRWISE_RELRANK__BCE_ONLY: delta AP +0.000070, 95% CI [-0.000042, +0.000276].
- R2_PAIRWISE_RELRANK__RANK_SELECTED: delta AP +0.000038, 95% CI [-0.000016, +0.000108].

## Decision

- Terminal: `R4_RELATIONAL_ZEROSHOT_NOT_SUPPORTED`.
- FIT-frozen primary: `R1_DEEPSET_RESIDUAL__RANK_SELECTED`; Stage 2 allowed: `False`.
- R3: NOT_RUN_FIT_GATE. R2 minus R1 FIT AP = -0.003166 on average, positive in 1/5 folds.
- Current B8 AP 0.599632; best B8 AP 0.605291. No B8 target support was used.

## Direct answers to the study questions

1. A1/B0 replay: PASS; exact fixed-query AP 0.576743463 and the prior 150-checkpoint audit passed within 1e-6.
2. R1 DeepSets: no AP gain. FIT-frozen ranking arm AP 0.571490, delta -0.005253.
3. R2 pairwise: no AP gain. Best R2 arm AP 0.570962.
4. R3 weighting: not tested because its prespecified FIT-only gate failed.
5. Full versus query-only context for the primary arm: +0.000272 AP; too small to rescue A1.
6. Correct versus wrong-patient context: +0.001029, CI [-0.001066, +0.003734]; not supported.
7. Correct versus shuffled relation: +0.000000; R1's pairing shuffle is an exact identity by design, and R2's difference is negligible.
8. Primary rank changes: Kendall tau 0.934; 0.415 top-5 positions changed per cell; 10 true-EZ promotions versus 12 demotions across 65 cells.
9. Frozen-A1 failure strata (primary delta AP): poor -0.006064, medium -0.001952, strong -0.007251; improvement is not concentrated in failures.
10–12. No relational arm reaches 0.590, current B8 0.599632, or best B8 0.605291. Best relational AP is 0.571490 (R1_DEEPSET_RESIDUAL__RANK_SELECTED).
13–14. Stage 2 is not eligible and was not executed; no end-to-end result is claimed.
15. The prespecified negative terminal stops this R4-only zero-shot patient-set route. Richer observables, not further tuning on these target outcomes, would be a separate study.
