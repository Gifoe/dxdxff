# Frozen A1 absolute-context complementarity probe

All reported results use the original frozen A1 checkpoints and FIT/validation patients only. No outer-test loader was built, and no outer-test predictions or performance metrics were evaluated. No A1 parameter was trained.

| Variant | VLOO Macro-F1 | EZ-F1 | BA | EZ-AUPRC | EZ-AUROC | EZ-MRR | Top1-EZ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | 0.625996 | 0.414294 | 0.673415 | 0.557200 | 0.743069 | 0.742103 | 0.676923 |
| P1 | 0.618627 | 0.400675 | 0.667313 | 0.556181 | 0.740478 | 0.753412 | 0.692308 |
| P2 | 0.617182 | 0.404699 | 0.669265 | 0.552183 | 0.738413 | 0.753278 | 0.692308 |

| Fold | P1-P0 Macro-F1 | P2-P0 Macro-F1 | P2-P1 Macro-F1 |
| --- | ---: | ---: | ---: |
| 1 | -0.004087 | +0.024541 | +0.028628 |
| 2 | +0.011905 | -0.028954 | -0.040859 |
| 3 | -0.041645 | -0.017281 | +0.024364 |
| 4 | -0.004313 | -0.010382 | -0.006070 |
| 5 | +0.001293 | -0.011994 | -0.013287 |

Mean P1-P0: -0.007369; P2-P0: -0.008814; P2-P1: -0.001445.
Positive folds: P2>P0 1/5; P2>P1 2/5.

P2 residual diagnostics on VLOO-selected validation patients:
- P1 mean |delta| 0.025427; P2 mean |delta| 0.052973; P2 fraction |delta|>0.45 0.000000; saturation near ±0.5 0.000000.
- Changed channel-pair orderings 0.004900; corr(A1 logit, delta) -0.19443195023090049.

Absolute-context correlations are **LABEL_USING_DIAGNOSTIC_ONLY**. The 9 pooled and 45 fold-level Pearson/Spearman comparisons are in `development/ABSOLUTE_CONTEXT_INFORMATION_DIAGNOSTIC.csv`; they were not used for model selection and are exploratory without multiplicity correction. Pooled `||mu||₂` versus true EZ fraction has Spearman `-0.384`; the other pooled absolute Spearman coefficients are at most `0.181`. True EZ fraction and K*/channel count are identical here, so their correlation rows are duplicates. These associations do not establish predictive complementarity.

Complementarity checks:
- P2_minus_P0_macro_f1_ge_0_010: FAIL
- P2_minus_P1_macro_f1_ge_0_005: FAIL
- positive_P2_vs_P0_folds_ge_4: FAIL
- positive_P2_vs_P1_folds_ge_3: FAIL
- P2_mean_EZ_F1_nondecreasing: FAIL
- P2_EZ_AUPRC_delta_ge_minus_0_005: FAIL
- P2_VLOO_macro_f1_ge_0_640: FAIL

Test-readiness checks:
- source_P0_reproduced: PASS
- complementarity_gate_pass: FAIL
- P2_apparent_fullval_mean_ge_0_665: FAIL
- P2_apparent_fullval_worst_fold_ge_0_620: FAIL
- P2_fraction_abs_delta_gt_0_45_lt_0_10: PASS
- no_leakage_or_pathology: PASS

P2 APPARENT_FULLVAL mean/worst fold Macro-F1: 0.654032/0.598767.
**Terminal: `ABSOLUTE_CONTEXT_COMPLEMENTARITY_NOT_SUPPORTED`.**

The locked development gate failed. Stop this route; no outer test or post-hoc probe adjustment was run.
