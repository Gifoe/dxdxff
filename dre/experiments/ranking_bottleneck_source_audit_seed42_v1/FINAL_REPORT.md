# Ranking bottleneck source audit (seed 42)

Development-only FIT/validation study on the frozen 80-patient cohort. No outer-test loader, prediction, or performance outcome was accessed for this experiment.
A1 replay: 150/150 checkpoints exact, maximum validation-grid error 0.0; S-F1 VLOO Macro-F1 0.6259962097.

| Independent source | S-RANK ΔEZ-AUPRC | ΔMRR | Positive AUPRC folds | S-F1 ΔMacro-F1 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| patient_attention | +0.004014 | -0.002580 | 3/5 | +0.001372 | FAIL |
| window_attention | +0.003613 | -0.012628 | 3/5 | +0.002148 | FAIL |
| both_attention | +0.002741 | +0.002432 | 3/5 | +0.004773 | FAIL |
| checkpoint_selection | -0.012803 | -0.001262 | 0/5 | +0.001027 | FAIL |
| ranking_objective | +0.000188 | -0.016443 | 3/5 | +0.001073 | FAIL |
| center_optimization | -0.009257 | -0.007461 | 1/5 | -0.010616 | FAIL |
| semantic_view_separation | -0.014438 | -0.045368 | 2/5 | +0.004358 | FAIL |

## Direct answers

1. Checkpoint-selection mismatch: CHECKPOINT_SELECTION_MISMATCH_NOT_SUPPORTED. S-RANK changes AUPRC by -0.012803; rank-first selection does not rescue A1.
2–4. Removing patient/window/both attention changes S-RANK AUPRC by +0.004014, +0.003613, +0.002741; each has only 3/5 positive folds and misses +0.010. No attention layer is established as harmful.
5. CTX0 at 25% channel dropout has mean retained-channel percentile-rank MAE 0.021837 and score Spearman 0.986370. Removing attention stabilizes ranks, but this does not establish a ranking-performance bottleneck.
6. CTX0 high-channel Q4 rank MAE is 0.016260, lower than the overall value; this audit does not show instability increasing with channel count. Frozen T0−T1 AUPRC is -0.016387; its n-channel Spearman is -0.161368.
7. Pairwise ranking OBJ1 changes S-RANK AUPRC by +0.000188, MRR by -0.016443; RANKING_OBJECTIVE_MISMATCH_NOT_SUPPORTED.
8. Center-equal CTR1 changes overall AUPRC by -0.009257; worst-center AUPRC changes from 0.424801 to 0.393392. Weak-center case counts are small; CENTER_OPTIMIZATION_RANKING_BOTTLENECK_NOT_SUPPORTED.
9. Semantic VIEW2 changes AUPRC versus VIEW0 by -0.014438 and versus random two-branch VIEW1 by +0.004233; SHARED_VIEW_INTERFERENCE_NOT_SUPPORTED.
10. FIT-only linear probe AUPRC by layer: L0=0.362298, L1=0.487586, L2=0.436277, L3=0.557200. L1→L2 loses linear accessibility, but final L3 recovers; this is descriptive, not a causal bottleneck claim.
11. No candidate has strong matched causal support under its prelocked primary ranking gate. Channel-set stability changes are mechanistic diagnostics, not utility confirmation.
12. No single source is strong enough to motivate a ranking-model redesign from these development data alone. Historical outer exposure makes this exploratory, not sealed confirmation.

Exact terminal: `RANKING_BOTTLENECK_SOURCE_UNRESOLVED`.
OUTER_TEST_ACCESSED = NO
