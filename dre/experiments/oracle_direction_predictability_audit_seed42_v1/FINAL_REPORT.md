# Oracle-direction predictability audit — development-only

No outer predictions or metrics were computed. The legacy cache initializer did materialize all 80 labels, so the literal no-outer-label-read requirement was not satisfied; this is not a sealed result. The 65 validation cells comprise only 47 distinct patients; 10,000 paired bootstrap resamples use patient-ID clusters, as recorded before aggregate gating in BOOTSTRAP_UNIT_AMENDMENT.json.

Exact A1 replay: 150/150 checkpoints, max grid error 0; mean VLOO Macro-F1 0.625996. R3/R4 dimension 64 and R4 classifier replay passed.

## Fixed 65-cell findings

Oracle-direction inner-fit stability: mean 0.860, median 0.872, q10/q90 0.755/0.949. No low-stability patient was excluded.
FIT oracle-direction cumulative variance explained: rank 1 0.082, rank 2 0.159, rank 4 0.292, rank 8 0.494.
Label-using full oracle AP 0.9339; label-using rank-4 projection AP 0.6104, median direction cosine 0.382, fraction cosine >=.75 0.015, shared FIT logistic AP 0.5225, rank-4 headroom retention 0.214. These oracle figures are nondeployable upper bounds.
FIT-only unlabeled context-distance/direction-cosine associations: R3_MARG rho=-0.007; R3_COV rho=+0.011; R4_MARG rho=-0.050; R4_COV rho=-0.079. Negative rho would mean closer contexts have more similar oracle directions; repeated FIT pairs make the bootstrap descriptive.
Best context KNN D2 AP 0.4870 (R3_MARG); best D3/D4 mean AP 0.5314 (D4_R4_MARG) versus shared D1 0.5225 and frozen A1 0.5572.
The four predeclared R3/R4 marginal/covariance contexts, all D2/D3/D4 directions, wrong-context controls, angle/cosine and paired 10,000-resample CIs are in the aggregate CSVs. The public tables report whether covariance adds information; no context was added or tuned after inspection.

## Answers to the registered questions

1. The patient-specific inner-fit oracle directions are reasonably stable within a patient (mean pairwise cosine 0.860, median 0.872, q10 0.755). No unstable case was removed.
2. FIT oracle-direction residual variance explained by ranks 1/2/4/8 is 0.082/0.159/0.292/0.494. The rank-4 direction subspace captures less than one-third of FIT variation.
3. The registered rank-4 direction reconstruction fails every expressivity criterion: median cosine to the full oracle 0.382, only 1/65 cells reaches cosine 0.75, and only 21.4% of the oracle-minus-shared AP headroom is retained. This argues against a four-dimensional *direction* correction; it is not a proof that every rank-4 feature rotation is mathematically incapable.
4. R3 contexts contain little usable predictive information in these fixed models. Their strongest D4 AP is 0.5113, below shared D1's 0.5225; direction cosine reaches only 0.1534.
5. R4 marginal context is relatively stronger but still inadequate: D4 mean cosine 0.1972, median 0.1922, AP 0.5314. That is +0.0088 AP versus D1 but below D0 population mean (0.5520) and frozen A1 (0.5572).
6. Adding covariance does not improve prediction: for each D2/D3/D4 method, both R3_COV and R4_COV yield lower AP than their corresponding marginal summaries. R4_COV has a somewhat stronger FIT pairwise association but not a useful prediction gain.
7. FIT patients with closer R4 contexts tend to have slightly more similar oracle directions (distance–cosine Spearman rho -0.050 for MARG, -0.079 for COV). The effect is weak and repeated FIT pairs make the reported intervals descriptive, not independent evidence. R3 association is near zero.
8. Three-nearest-patient context transfer does not work here: the best D2 AP is 0.4870, below D1 and D0.
9. Direct full-direction ridge does not predict the oracle well: its best mean cosine is 0.1554 (R4_MARG), and its best AP is 0.5092, below D1.
10. Rank-4 coefficient ridge is the strongest learned candidate, but R4_MARG still reaches only mean cosine 0.1972. It improves cosine over D0 in fewer than 65% of cells and has no positive lower bootstrap bound for that improvement.
11. No learned predictor meets the required ranking gain over shared FIT linear. Best D4 R4_MARG AP gain is +0.0088, positive in 2/5 folds; its patient-cluster bootstrap 95% CI is [-0.0199, +0.0424]. Its AP is also lower than the frozen A1 score.
12. For best D4 R4_MARG, correct context exceeds cyclic wrong context by +0.0188 AP and +0.0399 cosine. This control passes in isolation, but the correct-context predictor itself remains far below the full gate. The failed P2/P3 feature adapter had the opposite wrong-context behavior; neither observation proves useful zero-label oracle prediction.
13. The first P2/P3 failure is compatible with both insufficient rank-4 direction geometry and weak predictability from the current unlabeled summaries. It cannot be pinned solely on the feature-rotation mechanism. The earlier patient-specific oracle headroom is not refuted; the in-sample full-oracle AP of 0.9339 is a label-using upper bound and is not directly the earlier cross-fitted AP estimand.
14. A direct patient-conditioned low-rank readout is **not** justified by these fixed gates. Do not build another adapter or run outer test on this basis.

## Causal interpretation

The prior patient-specific direction headroom can be genuine while remaining nondeployable. This audit separates representational rank-4 expressivity from prediction using zero-label patient statistics; the three outcomes are not interchangeable.
A direct patient-conditioned low-rank readout is justified only if both fixed rank-4 and D4 predictability gates pass. No adapter or outer test was run.

`RANK4_ORACLE_SUBSPACE_INSUFFICIENT`
`UNLABELED_CONTEXT_DOES_NOT_PREDICT_ORACLE_DIRECTION`
`CURRENT_ORACLE_DIRECTION_TRANSFER_MECHANISM_UNRESOLVED`
