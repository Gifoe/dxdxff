# Selected M8 versus frozen D1 — completed Outer

**Terminal: `OUTER_POINT_GAIN_BUT_TWO_POINT_MACRO_F1_TARGET_NOT_MET`.** The actual80-patient/7,635-channel Outer was run, not inferred from Development. M8 improves the ranking point estimates and all five fold Macro-F1 point estimates, but the primary Macro-F1 gain is **0.01754063 (1.754063 percentage points)**, below the requested0.02. Its paired95% interval includes zero. Do not describe this as a confirmed two-point F1 improvement.

| Patient-equal metric | Frozen D1 | Selected M8 | M8 minus D1 | Paired95% CI |
|---|---:|---:|---:|---:|
| AUROC |0.732636|0.762397|+0.029761|[0.006912,0.056661]|
| AP |0.527175|0.554067|+0.026892|[0.003448,0.051300]|
| Macro-F1 |0.616556|0.634097|+0.017541|[-0.001153,0.036806]|
| EZ-F1 |0.408531|0.434087|+0.025556|[-0.006790,0.059056]|
| NEZ-F1 |0.824582|0.834107|+0.009526|[0.002121,0.017317]|
| Balanced accuracy |0.665741|0.681352|+0.015612|[-0.002208,0.035203]|
| EZ sensitivity |0.477193|0.497392|+0.020199|[-0.014773,0.058891]|
| NEZ specificity |0.854288|0.865313|+0.011025|[-0.000436,0.022310]|
| MRR |0.689399|0.749310|+0.059911|[-0.007945,0.129772]|
| Top1-EZ |0.575000|0.637500|+0.062500|[-0.037500,0.162500]|
| Accuracy |0.748950|0.760166|+0.011216|[0.000250,0.022507]|
| Predicted EZ fraction |0.210810|0.209195|-0.001615|[-0.013499,0.010725]|

Full precision, absolute intervals, five folds and center aggregates are in `outer_results/`. Primary averages weight each of80 unique held-out patients equally, once, under its original own-fold TEST checkpoint and numeric threshold. No Development appearance is added to this denominator. Ten thousand paired patient-ID bootstrap draws use seed42 and keep every numeric threshold fixed. Intervals are unadjusted, selection-conditioned and do not account for the adaptive eight-candidate Development search.

## Frozen protocol and actual execution

The user explicitly authorized this Outer after the completed M8 Development selection. `OUTER_PROTOCOL_AMENDMENT.json` adds that scope; all three original phase locks/results remain unchanged. Only D1 and the globally selected M8 are evaluated, not the seven other candidates. No new model fitting, weight modification, imputer/scaler fitting, threshold search, Raw/EDF extraction or physiological extraction occurred.

All ten selected checkpoint hashes, five original D1 bank preprocessors, five FIT-only conditioning scales, input registries and source hashes were sealed before inference. M8 thresholds stay0.410/0.255/0.295/0.365/0.395; D1 keeps its own pre-existing selections. Scores for all ten model/fold cells were atomically sealed **before this evaluation's full TEST label read**. That is an execution safeguard, not a claim that these cohort labels were historically unknown. Every patient/model private cache is bound to that freeze, checkpoint and exact archive indices. There was one successful frozen inference pass with no interruption/retraining. Three server contract tests passed before the freeze.

The cohort remains80 patients,256 seizures,7,635 unique canonical channels (1,743EZ/5,892NEZ). Fold counts remain16/16/17/15/16 patients and1,631/1,498/1,746/1,425/1,335 channels. Genuine window masks, within-patient context and original250Hz Raw banks remain unchanged. Existing preprocessing is applied without fitting. The inference graph is the delivered M8 factory, with9,481 new trainable parameters in its original training, and frozen1,537-parameter D1; this run has no trainable update.

Embedded D1 versus separately replayed D1 logit maximum difference is zero in all five folds. The cap1 correction is verified. The new D1 aggregate exactly matches the previously published original D1 Outer values; those values were not used to alter this model or its threshold.

`OUTER_INDEPENDENT_VERIFICATION.json` is PASS: independent sklearn calculations verify160 patient metric cells, all fold/center summaries, pooled confusions, ten original checkpoint/bank selections and all primary/absolute bootstrap intervals. Bootstrap intervals are independently reproduced by multinomial patient multiplicity counts, without Raw/model replay. Downloaded CSV hashes agree with the trusted-server verification record.

## Fold and center behavior

| Fold | D1 Macro-F1 | M8 Macro-F1 | Gain |
|---|---:|---:|---:|
|1|0.600166|0.616079|+0.015913|
|2|0.579680|0.586124|+0.006444|
|3|0.634277|0.655064|+0.020787|
|4|0.593764|0.601743|+0.007979|
|5|0.672363|0.708143|+0.035780|

All five fold AUROC/AP/Macro-F1 averages improve. Across HUP/LZU/multicenter/pediatric, Macro-F1 increases0.011689/0.005974/0.042077/0.028231. These descriptive aggregates do not prove center-specific significance, particularly pediatric(n=8). LZU EZ-F1 and sensitivity decline slightly despite improved AUROC/Macro-F1; HUP and LZU patient ranking do not uniformly improve every metric. Do not claim uniformly better patient decisions.

Secondary pooled-channel Macro-F1 is0.640231→0.652210, AUROC0.689384→0.703456 and AP0.393823→0.439382. These are **not** the primary patient-equal estimates. Pooled EZ-positive confusion terms are D1 TP722/FP844/TN5,048/FN1,021 and M8 TP747/FP799/TN5,093/FN996:25 fewer FN and45 fewer FP. Foldwise thresholds and cross-patient score calibration explain why pooled and patient-equal ranking/F1 have different values; no global Outer threshold is optimized.

## Interpretation and stop

The earlier Development Macro-F1 gain0.022248 was not a guarantee. Here it reduces to0.017541; the F1 interval crosses zero although the AUROC/AP intervals do not. Ranking improvement is better supported by these exploratory paired results than a stable thresholded F1 improvement. We do **not** choose a new threshold, retrain, change the loss or select another candidate to rescue the target after viewing these results.

Scientific status remains **`USER_AUTHORIZED_POSTHOC_OUTER_EXPLORATORY_REPEATED_TEST`**. Own-fold FIT/validation excludes that fold's TEST identities. However, architecture M8 was adaptively selected using repeatedly viewed Development appearances that can overlap other folds' TEST IDs, and the cohort has historical outcome exposure. This is not fully nested independent architecture selection, a blind holdout or clinical confirmation. It supports a bounded exploratory ranking increment, not novel physiological causality or an independently confirmed two-point Macro-F1 gain.

Only source, hashes and anonymous aggregates are published. Raw signals, channel/patient identity, individual predictions, labels, weights, preprocessing state and logs stay on the trusted server. No unrelated process/data was changed; no additional model/seed/Outer iteration is launched.

## Reproduction

With the original private artifacts and existing environment paths, run `tools/test_outer_contract.py`, then `tools/evaluate_outer_m8.py freeze`, `tools/evaluate_outer_m8.py evaluate`, and `tools/verify_outer_m8.py`. Set the same D1_PHASE1_CODE/D1_PHASE2_CODE/DGM_COMPLETED_CODE/DGM_BASE_CODE variables as the completed training and OMP/MKL/OpenBLAS thread limits1. An identical completed runtime returns without new inference. A mismatched source/checkpoint/scale/freeze or private cache is rejected, not silently overwritten.
