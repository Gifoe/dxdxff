# Raw-Delta Encoder — user-authorized frozen Outer addition

| Model | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR | Top1-EZ |
|---|---:|---:|---:|---:|---:|---:|
| D0 historical engineered DELTA replay | 0.726333 | 0.517057 | 0.624497 | 0.422314 | 0.699468 | 0.575000 |
| E1 engineered DELTA compact | 0.729348 | 0.514746 | 0.619449 | 0.410572 | 0.691941 | 0.575000 |
| D1 Raw-reconstructed DELTA | 0.732636 | 0.527175 | 0.616556 | 0.408531 | 0.689399 | 0.575000 |
| D2 true Raw dynamics | 0.737004 | 0.534363 | 0.619995 | 0.410912 | 0.723159 | 0.637500 |
| D3 shuffled Raw dynamics | 0.737659 | 0.531207 | 0.616865 | 0.402103 | 0.692460 | 0.575000 |

Primary unit is the patient-equal mean over **80 unique Outer-held-out patients**, each evaluated once with its original fold model; 7,635 unique canonical channels, 1,743 EZ and5,892 NEZ. Original fold patient counts are16/16/17/15/16 and channel counts1631/1498/1746/1425/1335. These are not the previous65 Development appearances/47 unique-ID estimand, or pooled-channel metrics.

**Conclusion:** the Raw residual provides a small AUROC/AP increase over D1, but its Macro-F1/EZ-F1 gains are uncertain. Correct temporal order does not beat the separately trained shuffle control in AUROC. D2 also does not exceed historical D0 patient Macro-F1 or EZ-F1. The prior Development terminal remains `RAW_INFORMATION_POSSIBLY_USEFUL_TEMPORAL_ORDER_NOT_ESTABLISHED`; this user-authorized evaluation is not a rescue or a reclassification of the failed scientific gate.

Scientific status: **`USER_AUTHORIZED_POSTHOC_OUTER_EXPLORATORY_REPEATED_TEST`**. Own-fold TEST patients are excluded from model fitting, scaler fitting and threshold selection. The cohort has nevertheless been viewed in historical experiments, and this addition was authorized after seeing Development results. It is not a blind independent confirmation or external clinical validation.

## Frozen protocol and execution

The original Development `PROTOCOL_LOCK.json` remains byte-identical. `OUTER_PROTOCOL_AMENDMENT.json` records the subsequent explicit user authorization. No models were retrained, no checkpoint edited, no imputer/scaler refitted, no threshold reselected and no Raw/physiological features re-extracted. Existing hash-verified E14/R14 and waveform-only banks are reused.

All25 model/fold checkpoints, selected epochs, thresholds, preprocessing hashes, input/registry/split hashes and source/test hashes were locked before inference. Preprocessing was replayed against legal FIT/VAL banks. All25 new prediction files, including the frozen D2 interventions, were atomically saved and sealed before the full feature-archive `y` member was read in this new evaluation. Legal historical FIT/VAL bank labels were available for role/parity checks; this seal is not a claim that no clinical label in the cohort has ever been seen by any other fold or prior researcher.

Freeze SHA256: `1a4504f32a5ee2e2a267533031bce3b2bd48aa83c6504f0f6ad4d96cfb330c4a`.

Prediction-seal SHA256: `3c88ae2fed392006383fabddffd8d117c6901a489d08009033a317600a47dcbf`.

The historical D0 checkpoint replay reproduced the published rounded historical Outer values without reopening historical individual Outer outputs. E1 is the matched compact architecture control; D0 remains a historical trained parameterization, not equivalent new training.

## All classification metrics

| Model | NEZ-F1 | Balanced accuracy | Sensitivity | Specificity | Accuracy | Predicted EZ fraction |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 0.826680 | 0.669411 | 0.488251 | 0.850571 | 0.752569 | 0.216585 |
| E1 | 0.828326 | 0.663749 | 0.470667 | 0.856830 | 0.755088 | 0.207612 |
| D1 | 0.824582 | 0.665741 | 0.477193 | 0.854288 | 0.748950 | 0.210810 |
| D2 | 0.829079 | 0.669566 | 0.477609 | 0.861522 | 0.754253 | 0.207794 |
| D3 | 0.831626 | 0.665597 | 0.460833 | 0.870362 | 0.755514 | 0.196484 |

Full-precision values, five folds and four centers are in `outer_results/OUTER_SUMMARY.csv`, `OUTER_BY_FOLD.csv` and `OUTER_BY_CENTER.csv`. A final aggregate-only denominator check confirmed that AUROC, sensitivity and specificity are estimable for all80 patients in every reported arm. The implementation retains original missingness semantics rather than replacing undefined AUROC with zero; no patient was dropped in this actual evaluation.

## Paired patient bootstrap

10,000 draws, seed42, unique patient-ID clusters; fixed models and numeric thresholds. Unadjusted exploratory percentile95% intervals:

| Contrast | Metric | Difference | 95% interval |
|---|---|---:|---|
| D1−E1 | AUROC | +0.003288 | [-0.003682,+0.010928] |
| D1−E1 | AP | +0.012429 | [-0.005878,+0.038345] |
| D1−E1 | Macro-F1 | -0.002892 | [-0.014186,+0.007288] |
| D1−E1 | EZ-F1 | -0.002041 | [-0.017739,+0.013351] |
| D2−D1 | AUROC | +0.004368 | [+0.000845,+0.008121] |
| D2−D1 | AP | +0.007188 | [+0.002717,+0.012229] |
| D2−D1 | Macro-F1 | +0.003439 | [-0.001122,+0.008215] |
| D2−D1 | EZ-F1 | +0.002381 | [-0.004999,+0.009997] |
| D2−D3 | AUROC | -0.000655 | [-0.004532,+0.003250] |
| D2−D3 | AP | +0.003156 | [-0.001986,+0.008854] |
| D2−D3 | Macro-F1 | +0.003130 | [-0.002010,+0.008853] |
| D2−D3 | EZ-F1 | +0.008809 | [+0.000787,+0.018187] |

All metrics, descriptive E1−D0/D1−D0 contrasts and absolute model intervals are provided. These intervals do not include training-seed uncertainty or historical test-reuse/selection effects. A favorable interval in one metric does not establish general superiority.

## Frozen mechanism interventions

All D2 interventions use **D2's existing per-fold thresholds**, not newly optimized thresholds or D1's numeric thresholds.

| D2 intervention | AUROC | AP | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|
| Full | 0.737004 | 0.534363 | 0.619995 | 0.410912 | 0.723159 |
| gamma=0 | 0.732636 | 0.527175 | 0.617401 | 0.409408 | 0.689399 |
| Frozen stage-local shuffle | 0.737445 | 0.532854 | 0.618988 | 0.409658 | 0.717097 |
| Absolute latent instead of subtraction | 0.735590 | 0.532163 | 0.618095 | 0.407681 | 0.706064 |

Gamma-zero exactly recovers embedded D1 logits. Maximum difference versus separately batched standalone D1 logits is2.384186e-7, and decisions at D1's original threshold match exactly. The gamma-zero Macro-F1 differs from standalone D1 because intervention thresholds remain D2's.

Inference shuffling slightly raises AUROC (+0.000441), while Macro-F1 changes only−0.001008. Together with D3's slightly higher primary AUROC, this does not support correct chronology as the useful mechanism. Absolute-latent replacement causes modest drops, but is an inference-only perturbation, not a retrained absolute-Raw baseline.

Residual utility is uneven. D2 AUROC exceeds D1 in four folds and ties in fold2; most gain is in fold5. Mean absolute residuals, folds1–5, are0.034137/0.000624/0.001056/0.136296/0.133732; folds2–3 remain nearly inactive. Full D2 versus gamma-zero corrects67/spoils41 channel decisions, net+26: TP725→727, FP839→815, TN5053→5077, FN1018→1016. This is mainly fewer false positives, not a large improvement in EZ sensitivity. These intervention counts use D2's frozen thresholds and therefore differ from the separately thresholded D1 baseline.

For the separately selected D1 comparison, pooled counts are D1 TP722/FP844/TN5048/FN1021 and D2 TP727/FP815/TN5077/FN1016. Pooled D2 Macro-F1 is0.644922, not the primary patient mean0.619995. Do not substitute the pooled value to portray a patient-level F1 success.

## Heterogeneity and limits

| Center | Patients | D1 AUROC | D2 AUROC | D1 Macro-F1 | D2 Macro-F1 |
|---|---:|---:|---:|---:|---:|
| HUP |36|0.781675|0.785325|0.665235|0.667159|
| LZU |21|0.639171|0.647846|0.547782|0.557059|
| multicenter |15|0.806417|0.808735|0.612662|0.616321|
| pediatric |8|0.618964|0.619101|0.585337|0.579857|

Pediatric Macro-F1/EZ-F1 worsen; the tiny AUROC increase does not establish center-wide utility. True-order D2 exceeds D3 AUROC in only two folds. No fold/center was removed, no new subset selected and no test-driven iteration followed.

The evidence is consistent with Raw-derived DELTA preserving ranking ability and a small generic Raw residual improving D1's ranking. It does not establish true temporal-order benefit, broad classification improvement over the historical engineered DELTA model, or an independently validated clinical localization method. Development and Outer F1 should not be compared as if the cohorts and selection optimism were identical.

## Verification and engineering provenance

Sixteen deterministic tests passed locally and on server. Independent sklearn calculations verified720 patient-metric cells across nine reported model/intervention arms, all25 primary model/fold cells, pooled confusions, frozen thresholds, all absolute intervals and paired bootstrap via independent multiplicity counts. Source/input/checkpoint/scaler dependencies remain unchanged; `OUTER_INDEPENDENT_VERIFICATION.json` binds public CSV hashes.

Server PyTorch import was initially slow, so contract tests took335 seconds before completing. A prefreeze guard rejected an initially incorrect equal16-per-fold count assumption; the corrected16/16/17/15/16 counts agree with both the hash-pinned original split metadata and the prior split contract, before any Outer scores or full test-label access. No cohort or split was changed.

E: subsequently had zero free space, leaving a newly transferred verifier file empty. It was not accepted as verification. The byte-hash-verified identical verifier was executed from a D: staging directory using the unchanged frozen E: evaluation modules; it passed. Predictions/checkpoints/results remain on D:/C: and were neither changed nor regenerated. No unrelated process or file was stopped/deleted. The local transport's non-UTF8 stderr decoder could hide some filesystem errors, so file hashes and completion JSONs—not a shell exit alone—were required.

Only code, hashes, anonymous aggregate outputs and this report are published. Raw, labels/identities, preprocessing arrays, checkpoints, prediction caches and logs remain private. The complete original Development report and gate artifacts are retained separately and have not been overwritten.
