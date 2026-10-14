# Raw-Delta Encoder — Seed42 Development

| Model | Input | Total parameters | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| Historical D0, replayed | Engineered DELTA | 8,817 | 0.716600 | 0.504414 | 0.642347 | 0.443493 | 0.695850 |
| E1 | Engineered DELTA compact | 1,537 | 0.715424 | 0.509553 | 0.642887 | 0.440484 | 0.682050 |
| D1 | Raw-reconstructed DELTA | 1,537 | 0.725978 | 0.503601 | 0.644546 | 0.443537 | 0.658502 |
| D2 | Frozen D1 + true Raw dynamics | 13,643 | 0.732515 | 0.515027 | 0.650238 | 0.451028 | 0.684968 |
| D3 | Frozen D1 + shuffled dynamics | 13,643 | 0.727976 | 0.509847 | 0.649008 | 0.445928 | 0.669576 |

**Terminal: `RAW_INFORMATION_POSSIBLY_USEFUL_TEMPORAL_ORDER_NOT_ESTABLISHED`.**

All authorized stages actually ran on the trusted server: Raw reconstruction, D0 Development replay, ten E1/D1 and ten D2/D3 training cells, frozen interventions and10,000 paired ID-cluster draws. No new Outer inference/evaluation, historical individual Outer-output access, installed skills, extra seeds or posthoc rescue occurred.

Primary estimand: **65 original Development patient-fold appearances /47 unique identities**. Not pooled channels, not the secondary unique-ID estimator and not Outer. D0 matches its historical rounded Development references; E1 is the mandatory equivalent-parameterization control, whereas D0 is not optimizer-equivalent. D2/D3 each have12,106 trainable plus1,537 frozen parameters. Validation patients selected checkpoints/thresholds; results are extensively reused, exploratory and selection-conditioned, not independent confirmation.

## Other requested metrics

| Model | NEZ-F1 | Balanced accuracy | Sensitivity | Specificity | Accuracy | Predicted EZ fraction | Top1-EZ |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0 | 0.841201 | 0.689779 | 0.524209 | 0.855350 | 0.769519 | 0.206949 | 0.600000 |
| E1 | 0.845291 | 0.686732 | 0.510365 | 0.863099 | 0.774382 | 0.202071 | 0.553846 |
| D1 | 0.845556 | 0.686159 | 0.506245 | 0.866074 | 0.775170 | 0.194371 | 0.523077 |
| D2 | 0.849449 | 0.692227 | 0.512597 | 0.871856 | 0.780388 | 0.191448 | 0.569231 |
| D3 | 0.852088 | 0.687553 | 0.494540 | 0.880565 | 0.782405 | 0.181422 | 0.538462 |

All CSVs retain full precision. Original NEZ-positive orientation, fixed folds, FIT class weighting, patient-equal updates, stopping and threshold ties are preserved.

## Reconstruction and admission gates

Stage A **PASS**:80/256/7,635 identity;1,471,965 genuine channel-windows with exact500-sample parity; all106,890 E14 coordinates bit-identical after the documented original float32 reduction-layout correction. The initial3-coordinate mismatch and native NumPy interruption are retained, not hidden. No tolerance was relaxed. All5 A/B families pass; mean correlations use79 estimable FIT-union patients, SD72 with seven nonestimable cases disclosed.

Beta/low gamma are effectively numerically recovered. RMS/variance agree closely. Line length has normalized MAE approximately0.173/0.102 (mean/SD), despite high rank agreement. Partial high gamma has0.373/0.340 and entropy0.164/0.168; they are Category C, not gate families. Raw250 cannot restore125–150 Hz or certify device/EDF filters or independent onset clocks. Feature/Raw sample bounds use1000/250-Hz units; after converting to seconds, median differences are0 and maximum0.001 s, not an onset-provenance validation.

Stage B **PASS**: D1−E1 AUROC+0.010554/AP−0.005952/Macro-F1+0.001659/EZ-F1+0.003053 all meet the locked feasibility tolerances, while D1 AUROC≥0.700 and Macro-F1≥0.620. D1 consumes Raw-derived measurements, never engineered E14. E1's fourteen preprocessed columns exactly match historical preprocessing. Historical compact parity maximum float32 logit drift5.960464e-7, score1.192093e-7, float64 synthetic4.440892e-16; validation decisions and selections are exact.

## Paired patient-cluster bootstrap

| Contrast | Metric | Difference | Exploratory95% interval |
|---|---|---:|---|
| D1−E1 | AUROC | +0.010554 | [-0.000012,+0.022477] |
| D1−E1 | AP | -0.005952 | [-0.024813,+0.009256] |
| D1−E1 | Macro-F1 | +0.001659 | [-0.010680,+0.013744] |
| D1−E1 | EZ-F1 | +0.003053 | [-0.017934,+0.022854] |
| D2−D1 | AUROC | +0.006537 | [+0.000537,+0.012527] |
| D2−D1 | AP | +0.011426 | [+0.004663,+0.018725] |
| D2−D1 | Macro-F1 | +0.005692 | [+0.000513,+0.010672] |
| D2−D1 | EZ-F1 | +0.007491 | [+0.000024,+0.014858] |
| D2−D3 | AUROC | +0.004539 | [+0.000473,+0.008685] |
| D2−D3 | AP | +0.005179 | [+0.000106,+0.010317] |
| D2−D3 | Macro-F1 | +0.001231 | [-0.002751,+0.005148] |
| D2−D3 | EZ-F1 | +0.005101 | [-0.000924,+0.011258] |

Seed42/10,000 draws resample47 unique IDs retaining repeats, with fixed weights/thresholds. Intervals are unadjusted and conditional on selected models; they do not account for selection uncertainty, training-seed variation or independent replication. `PAIRED_BOOTSTRAP.csv` includes all requested metrics and descriptive E1−D0/D1−D0 contrasts.

## Why the complete gate fails

D2−D1 AUROC+0.006537 < registered+0.010. D2−D3 AUROC+0.004539 < registered+0.005. These two failures stand despite positive intervals or proximity to the cutoff. Other conditions pass: AP/EZ-F1 do not decrease, Macro-F1 gain exceeds+0.005, specificity improves+0.005782, D2 AUROC exceeds D1 in all5 folds, D2−D3 interval lower bound is positive and the actual residual has a small useful effect.

Inference-only D2 shuffling gives AUROC0.732724/AP0.516315, slightly higher than true order0.732515/0.515027; Macro-F1 drops only0.001348. Thus the separate trained-control gain does not robustly establish chronological order as its cause. Most D2−D3 gain is in fold5, with differing selected epochs/gates. No fold or threshold was changed to rescue this finding.

## Secondary estimator, folds and centers

Unique47-ID secondary AUROC/Macro-F1: D0 0.736154/0.657379; E1 0.738404/0.659136; D1 0.746368/0.658608; D2 0.750689/0.662808; D3 0.747481/0.661973. These do not replace the primary estimator.

| Center | Appearances | D1 AUROC | D2 AUROC | D1 Macro-F1 | D2 Macro-F1 |
|---|---:|---:|---:|---:|---:|
| HUP |28|0.783749|0.786913|0.677116|0.686375|
| LZU |14|0.632695|0.637762|0.562578|0.568192|
| multicenter |11|0.887536|0.891587|0.767753|0.764476|
| pediatric |12|0.551913|0.570315|0.551241|0.556923|

All-center AUROC increases, but multicenter Macro-F1/EZ-F1 decline and pediatric remains weak. Fold2/3 residuals have almost no decision effect; fold5 drives much of the average gain. Full five-fold and center tables are provided without selective omission.

## Answers to the fourteen research questions

1. Raw250 can reproduce the registered comparable-family rank and predictive DELTA behavior, not the entire native-rate physical representation.
2. Beta/low gamma agree most closely; RMS/variance are close; line length preserves ranks but differs in scale.
3. Full80–150-Hz high gamma is impossible at250 Hz; entropy is bandwidth-dependent; time-domain/filter and onset provenance remain uncertain.
4. D1 retains patient-relative EZ/NEZ discrimination and passes the locked predictive recovery gate.
5. It approaches historical D0 under the same Development estimand; the matched E1 control does not establish statistically confirmed Raw superiority.
6. D2 adds small exploratory gains in all four primary classification/ranking metrics, below the registered AUROC minimum effect.
7. True order has a small separately-trained AUROC advantage, but effect-size failure and no inference-shuffle AUROC loss prevent a temporal-order claim.
8. D2−D1 AUROC/AP/Macro-F1/EZ-F1 point differences and exploratory intervals are positive; D2−D3 F1 intervals cross zero.
9. At frozen D2 thresholds, residual versus gamma-zero increases TP13 and reduces FP28 across6,273 channel appearances. Sensitivity/specificity improve while mean predicted EZ fraction falls: not merely predicting more EZ.
10. AUROC rises versus D1 in all five folds/centers, but gains and F1 are heterogeneous. This is not uniform center generalization.
11. Gamma-zero reproduces D1 logits exactly. Effective gates0.001839–0.202459 show some nearly inactive folds; interventions indicate small actual use, not universal utility. Gamma-zero retains D2's threshold, explaining its different F1. Absolute-latent intervention is not a retrained baseline.
12. The staged test is more defensible than prior Raw-only/Raw-MIL assertions because physiological recovery, matched controls, base identity and mechanisms are checked. That does not make the resulting model claim stronger than the observed effects.
13. These data support a recovery finding and small residual signal, not a validated JBHI temporal-localization model. Treating the full gate as passed would overstate the evidence.
14. Independent patients, multiple training seeds, native-rate EDF/filter verification and independently validated onset timing remain necessary under separate authorization. None was run as automatic rescue.

## Verification and publication

Thirteen deterministic tests passed locally/on server. Independent sklearn metrics and an independent multiplicity-bootstrap calculation verified20 new training cells and585 patient-metric cells. `INDEPENDENT_VERIFICATION.json` binds public CSV hashes. Publish only source/tests/hashes/anonymous aggregates/reports; Raw, IDs/labels, banks, checkpoints, scalers, predictions and logs remain private. All jobs completed; no additional or Outer experiment is active.
