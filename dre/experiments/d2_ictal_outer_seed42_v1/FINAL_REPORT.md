# ICTAL D2 frozen outer evaluation — seed42

Terminal: `ONE_FROZEN_D2_OUTER_PASS_COMPLETE`.

Scientific status: `USER_AUTHORIZED_POSTHOC_OUTER_EXPLORATORY_REPEATED_TEST`.

This is the original ictal D2 source-conditioned A0 model, not Omni-D2,
GraphRole, or a rank-compressed retraining. It uses the five originally selected
D2 checkpoints and validation thresholds without retraining, threshold changes,
or raw EEG extraction.

## Primary result: patient-equal outer TEST

All 80 patients are evaluated exactly once in their original held-out fold.
The 7,635 canonical patient-channel units include 1,743 EZ and 5,892 NEZ.
Each patient's metric is computed first and the 80 patient metrics are averaged.
All 80 patients have both classes, so all 11 metrics are estimable for every patient.
Intervals are patient-cluster percentile bootstrap intervals, 10,000 draws, seed42,
with the original numeric thresholds held fixed in every draw.

| Metric | Outer estimate | 95% CI |
|---|---:|---:|
| EZ AUROC | 0.730445 | [0.686937, 0.772829] |
| EZ AP / AUPRC | 0.533483 | [0.477494, 0.588071] |
| Macro-F1 | 0.620403 | [0.593651, 0.647664] |
| EZ-F1 | 0.410909 | [0.365616, 0.456747] |
| NEZ-F1 | 0.829898 | [0.802295, 0.856039] |
| Balanced accuracy | 0.652228 | [0.622845, 0.682507] |
| EZ MRR | 0.737216 | [0.652619, 0.819723] |
| Top1 is EZ | 0.650000 | [0.537500, 0.750000] |
| Sensitivity / EZ recall | 0.435823 | [0.377519, 0.497485] |
| Specificity / NEZ recall | 0.868634 | [0.838950, 0.895982] |
| Accuracy | 0.759548 | [0.730095, 0.788241] |

## Frozen selections and outer folds

Thresholds below operate on the NEZ probability: predict EZ iff score < threshold.
The selected epochs are the original D2 epochs, not the D0/A0 reference epochs.

| Fold | Epoch | NEZ threshold | TEST patients | TEST channels | Patient-equal AUROC | AP | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.415 | 16 | 1,631 | 0.711376 | 0.505920 | 0.605008 |
| 2 | 1 | 0.275 | 16 | 1,498 | 0.773011 | 0.465030 | 0.610657 |
| 3 | 5 | 0.365 | 17 | 1,746 | 0.747228 | 0.531773 | 0.634132 |
| 4 | 14 | 0.285 | 15 | 1,425 | 0.624880 | 0.513597 | 0.564513 |
| 5 | 7 | 0.365 | 16 | 1,335 | 0.788083 | 0.649958 | 0.683356 |

The primary overall result weights patients equally, not folds equally. Fold sizes
are 16, 16, 17, 15, and 16; an unweighted average of fold means is a different unit.

## Operational-source results

These are patient-equal metrics within each source, not center-held-out tests.
All sources were available during training on the other patients.

| Source | Patients | Channels | AUROC | AP | Macro-F1 | EZ-F1 | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HUP | 36 | 3,667 | 0.789174 | 0.539655 | 0.672629 | 0.456854 | 0.495382 | 0.908630 |
| LZU | 21 | 2,141 | 0.625872 | 0.550444 | 0.542499 | 0.404532 | 0.420610 | 0.758361 |
| Multicenter | 15 | 925 | 0.813430 | 0.558220 | 0.639219 | 0.397014 | 0.402974 | 0.910114 |
| Pediatric | 8 | 902 | 0.585074 | 0.414805 | 0.554607 | 0.246946 | 0.269330 | 0.900338 |

## Secondary pooled-channel result

Pooling all 7,635 channel units gives AUROC **0.714637**, AP **0.445773**,
Macro-F1 **0.640243**, EZ-F1 **0.433200**, NEZ-F1 **0.847286**, balanced accuracy
**0.633828**, sensitivity **0.402754**, specificity **0.864902**, and accuracy
**0.759398**. Confusion counts, with EZ positive: TP=702, FP=796, TN=5,096,
FN=1,041. Each channel uses its own fold's frozen threshold.

Do not confuse these pooled quantities with the primary patient-equal metrics.
Pooled AUROC includes comparisons between scores produced by independently trained
fold models; this is a secondary diagnostic, not the primary localization unit.
Patient-local MRR and Top1 are not computed by ranking channels across different patients.

## Development versus outer

The previously reported D2 development metrics were AUROC **0.718221**, AP
**0.519936**, Macro-F1 **0.647926**, and EZ-F1 **0.443199**. Those are 65
patient-fold validation appearances covering only 47 unique patients, with 6,273
channel appearances. They are not outer TEST metrics and not a paired comparator
to this 80-patient result.

The outer result has lower thresholded Macro-F1 and EZ-F1 than those development
summaries. Ranking metrics cannot establish a D2-versus-A0 improvement here because
this amendment evaluates D2 only and does not generate a matched A0 outer result.
It also cannot establish superiority over historical A1 development metrics.

The practical limitation is low EZ recall: approximately 44% at the patient-equal
level and 40% pooled. The pooled confusion matrix contains more missed EZ channels
(1,041 FN) than true detected EZ channels (702 TP). Source performance is uneven,
particularly pediatric AUROC 0.5851 and EZ-F1 0.2469. These findings do not justify
calling D2 a strong or uniformly generalizing classifier.

## Provenance, label gate, and execution

- Original experiment: `a0_source_posterior_decoder_seed42_v1`, public commit
  `394dbff2085114d5f2d162cc7802384c95f0ef8d`, original model size 9,221 parameters.
- The original protocol disabled outer TEST and its continuation gate did not pass.
  This run is a separate user-authorized posthoc amendment; it does not retroactively
  change or pass the original gate.
- Exact feature NPZ, feature order, split CSV, original code, five checkpoint files,
  five FIT+VAL banks, and development prediction bindings were hash-checked.
- Original patient-local z-normalization and stored FIT-only imputer/scaler were
  reused. A FIT-only reconstruction control matched every frozen moment and every
  FIT+VAL transformed feature bitwise. No TEST normalization moments were fitted.
- CPU inference used the original Python runtime, PyTorch 2.11.0+cu128, two threads.
  FIT+VAL replay probability drift was at most 1.7881393433e-7; maximum logit drift
  was 1.4305114746e-6, inside the predeclared 2e-7 and 2e-6 gates.
- All five selections were sealed before outer inference. All five outer prediction
  artifacts were atomically saved and hash-sealed before first full-cohort `y` read.
  The feature export's FIT+VAL clinical labels then matched the original banks exactly.
- Pre-test freeze SHA256:
  `f82bba4564cd4020ccd942160c2fd299b61147ce7080b5b6876765f40cfaa43b`.
- Outer prediction-seal SHA256:
  `2fda68490c84eac173e061178c774d13d464e12f485ee01681a1cba596c91ffd`.
- Four synthetic contract tests passed. A pre-freeze schema check was corrected to
  read the original checkpoint binding's `source` field; no TEST labels or outer
  predictions had been accessed at that point, and the failed source copy was retained
  on the server. No checkpoint, preprocessor, metric, or threshold changed.
- An independent read-only sklearn/NumPy verifier checked all 80 x 11 patient
  metrics, pooled confusion/F1, and all 10,000-draw confidence intervals; maximum
  per-patient metric difference was 2.2204460493e-16. It did not rerun the model.
- All frozen historical dependencies remained unchanged after evaluation.
- Earlier outer experiments on this cohort have been viewed. This is exploratory
  repeated-test evidence, not blind confirmation or a fresh external validation.
- Raw data, features, checkpoints, patient/channel predictions, bootstrap draw files,
  and runtime logs remain private on the server. Only source and aggregate artifacts
  are published.

Full precision results are in `results/OUTER_PATIENT_EQUAL_METRICS.csv`; source and
fold tables contain all 11 metrics. No further model iteration is part of this task.
