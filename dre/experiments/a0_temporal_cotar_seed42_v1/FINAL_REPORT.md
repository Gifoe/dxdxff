# A0 + Temporal CoTAR: completed T1/T2 development evaluation

Terminal: **TEMPORAL_FEATURES_USEFUL_COTAR_NOT_SUPPORTED**.

All ten requested T1/T2 runs completed. Neither provides a convincing primary
Macro-F1 increment. T2 is numerically below the equally parameterized T1 control
and fails the continuation gate. The terminal's "useful" denotes the prompt's
numerical T1-positive/T2-not-better branch, not proven clinical utility. No outer
evaluation, T3, raw-waveform training, SSL, rescue tuning or A0 retraining occurred.

## Complete matched metrics

These are patient-equal means across the same65 validation patient-fold appearances
(47 unique patients;6,273 channel appearances). Each fold has13 validation patients.
The validation sets selected checkpoints and thresholds: these are development
results, not independent outer-test estimates or the historical0.616167 outer
baseline. EZ=0; output is a NEZ-positive logit, EZ score=sigmoid(-logit).

| Metric | T0 frozen A0 | T1 local temporal control | T2 temporal CoTAR |
|---|---:|---:|---:|
| Patient Macro-F1 | 0.638080 | 0.641199 | 0.640770 |
| EZ-F1 | 0.431198 | 0.439072 | 0.439989 |
| NEZ-F1 | 0.844961 | 0.843326 | 0.841551 |
| Balanced accuracy | 0.665436 | 0.669397 | 0.670066 |
| EZ-AUPRC | 0.518235 | 0.522287 | 0.523259 |
| EZ-AUROC | 0.710667 | 0.708445 | 0.711127 |
| EZ-MRR | 0.686550 | 0.696620 | 0.698213 |
| Top1-is-EZ | 0.584615 | 0.600000 | 0.600000 |
| EZ sensitivity | 0.465904 | 0.479433 | 0.484489 |
| NEZ specificity | 0.864968 | 0.859360 | 0.855644 |
| Accuracy | 0.773174 | 0.772802 | 0.769942 |

Exact values: [VALIDATION_SUMMARY.csv](results/VALIDATION_SUMMARY.csv).

### Paired uncertainty

10,000 seed42 patient-ID cluster draws retain all appearances of a sampled ID.
Channels and repeated appearances are not independent bootstrap units. Selected
thresholds/checkpoints are fixed inside draws. Intervals are percentile95% CIs;
positive bootstrap fractions are descriptive, not p-values. These intervals do
not remove development-selection optimism or adjust for the33 contrasts.

| Contrast | Metric | Delta | 95% cluster interval |
|---|---|---:|---:|
| T1-T0 | Macro-F1 | +0.003119 | [-0.002485, +0.008989] |
| T2-T0 | Macro-F1 | +0.002690 | [-0.003441, +0.009001] |
| T2-T1 | Macro-F1 | -0.000429 | [-0.004416, +0.003116] |
| T1-T0 | EZ-F1 | +0.007874 | [-0.001009, +0.017076] |
| T2-T0 | EZ-F1 | +0.008791 | [-0.001587, +0.019105] |
| T2-T1 | EZ-F1 | +0.000917 | [-0.006290, +0.006596] |
| T1-T0 | EZ-AUPRC | +0.004051 | [-0.000313, +0.010751] |
| T2-T0 | EZ-AUPRC | +0.005024 | [+0.000429, +0.011962] |
| T2-T1 | EZ-AUPRC | +0.000972 | [-0.000324, +0.002453] |
| T1-T0 | EZ-AUROC | -0.002222 | [-0.009988, +0.003181] |
| T2-T0 | EZ-AUROC | +0.000460 | [-0.003587, +0.004224] |
| T2-T1 | EZ-AUROC | +0.002682 | [-0.001405, +0.009183] |
| T2-T0 | Sensitivity | +0.018585 | [+0.002261, +0.035698] |
| T2-T0 | Specificity | -0.009325 | [-0.019199, -0.000219] |

All33 contrasts across11 metrics: [PAIRED_BOOTSTRAP.csv](results/PAIRED_BOOTSTRAP.csv).
The AP increase is a small secondary development signal, not evidence CoTAR beats
T1 or that the primary objective succeeded. Sensitivity trades off specificity.

## Folds, centers and model selection

| Fold | T0 Macro-F1 | T1 Macro-F1 | T2 Macro-F1 | T1 epoch / NEZ tau | T2 epoch / NEZ tau |
|---|---:|---:|---:|---:|---:|
| 1 | 0.646844 | 0.652196 | 0.649286 | 19 / 0.380 | 5 / 0.450 |
| 2 | 0.657246 | 0.660255 | 0.660255 | 5 / 0.265 | 5 / 0.265 |
| 3 | 0.642299 | 0.645711 | 0.645711 | 2 / 0.340 | 2 / 0.340 |
| 4 | 0.640909 | 0.640909 | 0.640909 | 0 / 0.355 | 0 / 0.355 |
| 5 | 0.603102 | 0.606923 | 0.607689 | 6 / 0.455 | 9 / 0.435 |

T2 improves over T0 in4/5 folds, but beats T1 only in fold5, loses in fold1 and
ties in folds2-4. Fold4 selects the eligible zero-residual epoch0. Equal metrics
elsewhere do not imply identical trained weights: checkpoint hashes differ.
No bad run was discarded or repeated based on outcomes.

| Center | Appearances / unique IDs | T0 Macro-F1 | T1 Macro-F1 | T2 Macro-F1 |
|---|---:|---:|---:|---:|
| HUP | 28 / 21 | 0.669631 | 0.670042 | 0.670433 |
| LZU | 14 / 11 | 0.590312 | 0.595339 | 0.591297 |
| Multicenter | 11 / 9 | 0.740741 | 0.735337 | 0.742946 |
| Pediatric | 12 / 6 | 0.526085 | 0.541109 | 0.535612 |

T2 center gains range +0.000802 to+0.009528. They are descriptive small aggregates,
not four independent replications. T1 worsens Multicenter. Full11-metric results:
[VALIDATION_BY_FOLD.csv](results/VALIDATION_BY_FOLD.csv),
[VALIDATION_BY_CENTER.csv](results/VALIDATION_BY_CENTER.csv).

## Gate

| Prespecified condition | Observed T2 result | Pass |
|---|---:|---|
| T2-T0 Macro-F1 >=+0.015 | +0.002690 | No |
| T2-T1 Macro-F1 >=+0.005 | -0.000429 | No |
| T2-T0 EZ-AUPRC >=0 | +0.005024 | Yes |
| T2-T0 EZ-F1 >=0 | +0.008791 | Yes |
| T2 improves over T0 in >=4/5 folds | 4/5 | Yes |
| T2 improves over T1 in >=3/5 folds | 1/5 | No |
| Valid masks/isolation and no severe saturation | Passed; T2 saturation0 | Yes |

The +0.030 to+0.040 target is not reached. No bound, interval, block count, cohort,
loss or selection rule changed after outcomes. [VALIDATION_GATE.json](results/VALIDATION_GATE.json)
records the three failed continuation conditions. Stop this version.

## Temporal availability and source limitation

All80 patients,7,635 channels and256 seizures are retained, including the seizure
excluded only by the earlier raw-E3 crop. Original cache/split hashes match the
prompt. Source times are ordered and unique:253 records have59 windows and three
have51/40/56. Windows are2s, spacing approximately1s (observed extrema
0.9999990463/1.0000009537s). Nine selected features are finite. Twenty-seven
source windows potentially touching boundary padding are masked invalid.

For fixed stages [-10,0),[0,8),[8,20), every unique patient-channel has at least
one valid pre window, two early windows and one later window in at least one
seizure: all three cohort fractions100%. Every seizure has some early evidence.
There are25,267 canonical seizure-channel union slots and272 missing incidences;
record-channel stage coverage98.9235%. Missing slots remain masked; no interpolation,
position-only matching, patient dropping or synthetic zero observations occur.
A nonzero residual requires sufficient early evidence in at least one seizure.

Onset-relative coordinates are established from source acquisition/onset/valid
crop metadata, not an independent EDF/raw replay. Padding metadata is handled
before selecting genuine full windows. See [TEMPORAL_DATA_AUDIT.md](TEMPORAL_DATA_AUDIT.md),
[TEMPORAL_COVERAGE_AUDIT.json](audit/TEMPORAL_COVERAGE_AUDIT.json),
[TEMPORAL_COVERAGE_BY_CENTER.csv](results/TEMPORAL_COVERAGE_BY_CENTER.csv), and
[TEMPORAL_RECORD_COVERAGE_HISTOGRAM.csv](results/TEMPORAL_RECORD_COVERAGE_HISTOGRAM.csv).
Identifiable seizure-level audits remain private.

## A0 replay, controlled implementation and tests

Five original A0 checkpoints/preprocessors are reused. Architecture, thresholds
and metrics reproduce; probability differences from the prior serialized ledger
are <=1.672783e-7, within the predeclared2e-7 float32 gate, not bitwise zero across
runtimes. Matched Macro-F1=0.6380797828499001. Both temporal models have exactly
zero residual and bitwise-equal logits to the current A0 replay at epoch0. A0
and teacher parameters are not optimized and receive no residual gradients.

T1/T2 each have5,797 trainable parameters and identical per-fold initial state
hashes. Frozen A0 has8,817. Both use18D absolute/label-blind relative inputs,
actual-time16D embeddings, one block, identical FFN/head/masks/stage readout,
FIT-only normalization, optimizer, patient order and hard-label weighted BCE.
Only local per-token core (T1) versus masked temporal softmax/aggregation/broadcast
(T2) differs. Across-seizure pooling uses masked population mean/std, with no
electrode attention, learned patient-global bias or prevalence predictor.

Copied-weight unmasked T2 core forward is exactly float32-equivalent to pinned
official TeCh. Invalid-token perturbations are invariant; all-missing inputs have
finite outputs/gradients. Seven synthetic groups and three real FIT-patient smoke
updates per arm passed before training. They cover masking/alignment/robust
reference, encoder parity/interaction, state equality, pooling, orientation,
epoch0, FIT normalization, OOF isolation and interrupted-resume parity. Tests
establish implementation correctness, not benefit or temporal-order causality.
See [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md) and `audit/`.

## OOF-to-validation score mismatch

Twenty existing hard-label A0 inner teachers were provenance-checked and reused;
query patients are excluded from teacher training AND selection. Preprocessing
is teacher-training-only. Fresh deterministic eval-mode NEZ logits, not MC-averaged
scores, train the FIT residual. VAL uses full-FIT frozen A0. In-sample FIT scores
are diagnostic only; no score-distribution correction is fitted.

| Fold | FIT OOF mean / std | FIT in-sample mean / std | VAL A0 mean / std |
|---|---:|---:|---:|
| 1 | 0.290593 / 1.107031 | 0.139327 / 0.874027 | 0.140654 / 0.703634 |
| 2 | 0.123575 / 0.836427 | 0.180522 / 0.823774 | 0.164640 / 0.783934 |
| 3 | 0.137691 / 0.897875 | 0.067608 / 0.798870 | 0.170970 / 0.822845 |
| 4 | 0.191317 / 0.797950 | 0.149823 / 0.864738 | 0.141669 / 0.833570 |
| 5 | 0.158261 / 0.911298 | 0.103783 / 0.677474 | 0.109568 / 0.681420 |

OOF scores are notably broader than VAL in folds1/5, a residual-transfer risk.
Because FIT/VAL also contain different patients, these summaries cannot separate
teacher shift from cohort shift or prove the cause of failure. Full quantiles,
absolute margins and class-conditional summaries:
[A0_OOF_DISTRIBUTION_AUDIT.csv](results/A0_OOF_DISTRIBUTION_AUDIT.csv).

## Residual, threshold and error diagnostics

At each fold's fixed original A0 threshold, T1 Macro-F1=0.629826 and T2=0.635594,
below T0=0.638080. The small gain at each arm's own selected threshold therefore
does not persist with fixed A0 thresholds. Rankings change slightly, but T2's
AUROC gain over T0 is only+0.000460 and its AP advantage over T1 has a CI crossing
zero. There is no convincing incremental temporal-interaction value.

| Patient-equal residual statistic | T1 | T2 |
|---|---:|---:|
| Mean NEZ residual | -0.044027 | -0.036437 |
| Mean within-patient std | 0.051673 | 0.034298 |
| Mean absolute residual | 0.097777 | 0.060459 |
| Mean patient maximum absolute residual | 0.153048 | 0.108441 |
| Mean patient saturation fraction | 0.007485 | 0.000000 |
| Mean constant-shift energy fraction | 0.502938 | 0.516322 |
| Mean defined residual/A0-logit correlation | 0.573853 | 0.588171 |
| Mean channel pair-order reversal fraction | 0.018411 | 0.014881 |

Residuals are not purely constant, but approximately half the mean patient-wise
energy is the constant component and only1.5-1.8% of pair orderings reverse.
Correlation excludes undefined constant cases. The displayed maximum is a mean
of patient maxima, not a global maximum. Gate saturation uses channel-appearance
weighting, not the displayed patient average; T2's channel fraction is zero.

Own-threshold pooled repeated channel counts: T1 corrects94/spoils92 (net+2);
T2 corrects54/spoils93 (net-39). T2 gains32/loses17 EZ TPs, but adds76/removes22
FPs. At fixed A0 thresholds T2 corrects45/spoils124 (net-79), gains36/loses11 EZ
TPs and adds113/removes9 FPs. These explain specificity/accuracy cost, but are
not the patient-equal F1 statistic. See [DECISION_CHANGE_AUDIT.csv](results/DECISION_CHANGE_AUDIT.csv),
[RESIDUAL_DIAGNOSTICS.csv](results/RESIDUAL_DIAGNOSTICS.csv), and
[FIXED_T0_THRESHOLD_DIAGNOSTIC.csv](results/FIXED_T0_THRESHOLD_DIAGNOSTIC.csv).
Optional time permutation was not run; no onset-specific causal claim is justified.

## Execution, isolation and fifteen explicit answers

Ten formal server runs completed in one orchestration attempt, with113 student
epochs under original early stopping, no native crash, discarded epoch or
outcome-driven restart. Finalization/bootstrap completed in one attempt. Both
stderr files are empty. Code/protocol/input/checkpoint/selected-metric integrity
passed. Before aggregate outcomes were inspected, the unexecuted finalizer's
saturation denominator was corrected to the frozen channel-appearance definition;
the actually executed finalizer is separately hash-bound. Trainer/model/protocol
did not change.

The source cache and prior feature export contain original clinical labels and
were materialized before sealing FIT/VAL. "No outer evaluation" does not mean
outer labels were never present in memory. No outer label entered fitting,
normalization, selection or metric evaluation; there are no new outer predictions.
Temporal relative references use only unlabeled simultaneous valid features.
No ID/center/resection/outcome/true-K enters model features. Public outputs are
source, aggregates and hashes, not tensors, identifiable records, scores or logs.

1. A0 metrics/thresholds/architecture reproduce; ledger drift is <=2e-7 and current
   epoch0 identity exact.
2. Valid-window masking leaves full required unique-channel coverage; timing is
   annotation-confirmed, not independent EDF-confirmed.
3. All7,635 channels/256 seizures remain;25,267 possible record-channel slots,
   272 missing incidences. Early feasibility100% at unique-channel level.
4. T1 numerically gains+0.003119 F1; cluster CI crosses zero.
5. T2 numerically gains+0.002690 F1; cluster CI crosses zero.
6. T2 does not beat T1: -0.000429 F1, CI crossing zero.
7. Parameter counts, initialization and admitted input tensors are identical.
8. Official unmasked parity/masked exclusion/all-missing tests passed.
9. Small AP/ranking changes exist; extra CoTAR ranking benefit is unestablished.
10. T2 beats T0 in4/5 folds and numerically all centers, T1 in only1/5 folds.
11. T2 sensitivity/AP rise while specificity falls; this is not cost-free.
12. Some rankings change, but a large constant-shift component persists.
13. OOF/VAL distributions differ; mismatch is documented, not causally established.
14. The +0.03-0.04 target and minimum continuation gate both fail.
15. Expansion of CoTAR or opening TEST is not justified by this run. Stop this
    version; no automatic interval tuning, T3 or rescue model is initiated.

Completion: [RUN_STATUS.json](results/RUN_STATUS.json). Source, protocol and compact
findings are published together; all private data/checkpoints/OOF logits/logs
remain on the trusted server.
