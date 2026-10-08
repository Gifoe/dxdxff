# A0 + Patient-Local Label Reliability: completed seed42 development experiment

**Conclusion: PLLR is feasible to construct but does not provide a convincing
predictive benefit. Stop after development validation.** B2 improves patient
Macro-F1 by only **0.000144**, versus the required +0.020 continuation gain and
the desired +0.030–0.040. The paired 95% bootstrap interval spans zero. EZ-F1,
EZ-AUPRC and sensitivity decline. Five of six mandatory continuation gates fail.

Machine terminal: `PLLR_POSITIVE_BELOW_TARGET`. This predeclared mapping denotes
a positive numerical mean, **not a statistically supported or useful gain**.
Execution status: `COMPLETE_DEVELOPMENT_GATE_FAILED`. No new B1/B2 outer-test
evaluation was run; no hyperparameter, threshold rule or weight construction was
changed after observing outcomes.

## Scope and reproduction

The original cohort remains 80 patients / 7,635 unique patient-channel records,
with the original frozen five FIT/VAL/TEST splits and 88D feature order. All
three arms use the same 8,817-parameter PR-MLP, hard labels, initialization within
fold, patient-order policy, optimizer and checkpoint/threshold selection.
NEZ=1, EZ=0; logits score NEZ, and EZ ranking uses the complementary probability.
PCA8 is FIT-only geometry for offline weights, never a classifier input.

Fresh historical B0 checkpoint replay passed on all five folds: maximum score
drift <1.49e-7; historical Macro-F1 **0.616167**, EZ-F1 **0.423866**, EZ-AUPRC
**0.531996**, EZ-AUROC **0.726844**. This is the requested historical outer
inference reproduction, not a new trained-model test. Fresh matched A0 training
reproduced prior PR-UAS development Macro-F1 **0.638080**, with identical selected
epochs, numerical thresholds, IDs/labels and initialization. Maximum development
probability drift was <2.98e-8. Historical initialization was not archived, so
historical training bitwise equivalence is not asserted.

Development results below comprise **65 patient-fold validation appearances,
47 unique patient IDs and 6,273 channel appearances per arm**. Metrics average
the original patient-fold cells equally; repeated appearances are retained.
All five folds have 13 validation patients. These are validation-selected
development results, not independent test results. Do not subtract the historical
outer 0.616167 from development 0.638080.

## All development metrics and contrasts

| Patient-equal metric | A0 | B1 permuted | B2 PLLR | B1−A0 | B2−A0 | B2−B1 |
|---|---:|---:|---:|---:|---:|---:|
| Macro-F1 | .638080 | .637447 | .638224 | −.000632 | +.000144 | +.000777 |
| EZ-F1 | .431198 | .431203 | .423335 | +.000005 | −.007863 | −.007867 |
| NEZ-F1 | .844961 | .843692 | .853112 | −.001269 | +.008151 | +.009420 |
| Balanced accuracy | .665436 | .670782 | .660092 | +.005346 | −.005344 | −.010690 |
| EZ-AUPRC | .518235 | .520138 | .515109 | +.001903 | −.003126 | −.005029 |
| EZ-AUROC | .710667 | .709253 | .714029 | −.001414 | +.003362 | +.004776 |
| EZ-MRR | .686550 | .711227 | .711910 | +.024677 | +.025360 | +.000683 |
| Top1-is-EZ | .584615 | .615385 | .615385 | +.030769 | +.030769 | .000000 |
| EZ sensitivity | .465904 | .481951 | .435435 | +.016047 | −.030469 | −.046516 |
| NEZ specificity | .864968 | .859613 | .884749 | −.005355 | +.019780 | +.025136 |
| Accuracy | .773174 | .772057 | .782768 | −.001118 | +.009594 | +.010711 |

Full precision: [VALIDATION_SUMMARY.csv](results/VALIDATION_SUMMARY.csv).
All 33 metric/contrast confidence intervals: [PAIRED_BOOTSTRAP.csv](results/PAIRED_BOOTSTRAP.csv).

### Paired patient-ID cluster bootstrap

10,000 draws, seed42; resample the **47 unique patient IDs**, retaining all their
validation-fold appearances. Checkpoints and numeric thresholds stay fixed in
every draw. Percentile 95% CIs; no multiplicity adjustment. These intervals do
not undo optimism from development checkpoint/threshold selection.

| Contrast | Macro-F1 delta [95% CI] | EZ-F1 delta [95% CI] | EZ-AUPRC delta [95% CI] | EZ-AUROC delta [95% CI] |
|---|---|---|---|---|
| B1−A0 | −.000632 [−.010376, .009567] | +.000005 [−.015931, .016898] | +.001903 [−.006768, .012154] | −.001414 [−.007101, .004710] |
| B2−A0 | +.000144 [−.011930, .011775] | −.007863 [−.029053, .012406] | −.003126 [−.012590, .006252] | +.003362 [−.004071, .011351] |
| B2−B1 | +.000777 [−.012152, .014333] | −.007867 [−.031660, .015838] | −.005029 [−.015989, .003613] | +.004776 [−.002138, .011813] |

B2−A0 sensitivity delta is **−.030469 [−.051378, −.011356]**, whereas specificity
delta is **+.019780 [.008806, .033409]**. The development evidence indicates a
shift toward finding fewer EZ channels, not improved EZ sensitivity. Accuracy
and NEZ-F1 gains must not be presented as successful localization.

## Five folds and all available centers

| Fold | A0 Macro-F1 | B1 Macro-F1 | B2 Macro-F1 | B2−A0 |
|---|---:|---:|---:|---:|
| 1 | .646844 | .646672 | .644792 | −.002052 |
| 2 | .657246 | .655293 | .663350 | +.006105 |
| 3 | .642299 | .644836 | .635899 | −.006400 |
| 4 | .640909 | .639260 | .637850 | −.003059 |
| 5 | .603102 | .601176 | .609229 | +.006127 |

B2 improves only **2/5** folds. Selected epochs A0: 3/1/2/2/2; B1: 3/1/2/1/7;
B2: 4/2/5/1/4. Selected NEZ thresholds A0: .465/.275/.340/.355/.445;
B1: .465/.265/.340/.475/.350; B2: .490/.245/.305/.300/.390.
All secondary fold metrics are in [VALIDATION_BY_FOLD.csv](results/VALIDATION_BY_FOLD.csv).

| Center | Validation appearances / unique IDs | A0 F1 | B1 F1 | B2 F1 | B2−A0 |
|---|---:|---:|---:|---:|---:|
| HUP | 28 / 21 | .669631 | .678923 | .674616 | +.004986 |
| LZU | 14 / 11 | .590312 | .591915 | .589644 | −.000668 |
| multicenter | 11 / 9 | .740741 | .722937 | .716512 | −.024229 |
| pediatric | 12 / 6 | .526085 | .515427 | .538221 | +.012136 |

Center effects are inconsistent. The A0 worst center, pediatric, improves, so the
conservative worst-center nondecline gate passes. Multicenter deteriorates by
.024229. Small center samples do not establish reliable subgroup benefit.
All center metrics: [VALIDATION_BY_CENTER.csv](results/VALIDATION_BY_CENTER.csv).

## FIT feasibility, weight variation and control strength

Feasibility passed **before training**: eligible patient counts by fold are
45/51, 49/51, 46/50, 45/52 and 47/51 (86.5%–96.1%). Across repeated FIT appearances,
232/255 patient groups and 22,625/24,267 channel appearances support prototypes.
The 23 unsupported patient-fold groups retain all-one weights. Per-center
eligibility and stability are in [FEASIBILITY_REPORT.md](FEASIBILITY_REPORT.md).

Median patient mean bootstrap rank correlations are .729446/.764763/.752262/
.782697/.755929, above the pre-outcome .5 feasibility cutoff. Mean absolute
resampled weight changes are .0324–.0382. About 8.2%–15.6% of eligible groups have
mean correlation <.5. These are moderately stable geometric rankings, not
verified label-quality estimates; passing feasibility does not prove benefit.

Final weights range **.514117–1.362857**. Float64 patient/class coefficient-sum
error is <=5.69e-14; class means remain 1 without clipping. Sum of within-class
effective sample sizes is 23,951.22 / 24,267 (98.7%). GPU training uses float32.
Preserving coefficient sums is not equality of realized weighted losses or
gradients, which depend on feature/weight/loss correlations.

Across repeated FIT channels, raw downweighting occurs for **38.57% of EZ versus
16.81% of NEZ**; after class normalization, **32.84% versus 14.77%**. The rule has
no rarity term, but symmetry and class normalization do not rule out suppression
of difficult minority channels. The sensitivity decline is compatible with this
concern; it does not identify confirmed annotation errors or prove causality.

B1 preserves the exact B2 multiset within each patient/observed class. Every
nonconstant group changes some correspondence, but many weights are equal;
mean changed-channel fraction across patient groups is only **.333151**. The
control is therefore partial correspondence disruption, not a permutation that
changes every channel. All constant groups remain unchanged. This limitation
was reported without post-outcome modification.

## Ranking, thresholds, score shifts and corrected decisions

The small B2 AUROC increase (+.003362, CI spanning zero) is accompanied by lower
AP (−.003126). MRR/Top1 improve similarly under B1 and B2. There is no convincing
evidence of a general feature-informed EZ-ranking improvement over permutation.

Apply each fold's frozen **A0 numeric threshold** to each arm's already selected
checkpoint, without selecting anything again:

| Arm | Macro-F1 at A0 threshold | Difference from A0 |
|---|---:|---:|
| A0 | .638080 | .000000 |
| B1 | .628360 | −.009720 |
| B2 | .622932 | −.015148 |

Thus B2's near-zero primary gain does not survive this fixed-threshold diagnostic.
It is not an alternative benchmark or permission to optimize another threshold.
Patient-equal mean NEZ score shifts are +.011622 (B1) and +.015547 (B2), with mean
absolute score changes .024563 and .046379. Fold shifts vary; these summaries
do not establish a uniform calibration shift.

| Arm and threshold | A0 errors corrected | Correct A0 decisions spoiled | Net correction |
|---|---:|---:|---:|
| B1 own selected threshold | 118 | 113 | +5 |
| B1 frozen A0 threshold | 71 | 98 | −27 |
| B2 own selected threshold | 172 | 87 | +85 |
| B2 frozen A0 threshold | 105 | 169 | −64 |

These are pooled counts over 6,273 repeated validation-channel appearances,
not patient-equal metric changes. Own-threshold B2 reduces false positives while
losing true positives: EZ-positive confusion A0 TP/FP/TN/FN = 527/765/4304/677;
B2 = 494/647/4422/710. This is **118 fewer FP and 33 more FN**. Net correction or
accuracy alone conceals the sensitivity cost.

Descriptive A0 performance bins (predeclared .5 and .65 boundaries): B2 F1 deltas
are +.005014 in the <=.5 bin, +.007259 in (.5,.65], and −.007837 above .65. B1's
poor-bin gain is larger (+.009745). B2 AP/AUROC deltas in the poor bin are
+.001714/+.018389. These are retrospective, A0-outcome-conditioned diagnostics,
subject to regression to the mean, and were not used to tune weights.
Patients can appear in different bins in different folds; bin ID counts overlap.

## Continuation decision

| Mandatory development condition | Observed | Decision |
|---|---:|---|
| B2−A0 Macro-F1 >= +.020 | +.000144 | FAIL |
| B2−B1 Macro-F1 >= +.010 | +.000777 | FAIL |
| EZ-AUPRC nondecline | −.003126 | FAIL |
| EZ-F1 nondecline | −.007863 | FAIL |
| >=4/5 improved folds | 2/5 | FAIL |
| Worst-center nondecline | +.012136 | PASS |

Condition 7 also provides no robust ranking/fixed-threshold support. Desired
+.030–.040 gain is not reached. Failure is principally **lack of predictive
benefit**, not insufficient eligibility or a failed prototype-stability gate.
Do not turn these geometry weights into a claim of clinical label correction.

## Answers to the twelve task questions

1. **Reproduction?** Yes: fresh historical B0 inference and fresh matched A0
   development training passed the required score/selection checks.
2. **Sufficient support?** 45/49/46/45/47 eligible FIT patients by fold;
   232/255 repeated patient-fold groups, not 232 unique patients.
3. **Stable enough?** Yes under the frozen operational gate, median correlations
   .729–.783, with a material unstable minority. Not clinical reliability proof.
4. **Weight variation?** Final .514–1.363, class means 1; ESS 98.7% and more
   frequent EZ downweighting. Detailed center/class dispersions are in the CSV.
5. **B1 vs A0?** F1 −.000632, CI crosses zero; no convincing F1 change.
6. **B2 vs B1?** F1 +.000777, CI crosses zero and far below +.010; not supported.
7. **B2 vs A0?** F1 +.000144, CI crosses zero and far below +.020; not supported.
8. **Ranking or threshold?** AP declines; tiny AUROC gain is uncertain. At A0's
   fixed numerical thresholds B2 F1 declines .015148. No robust ranking claim.
9. **Sensitivity without excessive FP?** No. FP fall, but EZ sensitivity falls
   .030469 with a development bootstrap CI below zero.
10. **Consistent centers?** No: HUP/pediatric improve, LZU/multicenter worsen.
11. **Target reached?** No; +.000144 is effectively unchanged relative to +.030.
12. **Failure mechanism?** Feasible, moderately stable weighting without useful
    predictive increment. Clinical noise causation cannot be determined here.

## Execution integrity and limitations

All 15 student runs completed: A0/B1/B2 respectively 40/44/46 completed epochs,
130 total, with identical fold initialization and all epoch histories/checkpoints
retained privately. Formal training had no retry or discarded completed epoch.
Six synthetic test groups and real FIT-only all-one-gradient/training parity
checks passed. See audit/ and [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md).

A native Pandas access violation interrupted the first statistics-only finalizer
after five aggregate files had been written. Its logs/files were preserved; an
exact-command retry completed. All five pre-existing aggregates, including the
bootstrap, are byte-identical on retry. No retraining, new model inference,
prediction change or tuning occurred. The native fault's root cause is unknown.

The original all-cohort export contains outer labels and was materialized solely
for the mandatory historical replay and development sealing. Formal geometry
and training consumed sealed FIT/VAL banks without outer channels. This is not
a claim that outer labels have never been viewed. Validation checkpoint/threshold
selection and historically viewed outcomes preclude independent confirmation.
Only one seed and one fixed method were tested; no clinical label adjudication
was available. No new outer-test evaluation follows this failed gate.

Public artifacts contain only source and compact aggregate audits/results.
Features, patient/channel identities or predictions, PCA/scalers, weights,
checkpoints and runtime logs remain on the trusted server.
