# PR-UAS seed42: completed development, continuation gate failed

Terminal: `A2_POSITIVE_BUT_BELOW_TARGET`. This denotes a tiny numerical positive
Macro-F1 difference, **not statistical support or a useful improvement**.
The specified +0.030–0.040 target was not reached. No new outer TEST evaluation
was admitted or run. No lambda, threshold policy, loss or architecture was tuned
after outcomes.

## Completed experiment and metric unit

All 20 OOF teachers and 15 matched students completed on the trusted server.
The frozen original cohort contains 80 patients and 7,635 unique patient-channel
records. Development validation contains 65 patient-fold appearances per arm
(13 per fold), representing **47 distinct patient IDs**, not 65 independent
patients. Reported metrics average patient metrics over these original validation
cells. Repeated appearances are kept together in patient-ID bootstrap clusters.
There are 6,273 matched validation channel appearances, not 6,273 unique cohort
channels. Validation selects checkpoints and thresholds, so these results are
development results and are selection-optimistic, not held-out outer performance.

### All development metrics

| Patient-equal metric | A0 hard labels | A1 smoothing .10 | A2 OOF uncertainty |
|---|---:|---:|---:|
| Macro-F1 | 0.638080 | 0.637288 | 0.638229 |
| EZ-F1 | 0.431198 | 0.426363 | 0.428154 |
| NEZ-F1 | 0.844961 | 0.848213 | 0.848305 |
| Balanced accuracy | 0.665436 | 0.669135 | 0.664470 |
| EZ-AUPRC | 0.518235 | 0.516857 | 0.519669 |
| EZ-AUROC | 0.710667 | 0.707643 | 0.710264 |
| EZ-MRR | 0.686550 | 0.711898 | 0.696541 |
| Top1-is-EZ | 0.584615 | 0.615385 | 0.600000 |
| Sensitivity (EZ) | 0.465904 | 0.466817 | 0.456879 |
| Specificity (NEZ) | 0.864968 | 0.871453 | 0.872062 |
| Accuracy | 0.773174 | 0.775459 | 0.776732 |

Full precision: [VALIDATION_SUMMARY.csv](results/VALIDATION_SUMMARY.csv).

### Paired differences and 95% percentile confidence intervals

10,000 paired patient-ID cluster bootstrap draws, seed42; each sampled ID carries
all its validation appearances and all three matched arms. Numeric thresholds
remain the original selected thresholds; no bootstrap re-optimization.

| Comparison | Metric | Mean difference | 95% CI |
|---|---|---:|---|
| A1 − A0 | Macro-F1 | -0.000792 | [-0.009945, 0.008819] |
| A2 − A0 | Macro-F1 | +0.000149 | [-0.004712, 0.004356] |
| A2 − A1 | Macro-F1 | +0.000941 | [-0.007933, 0.009206] |
| A1 − A0 | EZ-F1 | -0.004835 | [-0.018765, 0.009043] |
| A2 − A0 | EZ-F1 | -0.003045 | [-0.009770, 0.002739] |
| A2 − A1 | EZ-F1 | +0.001791 | [-0.010823, 0.014472] |
| A1 − A0 | EZ-AUPRC | -0.001378 | [-0.011459, 0.009505] |
| A2 − A0 | EZ-AUPRC | +0.001434 | [-0.004523, 0.008353] |
| A2 − A1 | EZ-AUPRC | +0.002812 | [-0.003620, 0.009493] |
| A1 − A0 | EZ-AUROC | -0.003024 | [-0.013296, 0.005664] |
| A2 − A0 | EZ-AUROC | -0.000403 | [-0.005050, 0.004413] |
| A2 − A1 | EZ-AUROC | +0.002621 | [-0.004043, 0.011593] |

All secondary contrasts/CIs: [PAIRED_BOOTSTRAP.csv](results/PAIRED_BOOTSTRAP.csv).
These CIs describe the matched development comparison; they do not remove
checkpoint/threshold selection optimism or establish external generalization.
A1 does not improve primary F1 over A0. A2 is numerically above both controls in
primary F1, but the difference is negligible and neither contrast excludes zero.

## Exact B0 reproduction and controls

Historical frozen B0 inference reproduced: Macro-F1 0.6161672347405565,
EZ-F1 0.4238657223879644, EZ-AUPRC 0.5319959936611567 and EZ-AUROC
0.7268443834107704. Frozen decisions and thresholds match; maximum probability
drift across five checkpoints is below 1.49e-7. The original 88D feature order,
labels, identities and five folds were verified using a fresh reconstruction.
See [A0_REPRODUCTION.json](audit/A0_REPRODUCTION.json) and
[RUNTIME_ADMISSION.json](audit/RUNTIME_ADMISSION.json).

The old wrapper instantiated its MLP before reseeding optimization and did not
archive its initial weights. Therefore historical *training-from-scratch bitwise*
reproduction is not claimed. Historical checkpoint inference is reproduced.
New matched A0/A1/A2 have explicitly seeded identical initial states within each
fold, the exact 8,817-parameter PR-MLP, original preprocessing, weighted BCE,
optimizer, patient update/order policy and selection rules. Their sole scientific
training difference is the target construction. The historical outer metric
0.616167 must **not** be subtracted from these selected development metrics.

NEZ remains label 1 and the model outputs a NEZ logit; ranking/localization metrics
use EZ probability `sigmoid(-logit_nez)`. A1 epsilon is .10. A2 uses four FIT-only
OOF teachers per fold, ten Dropout-only MC passes, frozen q/u, and the specified
lambda schedule (zero through epoch5, .05–.25 at epochs6–10, then .25).
Teacher query, TRAIN and internal selection roles are disjoint; teacher scalers
use teacher TRAIN only. Student scalers use outer FIT only. No outer VAL/TEST
labels enter teacher construction. See [OOF_TEACHER_AUDIT.json](audit/OOF_TEACHER_AUDIT.json).

## Fold comparisons and why the gain is almost zero

| Fold | A0 Macro-F1 | A1 Macro-F1 | A2 Macro-F1 | A1−A0 | A2−A0 | A2−A1 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | .646844 | .642924 | .646844 | -.003919 | .000000 | +.003919 |
| 2 | .657246 | .656213 | .657246 | -.001033 | .000000 | +.001033 |
| 3 | .642299 | .645202 | .642299 | +.002903 | .000000 | -.002903 |
| 4 | .640909 | .642705 | .640909 | +.001797 | .000000 | -.001797 |
| 5 | .603102 | .599396 | .603849 | -.003706 | +.000747 | +.004453 |

Full fold metrics: [VALIDATION_BY_FOLD.csv](results/VALIDATION_BY_FOLD.csv).
A0 selected epochs are 3/1/2/2/2; A1 3/1/2/6/7; A2 3/1/2/2/6.
**Four of five selected A2 checkpoints precede any nonzero correction and recover
A0 validation predictions exactly.** Only fold5 selects a corrected model,
at epoch6 with lambda=.05, rather than the maximum .25. This is a failure of the
specified schedule/selection combination to retain a useful corrected model,
not evidence that a stronger, retrospectively adjusted schedule would succeed.
The original schedule and early stopping were not revised.

The A0 selected NEZ thresholds are .465/.275/.340/.355/.445, A1
.435/.265/.335/.340/.325, and A2 .465/.275/.340/.355/.345.
All are selected by the original validation grid and frozen with their models.

## Ordering versus operating points

A2 EZ-AUROC slightly declines (-.000403), while EZ-AUPRC changes by +.001434;
both intervals span zero. There is no convincing ordering improvement.
Across repeated channel appearances, A2 corrects 60 A0 errors but spoils 25
correct decisions (net +35); all changes occur in fold5. At the A0 numeric
threshold instead, its net correct-count change is **-37**. Thus the selected
operating point matters substantially; the small F1 change is not evidence of
robust ranking improvement.

Near the A0 boundary (NEZ probability distance <=.05), 43 errors are corrected
and 22 correct decisions spoiled; at distance .05–.15, the counts are 17 and 3;
none change beyond .15. Mean absolute probability change is .010824 overall
and .053252 in fold5. These are retrospective diagnostics, not selection criteria.

Pooled EZ-positive confusion counts over repeated appearances are A0
TP=527/FP=765/TN=4304/FN=677, A1 523/734/4335/681, and A2
514/717/4352/690. A2 reduces FP by 48 but adds 13 FN relative to A0:
specificity/accuracy increase while sensitivity/EZ-F1 decline. These pooled
counts must not be confused with the patient-equal metric averages above.
See [DECISION_CHANGE_AUDIT.csv](results/DECISION_CHANGE_AUDIT.csv).

## Center consistency

| Center | A0 Macro-F1 | A1 Macro-F1 | A2 Macro-F1 | A2−A0 |
|---|---:|---:|---:|---:|
| HUP | .669631 | .671282 | .673277 | +.003646 |
| LZU | .590312 | .588489 | .582875 | -.007437 |
| multicenter | .740741 | .743381 | .740741 | .000000 |
| pediatric | .526085 | .517650 | .527062 | +.000977 |

The effect is not consistent across centers. Full center metrics, counts and
evaluation units are in [VALIDATION_BY_CENTER.csv](results/VALIDATION_BY_CENTER.csv).

## What uncertainty and magnitude diagnostics do—and do not—show

Fold mean normalized predictive entropy is approximately .838–.898; the fraction
with u>=.8 is .684–.837. Observed-label mismatch is .332–.354 and entropy AUROC
for that mismatch is only .576–.626. Entropy is a deterministic function of q
and mostly describes teacher confidence. These labels were not independently
adjudicated: mismatch is **not** verified clinical annotation error. High entropy
cannot establish that the corresponding EZ/SOZ label is wrong, and confident
teacher errors remain possible. MC variance, expected entropy and MI were
diagnostics only, not extra objectives.

At maximum lambda=.25, mean absolute A2 target movement is .09146–.10071,
versus A1's .05. At selected A2 epochs it is zero in folds1–4 and .02014 in
fold5. The magnitude-matched constant-smoothing diagnostic derives epsilon as
twice mean absolute A2 movement, giving about .183–.201 and exactly matching
average movement in **target space**. No fourth model was trained or tuned.
Consequently, a performance comparison controlling smoothing magnitude was
**not performed**; this is a limitation, not evidence of an uncertainty-specific
causal advantage. The primary mandated fixed-epsilon A1 comparison is complete.
See [UNCERTAINTY_SUMMARY.csv](results/UNCERTAINTY_SUMMARY.csv) and
[SOFT_LABEL_DIAGNOSTICS.csv](results/SOFT_LABEL_DIAGNOSTICS.csv).

## Prespecified gate and stop decision

| Condition | Observation | Pass |
|---|---|---|
| A2−A0 Macro-F1 >=+.015 | +.000149 | no |
| A2−A1 Macro-F1 >0 | +.000941 | yes |
| EZ-AUPRC does not decline | +.001434 | yes |
| EZ-F1 does not materially worsen | -.003045 | no under frozen conservative zero tolerance |
| >=4/5 folds improve | 1/5 | no |
| Primary target >=+.030 | +.000149 | no |

The prompt did not quantify “material” EZ-F1 deterioration; a conservative zero
tolerance was locked before outcomes. Even a relaxed EZ-F1 tolerance would not
rescue the large primary-gain and fold-consistency failures. See
[VALIDATION_GATE.json](results/VALIDATION_GATE.json) and [RUN_STATUS.json](results/RUN_STATUS.json).

**Conclusion:** this fixed seed42 experiment does not support a meaningful
uncertainty-correction benefit over hard labels or ordinary smoothing. The
numerical terminal is `A2_POSITIVE_BUT_BELOW_TARGET`; it must not be presented as
scientific success. Stop here, without more losses, lambda tuning or outer TEST.

## Execution integrity and delivery

Synthetic tests and the actual FIT-only smoke passed before formal training.
Engineering changes are disclosed in [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md):
isolated runtime admission, exact frozen B0 replay, trusted feature hash checks,
and independently validated equivalent threshold evaluation. The old unstable
legacy metric loop did not successfully execute exhaustively on the server;
its eight synthetic cases were exhaustively evaluated locally and input/optimum
hash-checked on the server. Real selected-threshold metrics and one-epoch
original-source optimization parity were checked.

The successful training process saved all 35 completed cells, then raised a
native access violation during interpreter shutdown. A fresh independent
verification passed all last/best checkpoint hashes, model finiteness,
optimizer/RNG/history completeness, OOF identity/exclusion/scaler checks and
matched validation predictions. Aggregation/bootstrap then completed without
retraining. The native root cause remains unestablished; no hardware-fault claim
or silent discarded epoch is made. See
[ARTIFACT_INTEGRITY_AUDIT.json](audit/ARTIFACT_INTEGRITY_AUDIT.json).

Private features, OOF predictions, scalers, model/optimizer/RNG checkpoints and
logs remain at `C:\pr_uncertainty_aware_supervision_seed42_runtime` on the trusted
server. The repository contains only source and compact aggregate audits/results.
Historical outer outcomes were already inspected; any future evaluation on
those patients would remain exploratory repeated-test, not fresh confirmation.
