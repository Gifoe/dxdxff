# PURN: user-authorized frozen four-arm outer comparison complete

| Model | Params | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| B0 reused original A0 | 8,817 | 0.739768 | 0.537887 | 0.624919 | 0.419405 | 0.728620 |
| B1 RMSNorm-88 | 8,817 | 0.726821 | 0.527317 | 0.618997 | 0.418595 | 0.693256 |
| B2 Unique-LayerNorm | 8,817 | 0.739365 | 0.539176 | 0.625620 | 0.416186 | 0.752043 |
| B3 PURN | 8,817 | 0.728407 | 0.528789 | 0.615286 | 0.409093 | 0.686701 |

**PURN does not improve outer localization over A0.** Its AUROC difference is
−0.011361,95% descriptive patient-cluster CI[−0.023761,−0.001255]. AP, F1,
MRR and Top1 point estimates also decline. There is no supported PURN-specific
gain over ordinary RMS88, and it trails Unique-LN. Retain original A0; do not
rescue train, reselect thresholds, alter epsilon or automatically produce v2.

Status: `ONE_FROZEN_PURN_FOUR_ARM_OUTER_PASS_COMPLETE`.
Original development terminal/gate remains `PURN_DEVELOPMENT_GATE_FAILED`.

## Explicit authorization and estimand

The original attached task prohibited new outer evaluation. Its development
study completed and failed the fixed gate. Only after that notification did the
user explicitly request "把outer跑完". OUTER_PROTOCOL_AMENDMENT.json separately
authorizes this frozen extension for the same2x2 comparison, without changing
the original protocol, gate, model, preprocessing, optimizer or selections.

Exactly15 new inference cells(B1/B2/B3×5 folds) and5 reused B0 score files,
not new training or fresh B0 inference. All arms use the selected development
checkpoint and own fixed numeric threshold documented in MODEL_SELECTION.csv.
No patient, center, channel stratum, model or operating point is selected from
outer outcomes. This is **user-authorized post-development exploratory
repeated-outer**, not independent, external or blind confirmation. These80
patients have historical viewing in prior experiments.

Primary: equal-patient metrics over80 unique outer patients, each exactly once,
7,635 canonical channels,1,743 EZ/5,892 NEZ. Per-fold sizes16/16/17/15/16 are
not averaged equally; overall means weight patients equally. Ranking uses
ascending NEZ probabilities with stable canonical channel-string ties. EZ=0,
NEZ=1, predict EZ iff score<that model/fold's locked threshold. Patient Z uses
unlabeled channels within a patient; original stored FIT-only imputer/scaler
remains unchanged. Neither test-label normalization nor new scaler fitting.

## All primary outer metrics

| Metric | B0 | B1 | B2 | B3 |
|---|---:|---:|---:|---:|
| AUROC | 0.739768 | 0.726821 | 0.739365 | 0.728407 |
| AUPRC | 0.537887 | 0.527317 | 0.539176 | 0.528789 |
| Macro-F1 | 0.624919 | 0.618997 | 0.625620 | 0.615286 |
| EZ-F1 | 0.419405 | 0.418595 | 0.416186 | 0.409093 |
| NEZ-F1 | 0.830434 | 0.819399 | 0.835054 | 0.821478 |
| MRR | 0.728620 | 0.693256 | 0.752043 | 0.686701 |
| Top1-EZ | 0.637500 | 0.600000 | 0.675000 | 0.587500 |
| Balanced accuracy | 0.666405 | 0.659600 | 0.665154 | 0.657216 |
| Sensitivity | 0.471033 | 0.476650 | 0.457340 | 0.467236 |
| Specificity | 0.861776 | 0.842551 | 0.872968 | 0.847195 |
| Accuracy | 0.758773 | 0.748033 | 0.763400 | 0.750731 |
| Predicted EZ fraction | 0.207411 | 0.222665 | 0.196200 | 0.218473 |

B0 has highest AUROC and EZ-F1. B2 has the highest AP/Macro-F1/MRR/Top1 point
estimates but no supported AUROC/AP/F1/MRR increment over B0. It changes the
operating-point balance: sensitivity declines0.013693[−0.026961,−0.001479],
specificity rises0.011191[0.004220,0.018957]. It remains a prespecified control,
not a retrospectively selected replacement or new experiment.

## Paired patient-cluster bootstrap

10,000 common seed42 draws,80 unique ID clusters with replacement, all channels
kept together, numeric thresholds/checkpoints fixed. Percentile95% intervals
are descriptive and unadjusted for multiple metrics/previous cohort use.

| Contrast | ΔAUROC[CI] | ΔAUPRC[CI] | ΔMacro-F1[CI] | ΔEZ-F1[CI] | ΔMRR[CI] |
|---|---|---|---|---|---|
| B3−B0 | −0.011361[−0.023761,−0.001255] | −0.009098[−0.022600,0.004287] | −0.009634[−0.019680,0.000026] | −0.010312[−0.025314,0.004589] | −0.041918[−0.091474,0.003479] |
| B3−B1 | +0.001586[−0.001717,0.005717] | +0.001472[−0.005044,0.009573] | −0.003711[−0.011531,0.004077] | −0.009502[−0.022383,0.002786] | −0.006554[−0.038165,0.024097] |
| B3−B2 | −0.010958[−0.021399,−0.001291] | −0.010387[−0.023407,0.003235] | −0.010334[−0.020140,−0.000786] | −0.007093[−0.022589,0.008052] | −0.065342[−0.122290,−0.014339] |
| B2−B0 | −0.000404[−0.008153,0.007300] | +0.001289[−0.005920,0.008268] | +0.000700[−0.006793,0.008270] | −0.003219[−0.014954,0.008492] | +0.023424[−0.005756,0.057696] |
| B1−B0 | −0.012947[−0.025732,−0.002549] | −0.010570[−0.023387,0.002356] | −0.005922[−0.017019,0.004678] | −0.000810[−0.017309,0.015292] | −0.035364[−0.088688,0.015885] |

PURN's AUROC interval is below zero against A0 and Unique-LN, while its very
small point advantage over RMS88 has an interval spanning zero. Its MF1 interval
against A0 narrowly crosses zero; do not call it wholly negative by rounding
the upper bound to0. Against B2, MF1/MRR intervals are negative. Full80 metric
effects, confusion counts and sensitivity/specificity intervals are in
OUTER_PAIRED_BOOTSTRAP.csv.

PURN−A0 sensitivity is−0.003797[−0.027349,0.017485], specificity−0.014581
[−0.029058,−0.001664]. No additional detection advantage compensates for more
false-positive EZ predictions. Equal-patient mean FP increases10.425→11.300;
mean TP/FN stay9.275/12.5125, though their patient-level distributions change.
Mean counts are not substituted for pooled statistics or patient-equal recall.

## Fold and center consistency

| Fold | Patients | B0 AUROC | B1 | B2 | B3 | B3−B0 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 0.715573 | 0.703641 | 0.719636 | 0.702972 | −0.012600 |
| 2 | 16 | 0.773140 | 0.740245 | 0.773571 | 0.744471 | −0.028670 |
| 3 | 17 | 0.751367 | 0.750508 | 0.754376 | 0.752191 | +0.000824 |
| 4 | 15 | 0.665688 | 0.652669 | 0.652370 | 0.650677 | −0.015011 |
| 5 | 16 | 0.787718 | 0.780926 | 0.790493 | 0.785379 | −0.002339 |

PURN wins only1/5 folds against A0 and none against B2. Its development5/5
AUROC win pattern does not transport. Fold2 PURN EZ sensitivity is0.246949
versus A00.316417, despite its development-selected threshold; that threshold
is not changed to repair outer performance.

| Center | Patients | B0 AUROC | B1 | B2 | B3 | B3−B0 |
|---|---:|---:|---:|---:|---:|---:|
| HUP | 36 | 0.788700 | 0.783719 | 0.786426 | 0.785400 | −0.003300 |
| LZU | 21 | 0.637965 | 0.627920 | 0.634784 | 0.626970 | −0.010995 |
| Multicenter | 15 | 0.816435 | 0.776596 | 0.808465 | 0.778664 | −0.037771 |
| Pediatric | 8 | 0.643059 | 0.637063 | 0.672552 | 0.643978 | +0.000919 |

Only a tiny Pediatric AUROC point increase; all other centers decline. Pediatric
Macro-F1 still falls0.572267→0.562576. No center-held-out transport or center-
specific tuning is claimed. All metrics by fold/center/fixed channel strata
are in outer_results; subgroup changes are descriptive, not new claims.

## Secondary pooled-channel confusion and ranking

Each fold retains its own frozen threshold. These are not primary equal-patient
AUROC/F1. MRR/Top1 are not calculated as if the whole dataset were one patient.

| Arm | AUROC | AP | Macro-F1 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.697201 | 0.422658 | 0.646789 | 742 | 834 | 5,058 | 1,001 |
| B1 | 0.682078 | 0.424560 | 0.637831 | 753 | 938 | 4,954 | 990 |
| B2 | 0.692790 | 0.435455 | 0.645290 | 714 | 781 | 5,111 | 1,029 |
| B3 | 0.682254 | 0.423199 | 0.638774 | 742 | 904 | 4,988 | 1,001 |

PURN adds70 pooled FP versus A0 without any net additional EZ TP. Its patient-
equal sensitivity point estimate also falls. This is not the development
sensitivity benefit continuing into test; it is worse ranking plus extra FP.

## Factorial interpretation and development transport

Outer AUROC interaction(B3−B2)−(B1−B0) is+0.001990[−0.006262,0.010674],
AP+0.000183[−0.008970,0.009883], Macro-F1−0.004412[−0.016561,0.007738],
EZ-F1−0.006283[−0.025321,0.012847], MRR−0.029978[−0.079888,0.013496].
No supported complementary interaction or physiological causality is shown.

RMS-only arms B1/B3 both fall below A0 on outer AUROC with negative descriptive
intervals. Unique-stat centered B2 is near A0 with no supported increment.
This does not establish that redundant moments were harmless universally,
but the specific correction fails to yield useful held-out ranking here.
Development B3−B0+0.009153 becomes outer−0.011361. Different cohorts/weighting
and selection roles mean this is transport evidence, not a paired test of
development-minus-outer averages. No further iteration is authorized by it.

## Verification and delivery

All20 selections locked before new test inference. Validation CPU replay max
logit drift4.768372e-7 and probability drift1.788139e-7 are within the locked
2e-6/2e-7 tolerances; threshold decisions match. Frozen preprocessing replays
bitwise. All15 new outputs seal before full y; original clinical bank labels
match export, and each of80 patients/7,635 channels is covered once per arm.
Input, model, checkpoint, thresholds, code, baseline scores and original failed
development gate are unchanged. No training or tuning call exists in outer code.

One healthy frozen pass and independent verification PASS:320 patient-model
cells,96 matched contrast/interaction intervals, max discrepancy3.33e-16.
All outer phase stderr files are empty. Original predictions/checkpoints,
individual patient/channel data and logs remain on the server. Public output
is source, hashes, anonymous aggregates and reports only; no installed skills.

This extension completes the user's additional request, not independent
confirmation or a retroactive successful development gate. The evidence does
not support adopting PURN; the study stops with original A0 retained.
