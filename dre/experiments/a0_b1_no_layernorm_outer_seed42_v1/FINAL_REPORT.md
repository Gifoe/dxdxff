# B1: A0 without LayerNorm — frozen outer test completed

**Scientific terminal: `B1_OUTER_COMPLETE_NO_SUPPORTED_INCREMENTAL_GAIN`.**
Primary patient-equal AUROC is 0.740841 versus frozen A0's 0.739768.
The development ranking gain did not carry through to a comparable outer
gain. No checkpoint, threshold, feature, loss or architecture modification
was made after viewing outer results.

## Authorization and scope

The user explicitly requested B1 outer after seeing its PMA development
comparison. This separate amendment authorizes B1 only and does not relax
PMA's failed continuation gate. B2/B3 outer remains unperformed. Exactly
five previously frozen B1 checkpoints are used: epochs 2/2/2/1/3 and NEZ
thresholds 0.485/0.405/0.425/0.435/0.490. B1 has 8,641 parameters, retaining
the original A0 linear layers, GELU, Dropout/head and original88D Patient Z
plus stored FIT-only imputer/scaler. Only input LayerNorm was removed during
the previously completed training.

No new training, checkpoint changes, threshold re-selection, raw EEG
extraction, patient exclusions or alternative evaluation variants occurred.
Frozen B0 scores are reused; B0 is not re-inferred.

This is a **user-authorized post-development model-selection exploratory
repeated-outer evaluation**. The outer cohort has historical viewing, and
B1 was chosen after seeing development. It must not be presented as a
fresh blind confirmation or a preregistered new-B1 hypothesis test.

## Primary outer results

Equal-patient means across 80 unique outer TEST patients, each appearing
once. Coverage: 7,635 canonical patient-channels; 1,743 EZ/5,892 NEZ.
All80 patients have estimable AUROC; no patient is removed.

| Metric | B0 A0 | B1 w/o LN | B1−B0 | Paired95% CI |
|---|---:|---:|---:|---|
| AUROC | 0.739768 | 0.740841 | +0.001073 | [−0.016461,+0.017722] |
| AUPRC | 0.537887 | 0.549391 | +0.011504 | [−0.008181,+0.031785] |
| Macro-F1 | 0.624919 | 0.619532 | −0.005387 | [−0.019412,+0.007905] |
| EZ-F1 | 0.419405 | 0.427230 | +0.007826 | [−0.012777,+0.028284] |
| NEZ-F1 | 0.830434 | 0.811835 | −0.018599 | [−0.042557,−0.002143] |
| Balanced accuracy | 0.666405 | 0.670744 | +0.004340 | [−0.010107,+0.018751] |
| MRR | 0.728620 | 0.759681 | +0.031062 | [−0.031394,+0.095226] |
| Top1-EZ | 0.637500 | 0.675000 | +0.037500 | [−0.050000,+0.125000] |
| Sensitivity | 0.471033 | 0.512851 | +0.041818 | [+0.007523,+0.080970] |
| Specificity | 0.861776 | 0.828638 | −0.033138 | [−0.063193,−0.010066] |
| Accuracy | 0.758773 | 0.738412 | −0.020360 | [−0.042782,−0.003837] |

10,000 seed42 paired bootstrap draws resample80 unique patients with
replacement. The same sampled identities are used for both models; all
their channels remain together. Model/threshold selection is not repeated
inside the draws. Intervals are unadjusted percentile95%, not corrections
for the historical experiment sequence or model-selection uncertainty.
Absolute model intervals are in `OUTER_PATIENT_EQUAL_METRICS.csv`.

The primary ranking and F1 intervals all cross zero. Higher sensitivity
comes with lower specificity, NEZ-F1 and accuracy. This is not an established
overall localization improvement. The small AUROC point difference cannot
support replacing A0 with B1 as a superior model.

## Fold consistency

| Fold | Patients | Channels | B0 AUROC | B1 AUROC | Difference |
|---|---:|---:|---:|---:|---:|
| 1 | 16 | 1,631 | 0.715573 | 0.714049 | −0.001523 |
| 2 | 16 | 1,498 | 0.773140 | 0.763472 | −0.009668 |
| 3 | 17 | 1,746 | 0.751367 | 0.776011 | +0.024645 |
| 4 | 15 | 1,425 | 0.665688 | 0.664971 | −0.000717 |
| 5 | 16 | 1,335 | 0.787718 | 0.778761 | −0.008956 |

B1 improves AUROC in only1/5 outer folds. The aggregate is weighted by
patients, not an equal-five-fold mean, because fold sizes differ.

## Center decomposition

| Center | Patients | B0 AUROC | B1 AUROC | Difference |
|---|---:|---:|---:|---:|
| HUP | 36 | 0.788700 | 0.810080 | +0.021381 |
| LZU | 21 | 0.637965 | 0.626876 | −0.011088 |
| Multicenter | 15 | 0.816435 | 0.780384 | −0.036051 |
| Pediatric | 8 | 0.643059 | 0.654278 | +0.011220 |

Center effects are mixed. No center-dependent threshold or normalization
was introduced. Per-center values are descriptive, not selected subcohorts
for a positive claim. Complete11-metric decompositions are in the CSVs.

## Secondary pooled-channel statistics

This is not the primary equal-patient estimand. Each fold retains its own
frozen threshold; pooled MRR/Top1 are intentionally omitted.

| Model | AUROC | AUPRC | Macro-F1 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.697201 | 0.422658 | 0.646789 | 742 | 834 | 5,058 | 1,001 |
| B1 | 0.690766 | 0.372846 | 0.627837 | 767 | 1,060 | 4,832 | 976 |

B1 has25 fewer missed EZ channels but226 more false-positive EZ channels.
The predicted-EZ fraction increases from0.206418 to0.239293. These pooled
counts explain a sensitivity–specificity tradeoff; they must not be used
to replace or conflate the equal-patient primary metrics.

## Development versus outer

The original65-appearance development AUROC B1−B0 difference was+0.025754.
The new80-patient outer difference is+0.001073, with a CI spanning zero.
This suggests the development advantage is not a demonstrated generalizable
ranking improvement. Development and outer have different patient weighting
and selection roles, so their raw numbers are not paired against one another.
No further iterations are attempted to recover the development advantage.

## Verification

- Protocol/code, frozen development selections, original banks and B0 outer
  seal hashes verified before any new test feature inference.
- All five stored FIT preprocessing transforms replay bitwise on legal
  FIT/VAL banks, with zero FIT–TEST overlap per fold.
- CPU inference matches original outer backend; legal validation logit max
  drift9.536743e-7 and probability max drift1.192093e-7, within the original
  predeclared2e-6/2e-7 tolerances. Thresholded validation decisions are identical.
- All five B1 prediction files sealed before full-cohort y materialization.
- Complete unique-patient/channel coverage and clinical label parity checked.
- No dependencies, checkpoints, thresholds or parent PMA gate changed.
- Independent sklearn verification of160 patient-model metric cells and
  22 absolute/11 paired intervals passed; maximum metric drift3.33e-16.
- Completed evaluation refuses rerun. Hash-verified resume is available for
  interruption but was not needed during this completed outer pass.

Only source, hashes and compact anonymous aggregates are published. Features,
checkpoints, patient/channel predictions and detailed runtime records remain
private on the authorized server. No installed skills were used.
