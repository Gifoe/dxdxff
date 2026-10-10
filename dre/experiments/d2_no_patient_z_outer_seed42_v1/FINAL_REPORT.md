# D2 without patient-local z-standardization — ictal outer, seed42

Terminal: `ONE_FROZEN_NO_PATIENT_Z_OUTER_PASS_COMPLETE`.
Scientific status: `EXPLORATORY_REPEATED_OUTER_TEST_SINGLE_SEED`.

Removing patient_z substantially worsened this D2 experiment. Patient-equal
outer Macro-F1 fell from **0.620403 to 0.493978** and EZ AUROC fell from
**0.730445 to 0.660217**. Macro-F1 decreased in all five folds and all four
operational sources. Of 80 patients, 59 had lower Macro-F1 and 21 had higher
Macro-F1. The paired patient bootstrap Macro-F1 difference is **-0.126425**,
95% interval **[-0.162872, -0.089640]**.

## Exact ablation identity

This run removes only the original `uas_core.patient_z`: within-patient,
cross-channel mean/std normalization of the 88 aggregated input features.
It keeps the original 88 feature columns, including their pre-onset physiological
reference delta, zdelta and ratio views. It also keeps the model's LayerNorm.
Thus this is **D2 without patient-local input z-standardization**, not an
absolute-only feature model or removal of all patient-relative physiology.

The new arm is trained from scratch, not evaluated by sending differently scaled
inputs into an already trained PR checkpoint. Every fold retains the original
shared/source initialization tensors, model class, 9,221 parameters, acquisition
source correction, labels, patient split, objective, optimizer, patient update
order, validation selection rule and early stopping. Only the preprocessing path
changes. The original PR arm is reused from its already frozen outer predictions,
without retraining it or changing its probabilities/thresholds.

The no-patient_z path fits the original SimpleImputer and StandardScaler only on
FIT patients. A patient's inference does not use that patient's other channels
to estimate feature normalization. Validation chooses each newly trained arm's
own checkpoint and threshold; reusing the old PR numeric thresholds would not be
the original selection protocol for a retrained ablation.

## All primary patient-equal outer metrics

There are 80 unique TEST patients, each evaluated once in its original fold,
covering 7,635 canonical channel pairs with 1,743 EZ labels. All patients have both
classes. EZ=0, NEZ=1; model sigmoid output is NEZ probability. Predict EZ when
score is below the selected fold-specific threshold. Ranking uses original
ascending NEZ probability with stable canonical channel-string ties.

| Metric | Original D2 PR | D2 without patient_z | Difference |
|---|---:|---:|---:|
| Macro-F1 | 0.620403 | 0.493978 | -0.126425 |
| EZ-F1 | 0.410909 | 0.259845 | -0.151064 |
| NEZ-F1 | 0.829898 | 0.728112 | -0.101787 |
| Balanced accuracy | 0.652228 | 0.569316 | -0.082912 |
| EZ AP / AUPRC | 0.533483 | 0.480013 | -0.053470 |
| EZ AUROC | 0.730445 | 0.660217 | -0.070228 |
| EZ MRR | 0.737216 | 0.685571 | -0.051645 |
| Top1 is EZ | 0.650000 | 0.600000 | -0.050000 |
| Sensitivity / EZ recall | 0.435823 | 0.363673 | -0.072150 |
| Specificity / NEZ recall | 0.868634 | 0.774958 | -0.093675 |
| Accuracy | 0.759548 | 0.705398 | -0.054150 |

## Paired patient-cluster bootstrap

10,000 draws, seed42, resampling the same 80 patients for both arms, with numeric
thresholds fixed throughout. Each patient contributes equal weight. Selected
differences below are no-patient_z minus PR; all 11 differences and the two arms'
absolute intervals are in the result CSVs.

| Metric | Difference | 95% percentile interval |
|---|---:|---:|
| Macro-F1 | -0.126425 | [-0.162872, -0.089640] |
| EZ-F1 | -0.151064 | [-0.211649, -0.089214] |
| EZ AP | -0.053470 | [-0.094825, -0.014419] |
| EZ AUROC | -0.070228 | [-0.110716, -0.030976] |
| Sensitivity | -0.072150 | [-0.160249, 0.017169] |
| Specificity | -0.093675 | [-0.165674, -0.029260] |
| EZ MRR | -0.051645 | [-0.135962, 0.032036] |
| Top1 is EZ | -0.050000 | [-0.162500, 0.062500] |

These intervals are descriptive for a repeatedly examined cohort and one seed;
they do not adjust for previous adaptive model experiments. In particular, the
recall, MRR and Top1 intervals include zero, despite lower point estimates.

## Fold results and frozen selections

| Fold | TEST patients | TEST channels | No-patient_z selected epoch | NEZ threshold | PR Macro-F1 | No-patient_z Macro-F1 | PR AUROC | No-patient_z AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 1,631 | 12 | 0.280 | 0.605008 | 0.470769 | 0.711376 | 0.632274 |
| 2 | 16 | 1,498 | 3 | 0.345 | 0.610657 | 0.479568 | 0.773011 | 0.641728 |
| 3 | 17 | 1,746 | 5 | 0.490 | 0.634132 | 0.491698 | 0.747228 | 0.733034 |
| 4 | 15 | 1,425 | 2 | 0.485 | 0.564513 | 0.534516 | 0.624880 | 0.604933 |
| 5 | 16 | 1,335 | 10 | 0.325 | 0.683356 | 0.496018 | 0.788083 | 0.681110 |

Training stopped under the original rule at epochs 18, 9, 11, 8 and 16,
respectively (62 completed epochs across all five folds). A lower selected epoch
does not mean training was truncated: all epochs up to original early stopping
are retained in the aggregate training history and private resume checkpoint.

## Source results

Patient-equal within each known operational source; these are not center-held-out
tests. All 11 source-level metrics are in `OUTER_BY_CENTER.csv`.

| Source | Patients | PR Macro-F1 | No-patient_z Macro-F1 | PR AUROC | No-patient_z AUROC |
|---|---:|---:|---:|---:|---:|
| HUP | 36 | 0.672629 | 0.521074 | 0.789174 | 0.718479 |
| LZU | 21 | 0.542499 | 0.436626 | 0.625872 | 0.559010 |
| Multicenter | 15 | 0.639219 | 0.541626 | 0.813430 | 0.682873 |
| Pediatric | 8 | 0.554607 | 0.433262 | 0.585074 | 0.621229 |

The pediatric exception matters: its AUROC and AP improve without patient_z
while its Macro-F1 and EZ-F1 fall. Therefore the result supports an overall
benefit of patient_z in this run, not an unconditional ranking gain in every
source. Fold 2 no-patient_z EZ-F1 is only 0.045682 and EZ recall is 0.038389;
the selected threshold almost always predicts NEZ in that outer fold. This is a
failure under validation-to-TEST transport, not a reason to select a new threshold
from TEST labels. Ranking also falls overall, so the loss is not solely an F1
operating-point problem.

## Execution and verification

The execution reused the original hash-pinned `train.py` rather than a new trainer.
It preserved CUDA training in the original Python runtime (PyTorch 2.11.0+cu128),
with two CPU threads, AdamW lr0.001/weight_decay0.0001, source L2 coefficient0.001,
gradient clip1, one optimizer update per FIT patient, FIT-only NEZ-positive BCE
class weighting, seed42+1009*fold, maximum30 epochs, minimum6 and patience6.
The .005 validation threshold grid and original tie-break rules were unchanged.

All five initial-state hashes match the corresponding original D2 initialization.
The no-patient_z transformed feature banks match independently rebuilt FIT-only
sklearn imputer/scaler transforms bitwise. No raw signal extraction, feature
reselection, center-specific scaler, additional arm, hyperparameter search,
checkpoint warm-start or test-driven change was performed.

All five selected checkpoints/scalers/thresholds were frozen before new outer
inference. All five outer score files were then sealed before loading the full
NPZ clinical label member. Each fold's bank contained only its FIT+VAL labels
during training; the feature export's corresponding labels were subsequently
checked against those banks. Original PR scores match its prior frozen outer
metrics exactly and all historical dependency hashes remain unchanged.

New pre-outer freeze SHA256:
`3c5b19aaaad83dbd41f5306f31440e717e049d46290e062145b510deadb28b8e`.

New outer prediction-seal SHA256:
`459253f1dd88f8adb94050ad0170e036deac8d110571cc0b765b77d3ca905343`.

Two focused synthetic preprocessing tests passed before training. Independent
read-only sklearn/NumPy verification passed for 160 model-patient cells, all 11
metrics, both absolute and paired 10,000-draw bootstrap intervals, five FIT-only
preprocessors and original training bindings. Maximum metric discrepancy was
2.2204460493e-16. It did not generate a second set of test model predictions.

## Interpretation and scope

Under this fixed seed42 training contract, removing patient-local cross-channel
input z-standardization worsens D2. The existing source-conditioned head does not
compensate for that removal. The observed evidence favors keeping patient_z for
this D2 configuration; it does not identify a physiological cause, establish
unseen-center transfer, or imply that every possible absolute-feature model must
perform worse. This task ends here without retuning the failed ablation.

The result is exploratory repeated-outer evidence because prior results on these
80 patients have already been inspected. Only code, hashes and aggregate outputs
are published. Features, raw EEG, checkpoints, patient identifiers, per-channel
predictions and runtime logs remain private on the original server.
