# Ictal D2: combined Source and ZDELTA deletion, seed42

## Result

The jointly simplified model reaches **AUROC0.736977, AP0.530653,
Macro-F10.624236, EZ-F10.414671**, under equal-patient outer evaluation over
the same80 patients /7635 canonical channels. Its principal point estimates
do not outperform the existing source-free-only arm. It has slightly higher
AUROC/Macro-F1 but lower AP/MRR/Top1 than Full D2, with no supported primary
gain under the descriptive paired bootstrap intervals.

| Model | AUROC | AP | Macro-F1 | EZ-F1 | MRR | Top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full D2, frozen reference | 0.730445 | 0.533483 | 0.620403 | 0.410909 | 0.737216 | 0.650000 |
| w/o Source, frozen reference | 0.739768 | 0.537887 | 0.624919 | 0.419405 | 0.728620 | 0.637500 |
| w/o ZDELTA, frozen reference | 0.733994 | 0.526393 | 0.618839 | 0.409059 | 0.697692 | 0.600000 |
| **w/o Source + w/o ZDELTA, newly trained** | **0.736977** | **0.530653** | **0.624236** | **0.414671** | **0.694267** | **0.587500** |

## All eleven combination metrics

| Metric | Patient-equal estimate |
| --- | ---: |
| AUROC | 0.7369771785860808 |
| AP / AUPRC | 0.5306525381402717 |
| Macro-F1 | 0.6242355736555874 |
| EZ-F1 | 0.4146706708280264 |
| NEZ-F1 | 0.8338004764831485 |
| Balanced accuracy | 0.6639904530221834 |
| Sensitivity | 0.4600459975969726 |
| Specificity | 0.8679349084473943 |
| MRR | 0.6942666410497292 |
| Top1 EZ | 0.5875 |
| Accuracy | 0.7631719084697796 |

Full-precision values for all four models and all eleven metrics are in
`results/OUTER_SUMMARY.csv`. Fold/center tables are also provided; headline
values are not the mean of unequal-fold means.

## Paired effects

Direction is **combination minus reference**. Positive favors the new
combination. Use10000 paired resamples of the same80 unique patients, seed42,
with every model's own validation-selected thresholds fixed. Entries are
effect [unadjusted percentile95 CI].

| Reference | ΔAUROC | ΔAP | ΔMacro-F1 | ΔEZ-F1 |
| --- | --- | --- | --- | --- |
| Full D2 | +0.006532 [−0.010743, 0.026120] | −0.002830 [−0.020810, 0.014925] | +0.003832 [−0.011098, 0.018797] | +0.003762 [−0.019501, 0.026574] |
| w/o Source | −0.002791 [−0.013290, 0.006292] | −0.007235 [−0.020354, 0.006002] | −0.000684 [−0.009662, 0.008299] | −0.004734 [−0.019539, 0.009966] |
| w/o ZDELTA | +0.002984 [−0.006058, 0.012279] | +0.004260 [−0.004258, 0.012537] | +0.005397 [−0.006669, 0.018734] | +0.005611 [−0.015061, 0.029188] |

All twelve primary intervals span zero. Neither primary improvement nor
primary degradation is established at this descriptive interval level.
Secondary ranking does not improve: combination-minus-Full MRR is−0.042949
[−0.109181,0.021837], Top1 is−0.0625 [−0.1625,0.025]. These intervals also span
zero. All33 effects and intervals are included in `PAIRED_BOOTSTRAP.csv`.

## Exact configuration and execution

- One new arm, five original outer folds, seed42+1009*fold. No extra model,
  threshold, seed or feature search. There is no ensemble or score averaging.
- Original shared-only PR-MLP: LayerNorm88, Linear88→96, GELU, Dropout0.15,
  Linear96→1. **8817 parameters**, with no source V/u/b. This is not bias-only
  historical R0 and does not warm-start from a final model.
- Retain Patient Z and the original FIT-only imputer/scaler. Mask all26
  ZDELTA columns, including8 auxiliary duplicate copies, **after** complete
  preprocessing. Keep62 ABS/DELTA/RATIO columns in the original88D interface.
  The actual inputs are bitwise identical to the previous w/o ZDELTA arm;
  only its source model is replaced by the original shared-only constructor.
- Import the pinned original trainer without editing it. AdamW LR0.001,
  weight decay0.0001, FIT-only NEZ-positive weighted BCE, one patient update,
  gradient clip1, maximum30/minimum6 epochs, patience6. Source penalty is zero
  because no source parameters exist. Shared initialization matches original
  D2 exactly on all five folds.
- Every fold independently selects epoch and threshold on validation by the
  original patient Macro-F1/EZ-F1/earlier-epoch checkpoint rule and the exact
  original0.005 threshold grid/tie order. No TEST-derived choice in this run.
- Outer labels unchanged: EZ0, NEZ1. Score is sigmoid(NEZ logit); EZ iff score
  is below the frozen threshold. Rank ascending probabilities with stable
  canonical channel-string ties.

| Fold | Selected epoch | Validation-selected NEZ threshold |
| --- | ---: | ---: |
| 1 | 3 | 0.465 |
| 2 | 3 | 0.240 |
| 3 | 2 | 0.320 |
| 4 | 3 | 0.335 |
| 5 | 4 | 0.410 |

All five new selections and15 reference bindings froze before new inference;
all20 prediction files sealed before new full-cohort clinical labels were
materialized. The three references reuse exact immutable private scores and
reproduce the prior study's aggregates within1e-12. Actual patient membership,
feature order, stored FIT-only preprocessing, masks and source-free identity
were gated before training. A combined-path CUDA gradient and exact
uninterrupted/resumed model/Adam/history/RNG test passed on synthetic data.

The five new folds completed **45 epochs**, without error/retry. The scientific
train/freeze/evaluation pipeline took5.67 seconds, excluding implementation,
preparation, verification and transfer. Existing GPU work was not interrupted.

Independent score-only verification used the already admitted analysis runtime
(NumPy2.3.5, sklearn1.7.2), not the scientific training environment. It did not
run a model. All320 model-patient cells ×11 metrics and all33 paired effects
matched independent sklearn/ranking/bootstrap arithmetic. Maximum discrepancy
was **3.3306690738754696e-16**. Historical checkpoint/bank/prediction bindings
and source hashes remain unchanged. No installed skill was used.

## Interpretation

Jointly dropping Source and ZDELTA does not add a clear gain to source removal
alone. In fact its AUROC, AP, Macro-F1, EZ-F1, MRR and Top1 point estimates are
all lower than the source-free-only reference. The reductions are not proven
significant here. Compared with Full D2 it trades a small AUROC/Macro-F1 point
increase for lower AP and patient-top ranking; describing it as uniformly
better would be incorrect.

The choice of this combination was informed by the previous outer ablation
results. Therefore this is **posthoc outer-informed exploratory repeated
evaluation**, not a new blind test or external confirmation. Bootstrap
intervals are unadjusted and conditional on seed42; they do not remove the
adaptivity of choosing the configuration after earlier TEST viewing and do
not measure seed/split uncertainty. Acquiring-source stratification is not
unseen-center validation. Stop after this single requested combination.

Public delivery includes code, protocol, aggregate histories, metrics, fold/
center tables, bootstrap and audits only. Raw EEG, feature archives, patient/
channel identities, checkpoints, individual predictions and logs remain on
the original server.
