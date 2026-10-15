# Full M8 w/o PR and D1 w/o PR — completed seed42

## Primary Outer results

Original five folds, 80 held-out patients / 7,635 canonical patient-channels / 256 seizures for every arm. All entries below are **patient-equal means**, not pooled-channel metrics or unweighted means of five folds. EZ is the positive class. AP is EZ average precision.

| Model | AUROC | AP | Macro-F1 | EZ-F1 | MRR | Top1 |
|---|---:|---:|---:|---:|---:|---:|
| D1, original PR | 0.732636 | 0.527175 | 0.616556 | 0.408531 | 0.689399 | 0.575000 |
| D1 w/o PR | 0.653323 | 0.459677 | 0.479030 | 0.318277 | 0.654289 | 0.562500 |
| Full M8, original PR | 0.762397 | 0.554067 | 0.634097 | 0.434087 | 0.749310 | 0.637500 |
| **Full M8 w/o PR** | **0.701743** | **0.503378** | **0.502058** | **0.361961** | **0.716918** | **0.625000** |

| Model | Accuracy | Sensitivity | Specificity | NEZ-F1 | Predicted EZ fraction |
|---|---:|---:|---:|---:|---:|
| D1, original PR | 0.748950 | 0.477193 | 0.854288 | 0.824582 | 0.210810 |
| D1 w/o PR | 0.628425 | 0.495166 | 0.657272 | 0.639783 | 0.373115 |
| Full M8, original PR | 0.760166 | 0.497392 | 0.865313 | 0.834107 | 0.209195 |
| Full M8 w/o PR | 0.646415 | 0.553051 | 0.652563 | 0.642155 | 0.392503 |

**Result:** removing PR damages both models. Full M8 w/o PR improves on D1 w/o PR, but does not recover the original PR-enabled Full M8 or even PR-enabled D1 on AUROC, AP or Macro-F1. The new correction branch does not replace patient-relative normalization under this frozen training protocol.

## Paired patient bootstrap

10,000 draws, seed42, resample the same 80 patient IDs for each arm. Numeric fold thresholds remain fixed; no bootstrap threshold optimization. Delta is **left minus right**. Intervals are 2.5/97.5 percentiles.

| Comparison | AUROC delta [95% CI] | AP delta [95% CI] | Macro-F1 delta [95% CI] | EZ-F1 delta [95% CI] |
|---|---:|---:|---:|---:|
| M8 w/o PR − M8 PR | −0.060654 [−0.098794, −0.025714] | −0.050690 [−0.093549, −0.009788] | −0.132039 [−0.175705, −0.089291] | −0.072126 [−0.125333, −0.018262] |
| D1 w/o PR − D1 PR | −0.079312 [−0.127182, −0.033290] | −0.067498 [−0.113151, −0.023406] | −0.137527 [−0.178380, −0.096971] | −0.090255 [−0.144974, −0.033863] |
| M8 w/o PR − D1 w/o PR | +0.048419 [+0.031265, +0.065792] | +0.043700 [+0.021570, +0.065964] | +0.023028 [+0.006375, +0.040953] | +0.043684 [+0.022489, +0.068043] |
| M8 PR − D1 PR | +0.029761 [+0.006912, +0.056661] | +0.026892 [+0.003448, +0.051300] | +0.017541 [−0.001153, +0.036806] | +0.025556 [−0.006790, +0.059056] |

These are exploratory, unadjusted paired intervals, not multiplicity-corrected confirmatory tests. They show a stable PR-ablation loss within this evaluated cohort/seed, not proof that all possible no-PR training protocols fail.

## Confusion-matrix diagnosis

These are **channel-total counts**, distinct from patient-mean metrics above.

| Model | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| D1 PR | 722 | 844 | 5048 | 1021 |
| D1 w/o PR | 894 | 1945 | 3947 | 849 |
| M8 PR | 747 | 799 | 5093 | 996 |
| M8 w/o PR | 1018 | 1966 | 3926 | 725 |

Removing PR increases sensitivity but produces many more false positives. For M8, FP rises by 1,167 while FN falls by 271. For D1, FP rises by 1,101 while FN falls by 172. Specificity and NEZ-F1 deteriorate sharply. The ranking metrics also fall, so this is **not merely a threshold/F1 problem**. Validation-selected thresholds were frozen before this Outer pass; no attempt was made to repair them using Outer labels.

## What exactly was removed

`D1_NO_PR` uses the original 14D physiological DELTA input, **without cross-channel Patient Z**, followed by the existing FIT-only imputer/StandardScaler and exact 1,537-parameter CompactDelta topology. The original duplicate-aware 88-position LayerNorm semantics are retained; they are not the cross-channel PR operation.

`M8_NO_PR` removes **both** cross-channel PR sites:

1. Its frozen D1 base is the matched `D1_NO_PR` model.
2. Its Raw branch returns the 96D early/late, seizure-mean/population-SD summary **before** cross-channel Patient Z.

It retains the full shared morphology CNN, 7D physiology gate, same-channel latent pre-reference subtraction, same-channel physiological DELTA, 0–8 / 8–20 s stages, original valid masks, available-seizure aggregation, per-window channel-local waveform standardization, original FIT-only physiology scale, head topology and bounded residual `base + tanh(aux)`. No patient-relative replacement or new scaler is introduced for the 96D Raw summary.

The ablation removes both PR sites jointly. It **does not isolate** how much of the M8 drop comes from the D1-input site versus the Raw-summary site. No claim about their separate causal contributions is warranted by these four arms.

## Matched training and reuse

- Newly trained: **five** M8_NO_PR correction models, **71 total epochs** (23 / 11 / 11 / 16 / 10). Selected epochs: **17 / 5 / 5 / 10 / 4**.
- M8_NO_PR validation-selected numeric **NEZ** thresholds: **0.465 / 0.520 / 0.370 / 0.510 / 0.535**. EZ decision is `p_NEZ < threshold`; do not apply these numbers directly to EZ scores.
- Exactly reused: original M8 trainer, AdamW lr 0.0003, weight decay 0.001, gradient norm cap 1, original patient-equal updates/order, 0.75 weighted BCE + 0.25 patient soft Macro-F1, max30/min6/patience6 and original validation-only checkpoint/threshold tie rules. No hyperparameter search.
- Each correction model: **9,481 trainable + 1,537 frozen base = 11,018 total parameters**.
- Initial correction weights exactly match the corresponding original M8 constructor. Each new model initially outputs exactly its frozen no-PR D1 base. Three legal FIT smoke-gradient steps per fold are discarded; the model is rebuilt from the same seed before actual training.
- D1_NO_PR was **not trained again**: the exact scratch-trained A1 arm of the completed D1→M8 progressive ablation is hash-verified and reused, including all five selected checkpoints, FIT-only transforms, thresholds and sealed Outer scores. It is not the unrelated sklearn MLP baseline.
- Original D1_PR and M8_PR likewise reuse their exact sealed reference score ledgers. All three reused arms reproduce previous patient metrics exactly.
- No raw extraction or cache modifications. No unrelated GPU process was changed.

## Validation-only development results

65 fold-patient appearances, 47 unique patients per arm; distinct from the 80-patient Outer table.

| Model | AUROC | AP | Macro-F1 | EZ-F1 | MRR | Top1 |
|---|---:|---:|---:|---:|---:|---:|
| D1 PR | 0.725978 | 0.503601 | 0.644546 | 0.443537 | 0.658502 | 0.523077 |
| D1 w/o PR | 0.681432 | 0.470875 | 0.566046 | 0.332951 | 0.665033 | 0.584615 |
| M8 PR | 0.756633 | 0.542226 | 0.666795 | 0.477249 | 0.699708 | 0.569231 |
| M8 w/o PR | 0.722688 | 0.508467 | 0.590646 | 0.376351 | 0.723979 | 0.630769 |

## Integrity and scope

All five new checkpoints, input banks and validation thresholds were hash-locked before new Outer inference. Each own-fold TEST patient is excluded from FIT and validation, preprocessing and conditioning-scale estimation. All 20 model-fold score files (five new, fifteen reused) were sealed before the first new full Outer-label read.

Checks actually run:

- Original PR-enabled feature calculation versus copied implementation: **exact equality on an actual legal validation patient in each fold**.
- PR-disabled shared-channel invariance under channel removal: max error <1e-5 in all five actual validation checks; no cross-channel statistic remains in the ablated Raw summary.
- Original M8 initial full-state reconstruction and new-correction initialization parity: all five pass.
- Initial base identity and actual FIT gradient/frozen-base tests: all five pass.
- Outer embedded no-PR D1 logit parity: maximum **2.384185791015625e-7**, below 1e-6; residual bounds hold.
- Independent sklearn recomputation: **320 patient rows**; maximum metric error **3.3306690738754696e-16**.
- Independently reconstructed paired bootstrap: four comparisons; maximum error **1.0061396160665481e-16**.
- Original M8 and phase2 training source hashes unchanged; source bytes and public artifact SHA256s recorded.

One import-path failure occurred **before** setup/training, caused by dynamically loading the historical trainer without its sibling module directory on `sys.path`. The path was fixed before the new input freeze. No completed epoch/checkpoint was discarded, no mathematical/data/hyperparameter change occurred, and private initial logs/source were retained. Training and Outer then completed without a numerical/runtime failure.

Scientific status remains **EXPLORATORY_REPEATED_TEST**: the same Outer cohort has been viewed in earlier experiments, and M8 was historically developed adaptively. This run freezes the new ablation before its Outer pass, but does not turn the prior architecture or cohort into a previously unseen confirmation. This is a matched seed42 ablation, not a multi-seed generalization claim.

## Final conclusion

Under the original matched protocol, **PR should not be removed from either D1 or Full M8**. Full M8 without PR retains a useful increment over D1 without PR, but loses 6.07 AUROC percentage points and 13.20 Macro-F1 points versus original Full M8. D1 loses 7.93 AUROC points and 13.75 Macro-F1 points without PR. All requested original five-fold Outer results are complete; no additional tuning was performed.
