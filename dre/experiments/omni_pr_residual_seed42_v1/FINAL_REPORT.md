# Omni PR-Residual seed42 final report

This is an exploratory repeated-test result because the official test cohort
was viewed in earlier experiments. The CNN checkpoint, both residual-head
checkpoints, and all validation-only thresholds were hash-frozen before new
TEST embedding extraction. TEST was not used for training, model selection,
threshold selection, or iteration.

| Model | New trainable params | AUROC | AP | Macro-F1 | MRR | Top1 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen TimeConv-CNN | 0 | 0.798767 | 0.330291 | 0.657264 | 0.831977 | 0.772727 |
| ABS-only residual | 1,761 | 0.798727 | 0.330432 | 0.658424 | 0.837659 | 0.784091 |
| PR-Residual | 1,761 | 0.798489 | 0.330363 | 0.658275 | 0.837659 | 0.784091 |

## Outcome

Terminal: **FAIL**.

The frozen baseline replay passed at AUROC `0.7987670919`, an absolute error
of `2.55e-7` from the locked `0.7987673466` reference. PR-Residual reached
AUROC `0.7984892707`: delta `-0.0002778212` versus FrozenCNN and
`-0.0002372349` versus ABS-only. It did not exceed either the frozen baseline
or the published `0.8061` reference.

The thresholded metrics do not rescue the scientific claim. PR-Residual had
AP `0.330363`, Macro-F1 `0.658275`, pathological F1 `0.388235`, balanced
accuracy `0.665883`, sensitivity `0.408922`, specificity `0.922845`, and
accuracy `0.871668`. Its confusion matrix was TP/FP/TN/FN =
`330/563/6734/477`. The minor Macro-F1 increase over FrozenCNN is caused by a
different validation-frozen threshold and does not indicate improved ranking.

## Interpretation

The result is negative and the explanation is direct: the learned
patient-relative correction adds no measurable ranking information beyond the
frozen local morphology encoder. ABS-only is also flat, so extra classifier
capacity is not useful. PR-Residual scores are almost a constant negative
shift: mean `-0.379897`, SD `0.000998`, p5/p25/p50/p75/p95 =
`-0.380563/-0.380416/-0.380172/-0.379744/-0.378419`; saturation fraction at
`|delta| >= 0.49` is `0`. This is functionally close to intercept adjustment,
not a patient-relative correction.

The validation result already anticipated failure: pooled AUROC was
`0.935015` for FrozenCNN, `0.935031` for ABS-only, and `0.934929` for PR-CNN.
The official TEST confirms that the apparent train-side differences were
noise-level.

## Runtime and parameter audit

- Frozen CNN parameters: `11,302,241`, all frozen.
- New trainable parameters per residual head: `1,761` (<2,500).
- ABS-only training time: `24.593 s`.
- PR-Residual training time: `24.672 s`.
- Cached-head inference time: FrozenCNN `0.257 s`, PR-Residual `0.316 s`.
- TEST extraction: 237 EDFs, 240,074 segments, 8,104 labeled pairs.
- Forward identity max absolute error: `0.0`.

## Stopped diagnostics

The primary TEST metrics were complete when the user directed the run to stop
if AUROC did not improve. The remaining patient bootstrap, center analysis,
relative-zero, and 100-repeat patient-shuffle diagnostics were therefore not
completed and are not fabricated. `PATIENT_METRICS.csv` contains only the
already-completed aggregate patient distributions. Native NumPy process exits
and engineering retries are documented in `ENGINEERING_REPAIR_LEDGER.md`.

## Aggregation consistency

The frozen official evaluator averages sigmoid probabilities over segments.
The prompt also called `sigmoid(mean(logit))` official, but that alternative
replays at about `0.790139`, violating the mandatory baseline gate. The primary
result therefore adds one EDF-channel residual to every segment logit and
retains the actual frozen sigmoid-then-mean evaluator. The alternative remains
diagnostic-only in `TEST_METRICS.csv`.

`MODEL_FROZEN_BEFORE_FINAL_TEST = YES`

`FINAL_HELDOUT_ACCESSED = YES`

`FINAL_TEST_USED_FOR_TUNING = NO`
