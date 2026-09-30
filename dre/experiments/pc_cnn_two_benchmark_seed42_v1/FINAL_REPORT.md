# PC-CNN seed42: frozen two-benchmark terminal report

## Terminal

`ONE_BENCHMARK_ONLY`.

PC-CNN does not pass the Ictal minimum gate. It does show a small positive
Omni AUROC difference against the matched RawCNN on the frozen historical
supervised cohort, but the independently trained RawCNN is far below the prior
official-CNN reproduction reference. This is not evidence that PC-CNN is a
viable unified replacement, and this experiment stops here.

All checkpoint and numeric-threshold selections were hash-frozen before outer
test I/O in `FROZEN_BEFORE_TEST.json` (SHA-256
`95E4B6A10C107A953686FDF1A1A1AC14C19F7591A2B0EB90611B059A16EF7AEC`).
No test metric was used to choose a model, checkpoint, threshold, architecture,
or hyperparameter. Omni was a repeated-test exploratory analysis because that
test domain had previously been viewed.

## Main results

| Model | Benchmark | AUROC | AP | Macro-F1 | MRR | Top1 |
|---|---|---:|---:|---:|---:|---:|
| A1 historical | Ictal development (47-patient VLOO) | 0.7464 | 0.5767 | 0.6208 | 0.7400 | 0.6548 |
| RawCNN | Ictal frozen outer (80 x 20 queries) | 0.6482 | 0.4638 | 0.4636 | 0.6252 | 0.4883 |
| PC-CNN | Ictal frozen outer (80 x 20 queries) | 0.6420 | 0.4809 | 0.5099 | 0.6274 | 0.4965 |
| A1-v2 historical | Omni supervised scope | 0.6887 | — | 0.5937 | — | — |
| RawCNN | Omni frozen 174 EDF / 96 patient scope | 0.6895 | 0.1663 | 0.5744 | 0.4115 | 0.3077 |
| PC-CNN | Omni frozen 174 EDF / 96 patient scope | 0.7064 | 0.2024 | 0.5912 | 0.4391 | 0.3077 |

Omni AP is pooled EDF-channel AP. The historical A1 Ictal number is a
47-patient VLOO development result, not the paired 80-patient outer test used
here, so it cannot support a formal PC-CNN-minus-A1 significance claim.

## Paired inference

| Benchmark | Contrast | AUROC delta | 95% patient-cluster bootstrap CI | Conclusion |
|---|---|---:|---:|---|
| Ictal | PC-CNN − RawCNN | -0.0048 | [-0.0483, 0.0421] | No AUROC gain |
| Omni | PC-CNN − RawCNN | +0.0177 | [+0.0011, +0.0354] | Small positive AUROC difference |

On Ictal, PC-CNN increases AP by 0.0176 and Macro-F1 by 0.0464, but AUROC was
the primary gate and its confidence interval crosses zero. On Omni, the AUROC,
pooled AP, and Macro-F1 confidence intervals for PC-CNN minus RawCNN are
positive; pathological-F1 does not have a positive 95% lower bound, while
sensitivity drops by 0.0345 and specificity rises by 0.0388. This is an
operating-point tradeoff from the validation-selected thresholds, not a clear
localization gain.

## Required interpretation checks

- The matched Ictal RawCNN is 0.6482 AUROC, materially below historical A1's
  non-paired 0.7464 development reference. Therefore the morphology baseline
  did not transfer strongly enough to support the intended comparison.
- PC-CNN Ictal AUROC is 0.6420. It does not exceed A1, nor RawCNN, and it does
  not pass the Ictal minimum gate.
- PC-CNN Omni AUROC is 0.7064 versus RawCNN 0.6895. It exceeds the matched
  RawCNN by a small amount, but does not approach the published 0.8061,
  0.83, or 0.85 gates. Its validation-selected-threshold Macro-F1 is 0.5912,
  below 0.70.
- The positive-eligible Omni centers HUP, Open-iEEG, and SourceSink all have a
  higher PC-CNN AUROC than RawCNN. Zurich has zero pathological units and its
  pathological metrics are not estimable. There is no observed AUROC center
  reversal, but HUP and SourceSink Macro-F1 decrease.
- The architecture topology and initial disabled-module identity audits pass.
  However, the requested post-training P3 intervention does **not** exactly
  recover the independently selected RawCNN: after Stage C, its AUROC is
  0.7048 rather than 0.6895. This follows because Stage C unfreezes classifier
  and ResNet layer4. Consequently P1/P2/P3 intervention differences cannot be
  interpreted as clean causal contributions of physiology or patient context.
- The actual failure location is first the RawCNN transfer to Ictal, then no
  Ictal incremental physiology/context effect. Omni has a modest within-scope
  improvement but not enough to rescue the unified claim.

## Engineering provenance

The Windows host had three native interpreter/driver failures during frozen
Omni inference (`nvcuda64.dll`, `torch_cpu.dll`, and NumPy). Predictions were
therefore atomically cached per patient and model, each bound to the frozen
protocol, selected checkpoint, official-CNN source, and content hash. The final
aggregate contains 480 verified cache cells (five model variants by 96
patients), 8,104 EDF-channel units, and uses no re-inference after cache
completion. This is an engineering recovery of one frozen evaluation, not a
new selection or a test-driven rerun.
