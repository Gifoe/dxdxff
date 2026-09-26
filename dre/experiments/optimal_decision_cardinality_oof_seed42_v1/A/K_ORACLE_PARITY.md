# Optimal-K / threshold oracle parity

Using the frozen B0 EZ ranking, all `K=0..C` decisions were enumerated for 255 outer-fit OOF patient-fold episodes and 65 outer-validation episodes. The 25 missing nested B0 selected thresholds were deterministically replayed; each selected epoch and nested-validation Macro-F1 matched its saved cross-fit audit. The 25 existing OOF prediction files were reused, not retrained.

| Partition | Episodes | Top-K oracle patient Macro-F1 | Reachable threshold oracle | Difference |
| --- | ---: | ---: | ---: | ---: |
| Outer-fit OOF | 255 | 0.699772 | 0.699556 | +0.000216 |
| Outer-validation | 65 | 0.702841 | 0.702841 | 0.000000 |

One fit episode had exact-score ties and was the sole episode for which stable-order Top-K could split a tie and exceed a scalar threshold. No validation episode had tied scores. One validation episode had multiple, disconnected optimal K values; its complete optimal set was retained in the private target, rather than replacing it by a single encompassing interval. For all patients without ties, the two oracle maxima matched within `1e-9`.

The validation Top-K oracle exactly reproduces the earlier ~0.70 patient-threshold ceiling. These oracles use patient labels for diagnosis and training-target construction only; neither is a deployable predictor or a held-out C3 result. Formal B0 validation Macro-F1 remained `0.628380`.
