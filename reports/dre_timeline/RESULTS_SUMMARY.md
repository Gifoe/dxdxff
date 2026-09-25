# Compact result summary

## P2-Q10 / V3-QBC seed-42 ablation (2026-07-20)

The source report describes 80 patients, 7,635 channels, and five folds. Its status labels matter: P2-only and V3-only are `ABLATION_ONLY_NOT_FOR_MODEL_SELECTION`; the 0.90/0.10 locked fusion is `EXPLORATORY_LOCKED_FUSION`. The source report says the fusion weight was locked after observing the current seed-42 development OOF results, so these numbers are exploratory rather than an independent confirmatory estimate.

| Experiment | Patient-macro F1 | Patient balanced accuracy | EZ F1 | EZ AUPRC | EZ MRR | Δ patient-macro F1 vs P2-only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2-Q10 only | 0.6285 | 0.6653 | 0.4303 | 0.5212 | 0.6886 | reference |
| V3-QBC only | 0.6225 | 0.6710 | 0.4287 | 0.5553 | 0.7756 | -0.0059 |
| P2-Q10 + V3 locked probability fusion (0.90/0.10) | 0.6433 | 0.6835 | 0.4565 | 0.5340 | 0.7465 | +0.0148 |

The source paired-bootstrap table reports fusion-vs-P2 patient-macro-F1 delta `+0.01480`, interval `[+0.00699, +0.02285]`, over 2,000 resamples and 80 patients. The source does not label the interval level in that table, so it is reported without a confidence-level claim. This interval is conditional on the reported seed-42 data and should not be read as external replication. The audit report lists exact reproduction checks as passed for its fold thresholds and fold F1 values.

## B0/M1 tooling check, seed 42 (2026-06-23)

Two preliminary M1 variants have patient-macro balanced accuracy 0.63 (`m1_feat_HLV`) and 0.62 (`m1_gate_m3`), with patient-macro F1 0.6570 and 0.6533, respectively. These results come from a directory explicitly named `tooling_check`; do not interpret this as a completed model-selection or confirmatory experiment.

## PGC smoke run (2026-06-21)

Aggregate smoke metrics are preserved only to make the timeline complete. The run used one epoch and two splits, and its run metadata has a split-strategy inconsistency. These values are not suitable for model comparison or inference.

## Other attempts

- A10 and A9v11 result summaries were empty; no best configuration or performance result was produced.
- The July 4 four-center NeuroEZ read audit loaded zero LZU/HUP patients and skipped one multicenter row because its source root was missing. This is a data-availability failure, not a negative model result.
- The P23-lite report is `not_admitted`; its accompanying audit says the expected model outputs were missing. It does not establish a performance failure of a fully evaluated model.
