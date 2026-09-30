# PRiSM-EZ seed-42 two-benchmark experiment

`PRiSM-EZ` is a 75,473-trainable-parameter patient-relative spectral mixer. It
receives one 68-D token per 2-s/1-s-hop channel window: the validated A1 36-D
physiology descriptor and a new fixed 32-bin log-periodogram sketch. A
label-free average-tie rank across the signal-valid channels gates the absolute
token before two depthwise temporal-mixer blocks. Time and record quantiles are
pooled before a single patient-context classifier.

The two benchmarks instantiate exactly the same topology with separate weights.
The only dataset-specific behavior is the data-provided physiology reference,
native sampling rate, frozen split, and official aggregation unit. There is no
raw deep encoder, attention, recurrence, graph, score fusion, label-dependent
rank, test-time fitting, or final-logit normalization.

The runtime feature cache, checkpoints, private patient predictions, and raw
signals are deliberately excluded from Git. Public artifacts are compact audits
and aggregate metrics only. Formal outer/official evaluation is prohibited
until all development checkpoint and threshold selections have been frozen.

The historical A1 Ictal number is VLOO development and must not be described as
a paired comparison to a later outer evaluation. Omni official outcomes have
previously been viewed, so any formal test result is exploratory repeated-test
evidence, not blind confirmation.

## Stopped scope

The user stopped the experiment after all five Ictal train/validation folds and
the historical validation-only VLOO summary completed. PRiSM-EZ achieved Ictal
AUROC `0.745564`, AP `0.577895`, and Macro-F1 `0.601144`. It did not exceed the
predeclared A1 AUROC gate (`0.746382`), although AP was preserved. Omni
train/inner-validation was not started, and no outer or official test was
accessed. This is an incomplete two-benchmark experiment, not evidence for a
two-benchmark success/failure terminal.

See `FINAL_REPORT.md`, `ICTAL_VALIDATION.csv`, and
`ICTAL_VALIDATION_SCORE_FREEZE.json`. Private score grids, patient identifiers,
checkpoints, feature caches, and runtime logs remain off GitHub.
