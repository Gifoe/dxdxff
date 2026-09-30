# PRiSM-EZ seed42 — user-stopped partial report

| Model | Params | Benchmark | Scope | AUROC | AP | Macro-F1 | MRR | Top1 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| A1 | historical | Ictal | historical VLOO reference | 0.746382 | 0.576743 | 0.620810 | 0.740038 | 0.654771 |
| PRiSM-EZ | 75,473 | Ictal | validation-only 65-target fixed-query VLOO | 0.745564 | 0.577895 | 0.601144 | 0.756732 | 0.669511 |
| PRiSM-EZ | 75,473 | Omni | not run | — | — | — | — | — |

The experiment was stopped at the user's request after Ictal development. All
five frozen folds and the target-excluded historical VLOO selection completed.
No outer or official test was accessed. Omni train/inner-validation was not
started, so this report cannot assign a two-benchmark scientific terminal.

## Ictal result

PRiSM-EZ missed the primary AUROC gate by `0.000818` (`0.745564` versus
`0.746382`). AP was preserved and increased by `0.001152`, but Macro-F1 fell by
`0.019666`. MRR increased by `0.016694` and Top1 by `0.014740`. This is near
AUROC parity, not a demonstrated improvement over A1. The predeclared combined
minimum gate therefore failed.

Additional Ictal metrics were EZ-F1 `0.374673` and balanced accuracy `0.645805`.
The values are means over the exact 65 target cells and 20 deterministic query
repetitions, using target-excluded epoch and threshold selection. Ranking
metrics ignore one-class queries through the established NaN semantics.

## Runtime and implementation

The exact trainable parameter count is `75,473`, below the 100K gate. The
initial rank implementation caused thousands of host/device synchronizations
and was not a legitimate measure of model cost. After mathematically equivalent
batching, folds 2–5 averaged `4.55–4.99` seconds per epoch and peak allocated GPU
memory was at most `278,219,776` bytes. Numerical equivalence checks bounded the
maximum logit difference at `9.43e-7` and parameter-gradient difference at
`1.79e-7`.

## Missing by explicit stop

Omni inner validation, architecture freeze, formal Ictal/Omni evaluation,
patient bootstrap against formal baselines, rank/spectral interventions, and
conditional ablations were not run. Empty metrics are not imputed. The exact
terminal is `INCOMPLETE_USER_STOP`, not `PRISM_ROUTE_FAILED`.

Private patient/channel predictions, patient identifiers, caches, checkpoints,
and runtime logs are excluded from the repository. The compact score-freeze
audit contains hashes only. `FINAL_HELDOUT_ACCESSED = NO`.
