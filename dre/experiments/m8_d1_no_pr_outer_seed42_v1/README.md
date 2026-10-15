# Full M8 and D1 without Patient Relative — seed42

Completed matched original five-fold Ictal Outer evaluation. See [FINAL_REPORT.md](FINAL_REPORT.md) for results, interpretation and limitations.

- `D1_NO_PR`: the exact previously trained CompactDelta D1 ablation, verified and reused; **not** the generic sklearn MLP feature baseline.
- `M8_NO_PR`: five newly trained full M8 correction models on the frozen matched no-PR D1 bases. Both cross-channel Patient Z sites are absent.
- `D1_PR` and `M8_PR`: hash-verified original frozen reference scores, not retrained or inference-modified.
- Each arm covers 80 patients, 7,635 canonical patient-channels and 256 seizures. Primary metrics are patient means, not pooled channels or unweighted fold means.
- No new EEG extraction, new architecture search, Outer threshold selection or inference-only ablation.

## Contents

- `PROTOCOL_LOCK.json`: prospective ablation/training/evaluation definition.
- `code/model.py`: faithful original M8 feature calculation with optional cross-channel Z for parity testing; default and actual training have it disabled.
- `code/run.py`: pinned upstream readers, exact original M8 trainer, validation selection, freeze, sealed Outer pass and paired bootstrap.
- `tools/verify.py`: independent sklearn metrics and independent patient-bootstrap check.
- `results/OUTER_SUMMARY.csv`, `OUTER_BY_FOLD.csv`, `OUTER_BY_CENTER.csv`: anonymous aggregates.
- `results/OUTER_PAIRED_BOOTSTRAP.csv`: four comparisons, 10,000 patient draws, seed42, fixed thresholds.
- `results/DEVELOPMENT_*.csv`: validation-only development results, 65 appearances / 47 distinct patients per arm. These are not Outer results.
- Other JSON/CSV files: selection hashes, math/gradient/identity checks, freeze, status, confusion totals and integrity hashes.

Raw data, labels, individual patient/channel score ledgers, weights, scaler states, training logs and private file manifests remain on the trusted server. The source contains server-side deployment defaults, but no credentials or patient identifiers.

## Reproduction dependency and reporting notes

This is an additive experiment in the existing repository, not a standalone data release. It intentionally imports the immutable historical M8 trainer, CompactDelta implementation and exact cached data contracts. It refuses changed source/input/checkpoint hashes. Existing trusted-server private artifacts are required.

Run `code/run.py train`, then `code/run.py evaluate`, then `tools/verify.py` under the original training environment. Do not copy into or edit the historical source directories. Existing completed cells and score files are reused only when their bindings match.

The historical trainer's internal objective/arm key remains `M8_MACRO_AWARE` in `TRAINING_HISTORY.csv`; all 71 rows in that file belong to **new M8_NO_PR training**. This preserves the original loss dispatch, not the old PR representation. `MODEL_SELECTION.csv` preserves inherited historical metadata: a `reused_frozen=False` on a D1_NO_PR row describes its original scratch fit in the prior experiment, **not** a new D1 fit here. The actual reuse provenance is explicitly recorded below and in the final report. `RUN_STATUS.json` is the trainer's last Development status; final completion is in `OUTER_STATUS.json` plus `INDEPENDENT_VERIFICATION.json`.

Scientific status: **EXPLORATORY_REPEATED_TEST**. This seed42 ablation reuses previously viewed Outer patients; it is not a new blind confirmation or an independently selected architecture.
