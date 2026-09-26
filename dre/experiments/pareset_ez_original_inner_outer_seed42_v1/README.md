# PaReSet-EZ v1: validation-selected five-fold evaluation, seed 42

This package uses the original frozen **fit / validation / test** membership in each of five folds. Training and standardization use `fit` patients only. The `validation` patients select each model's checkpoint and decision threshold; the disjoint `test` patients are evaluated once after selection. The five test groups cover all 80 patients once. This is the original validation-selected protocol used for the matched 36-D CDEL comparison, **not** the later incomplete flat no-inner/threshold-0.5 rerun.

The exact PaReSet full, retained base, matched PRQ, and matched BCR models were trained on the same adapted records and five-fold membership. The matched CDEL score is `0.8 × PRQ_probability + 0.2 × BCR_probability`, with its decision threshold selected on the corresponding validation fold. The controls use the 36-D adapted input and a finite-gradient variance repair. They are **not** the historical paper's 28-D checkpoints, and their newly matched common schedule is not the original PRQ/BCR branch-specific schedule. See [AUDIT.md](AUDIT.md) and [ADAPTER_DIFF.md](ADAPTER_DIFF.md).

The completed seed-42 outer-test patient-equal means are:

| Method | Macro-F1 | EZ-F1 | Balanced accuracy | EZ-AUPRC |
| --- | ---: | ---: | ---: | ---: |
| PaReSet full | 0.5935 | 0.3733 | 0.6323 | 0.5003 |
| Retained base | 0.5949 | 0.3850 | 0.6351 | 0.4921 |
| Matched repaired BCR | 0.6085 | 0.4119 | 0.6563 | 0.4837 |
| Matched repaired CDEL | 0.6029 | 0.3835 | 0.6492 | 0.5121 |

PaReSet full minus base Macro-F1 is **−0.14 percentage points** (paired 80-patient bootstrap 95% CI **[−1.21, +0.90] pp**). This seed-42 outer test does **not** support an improvement of full over base or CDEL. No additional final sealed cohort was accessed; this is one seed, not a three-seed confirmation. The independent test assessment here was completed before the later flat five-fold rerun, which must not be merged into this table.

Provenance and audit:

- Original outer-test lock SHA-256: `734ccb3b185e1c6b660df71fec7e0e510fdfb470cae81fd9cc9da91198d13dc5`.
- Adapted patient-record SHA-256: `e4cb1347cdc343d6aef288cdcb701c0e7575d71dcdde485e55157465134a77e6`.
- Frozen partition-manifest SHA-256: `fd897fa7eed2c521fd5b14c1ae95d91b5d43b2ae85822317dcda08a878f58278`.
- Validation replay before outer access: 20/20 cells passed; zero threshold/prediction mismatches.
- `results/OUTER_STATUS.json` records `outer_test_used_for_tuning=false`; no new test-based model or threshold choice is made here.

`code/` contains the exact PaReSet model, the supplied EpiLENS supplement, the audited 28-D-to-9-D record adapter, the matched-control numerical repair, frozen development runners, lock preparation, replay validation, and outer evaluator. `results/` contains **only aggregate CSV/JSON/Markdown**. Patient/channel ledgers, source/adapted caches, raw EEG, and checkpoints remain private on the server and are not part of this repository.
