# PaReSet-EZ: flat five-fold patient-level CV (seed 42)

**SUPERSEDED / INCOMPLETE.** The user clarified that the intended comparison keeps the original fit/validation/test workflow used for the matched CDEL evaluation. This flat no-inner, threshold-0.5 rerun was stopped after two completed fold-1 cells; an epoch-level fold-2 full-model resume checkpoint remains on the server. Its partial outcomes are not published, should not be compared with the original-protocol table, and this branch is retained only as an auditable record of the abandoned protocol. The completed original-protocol code and aggregate results are on [`codex/pareset-ez-original-inner-outer-seed42-v1`](https://github.com/Gifoe/dxdxff/tree/codex/pareset-ez-original-inner-outer-seed42-v1).

This is an **exploratory protocol amendment**, not a new independent final test. The same 80 patients' historical outer-fold results were inspected before this rerun. Do not use this result to claim a prospectively confirmed gain.

Each frozen fold uses the original `fit ∪ validation` patients for training and the original `test` patients for evaluation. No inner split, resplit, validation scoring, early stopping, or outcome-based checkpoint/threshold selection is used. Each model is evaluated at **epoch 45**, with a fixed **NEZ threshold of 0.5**. The five test groups are disjoint and cover all 80 patients once. Fold train/test patient counts: 64/16, 64/16, 63/17, 65/15, 64/16.

Methods: PaReSet-EZ full, its retained base ablation, and the matched 36-D repaired BCR control. All use the same adapted patient records, frozen patient membership, train-fold-only standardization, AdamW (`lr=1e-4`, `weight_decay=1e-3`), dropout 0.4, batch of four patients, gradient clipping at 1, and seed 42. The repaired BCR preserves the original zero-variance forward value while giving finite gradients. Its repair is not an outcome-tuned change.

Priority amendment: at the user's request, complete and report **full versus base** first. BCR is deferred, with any existing files retained. `code/run_flat5_full_base_priority.py` calls the unchanged frozen trainer, resumes existing cells, and aggregates only the 10 full/base cells. The amendment lock is frozen separately; it changes execution order and report scope, not epochs, model, threshold, data membership, or other scientific rules. Priority amendment SHA-256: `b0a52f82486e767c6d3c4b3f8ea41bdce61f6ce6bd223963a836b32df9ce2288`.

On the original Windows server, the run is launched with `code/run_flat_5fold.py prepare ...` followed by `code/run_flat_5fold.py run ...`. The required flags are `--supplement`, `--model`, `--repair`, `--data`, `--manifest`, and `--output`; see `--help`. The runner validates all 80 patient memberships, writes a SHA-256 protocol lock before evaluating test folds, and supports epoch-level resume. The exact run uses the server paths in the protocol lock. The private adapted cache, patient-level predictions, logs, and model checkpoints are deliberately not published.

The lock records these input hashes:

| Input | SHA-256 |
| --- | --- |
| Adapted patient records | `e4cb1347cdc343d6aef288cdcb701c0e7575d71dcdde485e55157465134a77e6` |
| Frozen partition manifest | `fd897fa7eed2c521fd5b14c1ae95d91b5d43b2ae85822317dcda08a878f58278` |
| Flat-five-fold runner | `a4815692a74702be7d21366c0280f8d35c1ae8ed24575f90bd4ac2079386c561` |

The full local protocol lock SHA-256 is `340d7ec199eb590f68119b18f32051e59b04fd183abdb097a4844557cc5ef927`. The code under `code/epilens` is the exact supplied supplement version used in the run; `code/build_patient_records.py` records the cache adapter. Only compact aggregate outputs belong in this repository. This amended result must not be numerically merged with earlier inner-selected results as if they were the same protocol.
