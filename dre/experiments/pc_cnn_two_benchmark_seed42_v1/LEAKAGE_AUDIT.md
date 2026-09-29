# Leakage audit (in progress)

- The PC-CNN protocol lock was written before PC-CNN outcomes and before any
  new official test read. The code verifies hashes for the ictal fold manifest,
  Omni official split, Omni 141-patient inner split, descriptor source, and
  official CNN source.
- Ictal descriptor moments use only each fold's `fit` patients. Omni moments
  use only `inner_train`; there is no center-specific validation/test fit.
- `prepare_omni_train.py` filters the official split to TRAIN only. A separate
  test extractor refuses to run without a complete model/threshold freeze.
- Validation predictions and patient/channel-level rows remain in private
  runtime and are not tracked in Git. Model selection is validation-only.
- Ictal's historical A1 `0.746382` is a development VLOO fixed-query result,
  not an outer-test result. See `METRIC_PROVENANCE_NOTE.md`; an unmatched
  comparison must not be presented as a paired causal gain.
- Historical Omni official test results were viewed in earlier experiments.
  Even a single new frozen test pass here is exploratory repeated-test, not
  blind prospective confirmation.

Current state: final heldout has not been accessed for PC-CNN. This document
must be updated after model freeze and evaluation; it is not a final pass.
