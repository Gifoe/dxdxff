# PC-CNN seed42: two-benchmark experiment

PC-CNN is one official-style RawCNN morphology backbone with a 36-D
physiology FiLM at the TimeConv stem output and one late, permutation-equivariant
patient-channel context block. Ictal and Omni use the same Python class and
parameter shapes, with separately trained weights. There is one classifier;
there is no A1 score fusion, router, SSL, connectivity graph, or final score
normalization.

The frozen source references and split/cache hashes are in `PROTOCOL_LOCK.json`.
The official CNN source is the pinned Omni-iEEG revision
`57c22a75a59b5c3a98006806ad42000f6a3fa5b6`, file
`omni_ieeg/channel_model/channel_model_train/model/cnn.py`, SHA-256
`c493b43ec025f74153f0ada3041e6640bfa07beac58f058139ad5122da043f95`.
The script verifies this hash before import. Raw iEEG, derived training tensors,
checkpoints, validation patient/channel predictions, and all test data remain
outside Git.

Two zero-output residual projectors are preserved. The user-approved effective
gate initialization is `0.02`, which keeps step-zero logits identical to
RawCNN but permits nonzero projector gradients. Synthetic tests, real fit
sample identity, architecture/parameter audits, and a one-step Stage-B BN
audit are included. These are engineering audits, not performance results.

The matched RawCNN and PC-CNN training entry points read only frozen
fit/inner-validation roles. `run_train_pipeline.py` runs one GPU job at a time
and resumes existing per-epoch checkpoints. A separate `freeze_before_test.py`
hash-locks all selected models and the Omni validation-derived numeric
threshold before `prepare_omni_test.py` can open official test data.

The historical A1 ictal `0.746382` AUROC is a development VLOO fixed-query
number, not a true outer-test estimate. See `METRIC_PROVENANCE_NOTE.md` before
comparing it with PC-CNN results. Official Omni test outcomes have been viewed
in preceding project work, so this new evaluation is exploratory repeated-test.

Current repository status: train-side implementation and audits are present;
the final two-benchmark result tables will only be added after the frozen
train/validation and test workflows complete and are verified.
