# Implementation audit (completed)

- Official split and Task-2 inclusion code reviewed at Omni commit
  `57c22a75a59b5c3a98006806ad42000f6a3fa5b6`.
- Data fixed to the public frozen dataset revision
  `73b9c5180a57828ab2a83c040e7e9d112e77b2cc` and verified
  native-signal HDF5 cache `F:\Omni-iEEG\signal_cache`.
- Official filtering yields 151 train and 102 test patients, 399/237 EDFs.
  All 636 have an HDF5 and channel sidecar. Train/test patients are disjoint.
- All 385 Zurich EDFs (250 train, 135 test) have no `good=1` channel with a
  known SOZ 0/1 label. They remain in the cohort audit but cannot contribute
  supervised examples under the locked unknown-label exclusion rule. The
  evaluable cohort is 139 train patients/149 EDFs and 94 test patients/102 EDFs.
- Supervised labels are only `channels.tsv:soz` (1=EZ/SOZ; 0=NEZ).
  Resection and surgical outcome are not used. The original A1 model uses
  NEZ-positive logits, inverted only at the loss/evaluation interface.
- `audit/LABEL_USAGE_AUDIT.json` checks 20 randomly selected official-train
  patients. `audit/FEATURE_DISTRIBUTION_AUDIT.json` checks 5 train patients.
- The exact historical A1 model source hashes match the source commit. A
  synthetic 2-patient/4-record/100-channel/59-window forward and backward
  passed on the server. Trainable parameter count: 27,713.
- Inner validation is a deterministic official-train-only patient split:
  111 fit, 28 validation, stratified by dataset and SOZ presence.
- Train-side label sign audit passed for 20 patients; feature distribution
  sanity passed for 5 patients; a three-patient training smoke passed before
  the official test. Tied-score AP/AUROC bootstrap arithmetic passed against
  an independent expanded-sample calculation.
- The final model/normalizer were frozen and their hashes committed before
  official test scoring. Official test was scored once; no test tuning was
  performed. Independent `code/validate_results.py` checks the output and
  writes `outputs/VALIDATION.json` with `pass=true`.

The official Omni reference channel benchmark constructs some labels using
resection and surgical outcome. This experiment deliberately uses strict SOZ
labels instead. Published Omni scores using that other label construction are
context, not a head-to-head comparable baseline.
