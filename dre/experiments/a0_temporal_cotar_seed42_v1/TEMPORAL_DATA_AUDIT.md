# Source temporal feature audit: PASS before training

Original source cache SHA256 and frozen split SHA256 match the task exactly.
All80 original patients,7,635 canonical patient-channels and256 seizures remain.
No raw waveform cache or waveform was loaded. Canonical names, not array positions,
match channels across seizures. Relative references use the valid simultaneous
record electrode set before restricting outputs to the A0 canonical channel set.

The cache has253 records with59 windows and three with51/40/56 windows. Two-second
windows have approximately one-second intervals (.99999905–1.00000095 seconds).
Relative centers are finite, unique and monotonically increasing in every record.
All selected nine-feature channel windows are finite. The fixed feature order is
stored in PROTOCOL_LOCK.json and TEMPORAL_COVERAGE_AUDIT.json.

Nominal crop coordinates must not be inferred as acquisition start+30 when a
record has left padding. The audit reconciles annotation onset, actual acquisition
start, metadata valid-start extent, nominal60-second duration and actual relative
centers. It rejects a window unless its entire2-second support lies in the valid
source range. Twenty-seven boundary windows in the full source ranges are invalid
(many are at the final60-second boundary); their masks are false. No padding is
counted as signal evidence. Source validity metadata, not independent EDF replay,
supports this decision. Clinical annotation accuracy was not independently verified.

Every unique canonical patient-channel has at least one seizure with >=1 valid
pre, >=2 valid early, and >=1 valid later window. Thus unique-channel early-stage
coverage is100%, above the90% gate. Across25,267 canonical seizure/channel slots,
272 missing channel incidences remain masked; record/channel coverage is98.9235%
for all three stage-support requirements. Every seizure has early-stage evidence
for some channels. Neither missing optional stages nor absent channels are imputed
or grounds for removing a patient or seizure.

The masks retain genuine missingness. Zero tensor entries for absent stages are
storage values excluded from normalization, core softmax, means and standard
deviations. At least one seizure with two valid early windows is required before
any nonzero channel residual is allowed. Normalization is fitted on valid FIT
tokens only. A0 preprocessing is unchanged. T1/T2 read the same sealed private
file per fold, giving bitwise-identical tensors and masks.

Public center counts and record-coverage histograms are in results/. The detailed
seizure/channel identity and coverage ledger and temporal banks remain private.
Reading the all-cohort source payload for a label-blind timing audit does not
constitute an outer-test prediction evaluation; source clinical fields exist but
are not features. Formal training sees only sealed FIT/VAL banks.
