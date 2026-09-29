# CRST-iEEG: two-benchmark seed-42 experiment

This branch is an in-progress, outcome-sealed implementation. The 0.85 AUROC
targets are engineering targets, not achieved or claimed results. The two
benchmarks instantiate the identical 1,550,103-parameter `CRSTiEEG` class and
train independent weights. Neither test set is used for model design, SSL,
checkpoint selection, or threshold selection.

Phase 0 established exact historical ictal 80-patient/256-record cohort,
channel, canonical-label, and onset alignment. Three original records contain
only 41, 52, or 57 seconds of observed signal in a 60-second tensor; their
non-observed windows are explicitly masked. Source per-record labels can
conflict across seizures, so training uses the frozen historical canonical
patient-channel labels, not a newly derived vote.

Phase 1 synthetic tests check variable physical sampling rate, frequency
availability, padded windows, connectivity symmetry, zero-initial network
bias, and channel permutation. FP32 permutation numerical error on the
synthetic stress case is at most 0.00137 in logit, not bitwise zero.

Private spectral tensors, patient/channel identities, train logs and model
checkpoints stay on the server at `F:\Omni-iEEG\crst_ieeg_seed42_runtime`.
Only aggregate audits, source, locked protocols, metrics and reports may be
committed. Historical ictal comparison uses the original fixed 65-cell,
20-query-repetition, 47-ID VLOO scheme, not an incomparable pooled 80-patient
metric. The official Omni comparison requires 141 train / 96 test patients,
296 train EDFs, official ordered labels and a train-side threshold.

No positive scientific claim should be made until both benchmark evaluations
and required bootstrap/intervention audits are complete.
