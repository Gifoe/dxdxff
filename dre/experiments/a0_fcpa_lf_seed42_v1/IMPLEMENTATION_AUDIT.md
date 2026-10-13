# Implemented stage, not an unexecuted model claimed as complete

Implemented only Stage A admission and anonymous spectral summarization.
The companion data audit actually hashes source/cache/split dependencies,
maps EDF channels and complete valid supports, computes Welch integrals, and
 runs fixed0.5/1/10Hz diagnostic filter controls. It neither loads model
checkpoints nor indexes clinical labels or predictions.

`audit_source.py` inspects the actual256 private pair results and derives
canonical-channel evidence coverage without exporting identifiers. It uses
the task's95% requirement and requires representation from each center.
It reports four-second **candidate support geometry**, not extracted LF24
values. Two valid preictal and two valid post-onset4s windows are the fixed
minimum candidate geometry; this is label-blind and cannot rescue a failed
source provenance or time/acquisition requirement.

`audit_spectral_fidelity.py` reports measured same-window Welch band ratios,
 correspondence and fixed negative-control comparisons. Total1–40 minus
 delta/theta/alpha is retained as a residual proxy: nonaligned native bins
 prevent calling it an exact13–40 integral for every file. The qualifiedV2
 table corrects that interpretation while preserving values. This does not
 create any high-gamma feature or modify A0. These are integrated PSD
 diagnostics, not a measured dense transfer-function or acquisition proof.

Synthetic `gate_contract.py` tests exact95% boundary admission and rejects
missing-center, unverified-source, changed-cohort and invalid-support cases.
`verify.py` checks the actual negative gate, protocol/code binding and absence
of downstream training or outer evaluation.

No torch or sklearn model is imported by the executed FCPA-LF code. The
baseline's original training/source-free implementation was read, not run.
Model parameter counts and epoch-zero identity are **not runtime-verified**
because the mandatory data gate stopped execution before implementation.
There is no claim that B0 was newly reproduced or B1–B5 evaluated.

Header-format inspection identifies94 EDF+D records. The initial sample-index
diagnostics are retained, but timeline-qualified summaries separate them from
continuous-header records; no EDF+D physical-time proof is claimed. These
report qualifications do not change any data or gate. The semantic comment
in the delivered helper clarifies its residual-band limitation; its executed
function AST/math remains identical to the server version.
