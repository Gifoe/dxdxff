# Raw time and padding provenance

Actual waveform cache audited, not inferred from a filename. Matched coverage:
80 patients, 256 run records, 7,635 canonical patient-channel pairs and 24,995
run-channel incidences. Shape/sfreq/duration, canonical names, sample/run keys,
relative center equality, duplicate timestamps, finite/nonconstant samples,
valid interval bounds and actual waveform lengths were independently checked.

Independent clinical EDF-to-cache onset confirmation remains **0/256**.
Nominal duration midpoint maps cache-relative zero; it is not independently
verified clinical onset. No onset labels, propagation target or EZ-dependent
window selection is introduced. Clinical/outcome fields present in the trusted
historical pickle are ignored; only allowlisted raw identity and sampling fields
are used. Original feature-export `y` is not read by raw preparation.

Three matched records contain boundary padding. All existing candidate 2-s
intervals lie inside their measured intervals; the candidate grid has already
excluded unusable boundaries. There are 1,471,965 valid candidate channel-windows
and no boundary-invalid candidate channel-windows. The earlier E3 failure arose
from a different fixed preictal crop that extended beyond measured data; it does
not warrant deleting this record or its patient here. No padded waveform sample
is used as real EEG. Sampling always checks interval, finite and nonconstant
validity before encoder input, with explicit masked empty cases tested.

These audits support cache-relative predictive comparisons only. They do not
support independent clinical early propagation or causal biomarker claims.
