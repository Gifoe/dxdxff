# Identity, valid support, and independent-time limits

Actual exact main-cohort intersection:80 patients,256 seizure records,
7,635 canonical patient-channels,24,995 seizure-channel incidences. Original
five patient-held-out memberships and A0 feature hashes are unchanged.
Compound record keys and normalized channel names prevent duplicate run-name
or list-order guessing. No patient/channel was excluded by signal power.

All Raw arrays are250Hz/60s. The exact stored feature centers and valid-start/
valid-length fields yield15,074 fully valid2s record-windows and1,471,965
channel-windows. No padding participates. The feature-center grid itself
contains no invalid retained windows; a separate nominal59-window grid finds
30 invalid windows across3 padded records:27 LZU,3 Multicenter. The user-named
historical27-window ledger was not recovered and is not reconstructed by
count alone. This older stricter audit limitation is **not** substituted for
FCPA-LF's95% source coverage criterion.

Candidate4s anchors[-28,-24),...,[-4,0) and[0,4),...,[24,28) were checked
geometrically against valid sample intervals. Every canonical channel has
at least one record with two full preictal and two full post-onset candidate
windows. This does not mean a verified LF24 feature bank exists. No4s
spectral features or preictal-reference values were extracted.

Independent sidecar event-time corroboration covers162/256 records and
4,592 canonical channels. LZU/pediatric source files are EDF+D; no TAL clock
verification was performed. Presence of onset fields in a cache is not an
independent clinical onset check. All original identifiers and per-record
matching details stay private.

After continuous-header timing, header-declared bandwidth and valid support
requirements, the documented candidate subset has925 canonical channels,
all from Multicenter:12.1153% overall, not the required95% across four centers.
This is a coverage diagnostic, not a restricted replacement cohort. All80
patients and7,635 channels remain in the failed-gate report.
