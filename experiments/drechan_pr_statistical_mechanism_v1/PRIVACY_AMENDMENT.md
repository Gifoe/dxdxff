# Publication-only complementary-aggregate privacy correction

The numerical study, frozen hypotheses, all80 primary results and32 prespecified
strict79 sensitivity intervals remain unchanged. This correction concerns export,
not statistical selection or recomputation.

Before publishing any numerical results to GitHub, release review identified that
an exact80-person mean and its79-person leave-one-exception counterpart expose
the omitted patient's endpoint through subtraction. The original frozen engine
had placed that complementary sensitivity table in its aggregate intermediate.
Integrity/schema checks alone did not detect this differencing disclosure risk.

The mandatory publication filter therefore retains that exact table on the trusted
server and exports only its qualitative negative conclusion and verification
commitment. All17 other original aggregate/verification payloads are byte-identical.
An additional privacy-review JSON replaces the sensitive complementary table, for
18 final payloads plus a new manifest. The original frozen intermediate and its
hashes remain preserved privately. No individual identities or raw waves were
included in that intermediate, but its complementary means were not suitable for
publication.

The public source records the predeclared sensitivity and its independent checks;
the qualitative finding that it does not support increased transferable T remains.
Neither the80-person primary nor any unfavorable arm, source, estimator, numerical
sensitivity or negative-control result is removed. Only exact complementary79-person
statistics/intervals are withheld for privacy. No result has been recomputed.

The eight frozen scientific files remain byte-identical to the pre-inference
source commit. `sanitize_public_export.py` is a separate release operation, called
by the current entrypoint after the frozen private-intermediate packaging stage.
Its deterministic ZIP entry date is a packaging convention, not an execution date.
Only the filtered archive is eligible for transfer or publication.
