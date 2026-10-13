# EDF/Raw source provenance: LOW_FREQUENCY_PROVENANCE_UNVERIFIED

Actual cache, feature-order, split and source hashes are in the companion
signal audit; the FCPA data gate binds its private metadata plan and Raw PSD.
All256 original EDF paths resolve and selected header channels match the
frozen cohort. Per-EDF content hashes/paths and identities stay private.

| Center | Records | Header format | Event sidecar time corroboration | Header HP/LP evidence |
|---|---:|---|---:|---|
| HUP |117|EDF+C|117|Unknown for selected channels|
| LZU |72|EDF+D|0|Unknown|
| Multicenter |45|EDF+C|45|HP0; LP124.9375/500/1002.004Hz declarations|
| Pediatric |22|EDF+D|0|Unknown|

Selected EDF channels use uV header units; explicit digital-to-physical
calibration converts them to volts. Header-derived native rates span about
249.875–5000Hz and are preserved in the private record ledger. Anonymous
rate/unit/filter distributions are in `EDF_HEADER_SIGNAL_SPEC_SUMMARY.csv`.
250Hz Raw is a separate resampled representation, not the native EDF rate.

The45 Multicenter headers satisfy the predeclared bandwidth bounds at the
**header-declaration level**. Their925 canonical channels are documented
candidates, not independently certified acquisition-device/biological truth.
HP0 plus LP=Nyquist can describe an EDF export; it cannot retrospectively
establish all analog acquisition filters. Blank fields in other centers are
unknown, not evidence of no filtering.

EDF+D declares a discontinuous-record format. The audit has not parsed TAL
record clocks/onset annotations, so it cannot assert actual gap sizes or map
cached onset-relative intervals to physical time for those94 records.
Numerical sample-index replay does not resolve that issue. The completed
source gate explicitly excludes this unsupported physical-time claim.

The Raw payload lacks complete build CLI/filter/reference/resampling config
and generating-source SHA. Recovered logs include checkpoint reuse without
enough upstream provenance. Current10Hz module2 defaults and1Hz data_reader
defaults are distinct possible paths. Neither identifies the exact builder.
See the companion `EXTRACTION_PROVENANCE.md` for source hashes and distinctions.

**No sufficiently verified95%-coverage/all-center source was established.**
This is an evidence gap, not proof that no better source or record exists.
No LF re-extraction source was substituted after observing spectral values.
