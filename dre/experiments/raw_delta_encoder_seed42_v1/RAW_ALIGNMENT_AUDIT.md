# Raw alignment audit

The completed label-blind Stage A run rehashed the exact Raw cache, original window cache, 88D measurement archive and fixed split. It retained all 80 patients, 256 records and 7,635 canonical patient-channels. Compound run/sample/seizure keys and normalized channel names, not positional channel matching, identify the source slices.

All 1,471,965 valid channel-windows (15,074 record-windows) were compared against their original 500-sample float32 Raw slices with exact equality. There are 24,995 observed record-channel incidences and 272 missing incidences. Missing seizure channels are not observations; unavailable/padded entries cannot enter baseline, window means, seizure statistics or CNN inputs. All record-channels used have a genuine pre-reference window. The cohort was not selected by fidelity or outcome.

Stage C additionally reverified every valid sample of the existing waveform-only banks against the same compound Raw source and the newly verified timestamps/masks. This proves their record and channel order rather than assuming the old bank's tensor position is its identity. No new raw acquisition or EDF re-extraction occurred.

`RAW_COVERAGE.json`, `RAW_DYNAMIC_ALIGNMENT.json` and `VALID_RANGE_METADATA_UNITS.json` are actual-run aggregates. Feature-cache valid sample bounds use 1,000-Hz metadata while Raw uses 250 Hz; integer bounds disagree in all records but must not be compared without units. After conversion to seconds, median differences are zero and maxima are 0.001 s for both interval endpoints. This does not independently certify onset clocks or the filter chain.

Private channel/record identities, per-patient banks, masks, exact slice ledgers and waveforms remain on the trusted server. No new Outer inference or clinical label use occurred during alignment.
