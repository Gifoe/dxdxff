# Signal-only engineering ledger

No model, checkpoint, A0 feature export, patient membership, split, label,
classification threshold or sealed test result was changed or evaluated.

1. Metadata discovery initially decoded build logs as UTF-8. The recovered
   logs have UTF-16 BOMs; decoding was corrected, with the initial discovery
   snapshot preserved privately. No filter arguments were recovered after
   the correction. Log names are not interpreted as filter frequencies.
2. The initial EDF metadata probe assumed a physical header label equals
   the channel name exposed by MNE. MNE renames duplicate labels with
   suffixes and excludes annotation channels. That assumption can reject
   valid mappings. A bounded header probe found no NUL padding in the
   examined failed headers; NUL padding is not asserted as the cause.
3. Full-file annotation materialization was slow, and a diagnostic probe
   exited natively without a result. No statistics from that failed probe
   are reported. The replacement uses the exact MNE default duplicate-name
   helper and retains a private physical-header index for each channel.
   Ambiguous mappings still fail rather than being guessed.
4. A bounded 16-bit EDF reader applies header digital-to-physical scaling
   and unit conversion without filtering or implicit mixed-rate upsampling.
   A synthetic EDF with duplicate labels, annotation-channel exclusion,
   three physical units and a record-crossing crop passed waveform parity
   against MNE (`rtol=1e-12`, `atol=1e-18`). This is a parser test, not a
   clinical signal or classifier result.
5. The initial code, tests, EDF plan and pre-PSD freeze are retained privately.
   The revised metadata freeze links their hash and the unchanged Raw PSD
   artifact. AST comparison confirmed the Welch band integration, complete
   valid-window selection and Raw PSD generation functions were unchanged.
   Raw PSD was not recomputed. All scientific thresholds remain the original
   `PROTOCOL_LOCK.json` values.
6. The bounded metadata reader does **not** parse EDF TAL onset annotations.
   An annotation channel's presence is not a verified onset. Independent
   onset coverage is credited only to matched accessible sidecar events.
   Missing independent time confirmation remains a failed availability
   requirement, even when the cached onset permits a diagnostic crop.

The audit does not infer absent equipment-filter information from sampling
rate, nonzero PSD, current source defaults or a matching signal spectrum.

Subsequent FCPA-LF header qualification found94 EDF+D files. The original
all256 index diagnostics remain intact, not claimed as physical-time parity.
Continuous-header and declared-bandwidth summaries are separately reported
in the companion task. The native-rate Welch grids also do not universally
land on every closed band edge; a total-minus-low residual is explicitly a
proxy. Neither qualification changes the stopped gate, Raw PSD, source files
or any model outcome. No source/gate is selected from these diagnostic values.
