# A0/FCPA low-frequency data feasibility — strict no-training audit

Audit delta1–4Hz, theta4–8Hz and alpha8–13Hz without modifying the frozen
A088D interface, preprocessing, cohort or test artifacts. No ML training,
classification metrics, outcome-driven selection or new outer evaluation.
Only beta13–30Hz is included as a signal-comparison reference; no high-gamma
80–150Hz extraction is attempted from250Hz raw data.

The trusted-server workflow resolves actual cache/EDF identities, freezes
source hashes and diagnostic thresholds, audits only complete valid2-second
windows, and compares the same channels/times against accessible EDFs with
fixed filter controls. All individual identifiers, full source paths,
waveforms and record-level signal diagnostics remain private.

## Execution

```text
python code/discover.py
python tests/signal_contract.py
python code/audit.py freeze
python code/audit.py raw
python code/audit.py edf
python code/audit.py finish
```

Requires original trusted private artifacts and the existing Windows analysis
environment. See `PROTOCOL_LOCK.json` and `INPUT_REFERENCE_HASHES.json`.
Historical reference documents are staged privately on the server, verified
by hashes; they are not replaced by guessed defaults. `DATA_AUDIT.md` records
actual measurements versus historical statements, and
`EXTRACTION_PROVENANCE.md` distinguishes current source logic from proven
cache-generation provenance. No installed skills are used.

The second raw cache is a476-record/149-patient superset; its256 main-cohort
waveforms and time/channel metadata are byte-identical to the501-record/
155-patient main raw cache. It is not an independent low-frequency source.
