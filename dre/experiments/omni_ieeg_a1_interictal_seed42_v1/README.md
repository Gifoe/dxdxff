# Omni-iEEG × A1 interictal benchmark

Independent seed-42 experiment on the official Omni Task-2 split. The
protocol is fixed in `PROTOCOL_LOCK.json`. The model is the historical A1
architecture, trained from scratch with patient-equal EZ:NEZ=2:1 weighted
BCE. Clinical positive class is strictly SOZ, without resection/outcome
modification.

Server-only waveform and feature caches, checkpoints, optimizer states, and
runtime logs belong outside this repository. Compact metadata audits and
results belong here. The seed-42 run is complete; see `FINAL_REPORT.md` and
`outputs/VALIDATION.json`. Unknown SOZ labels leave all Zurich records outside
the supervised evaluation; this limitation is reported explicitly.
