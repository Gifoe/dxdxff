# Omni-iEEG × A1 interictal benchmark

Independent seed-42 experiment on the official Omni Task-2 split. The
protocol is fixed in `PROTOCOL_LOCK.json`. The model is the historical A1
architecture, trained from scratch with patient-equal EZ:NEZ=2:1 weighted
BCE. Clinical positive class is strictly SOZ, without resection/outcome
modification.

Server-only waveform and feature caches, checkpoints, optimizer states, and
runtime logs belong outside this repository. Compact metadata audits and
aggregate results belong here. `FINAL_REPORT.md` will be written only after
the final checkpoint is frozen and the official test has been evaluated once.
