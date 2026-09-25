# DRE / NeuroEZ experiment archive

This is a separate, results-only archive of DRE/NeuroEZ material found under the historical `nips-temp` workspace. It is intentionally kept outside the existing ReVA-DLM project directories in this repository; it is not part of ReVA-DLM and its metrics must not be attributed to that project.

The timeline distinguishes exploratory smoke checks, incomplete attempts, failed input audits, and the later seed-42 ablation report. A folder or filename containing “heldout” is not by itself evidence of a confirmatory or final-heldout evaluation; each entry is labeled according to its available protocol and audit artifacts.

## Contents

- `TIMELINE.md` — chronological inventory and status of DRE-related result stages located in `nips-temp`.
- `RESULTS_SUMMARY.md` — compact headline metrics and limitations.
- `summary_tables/` — small, sanitized aggregate tables only. Local paths and patient-level rows were removed or not copied.
- `p2_v3_seed42/` — aggregate seed-42 ablation metrics, report, and aggregate figures.

## Excluded

No checkpoints or model weights, raw EEG/iEEG, serialized patient records, cache files, per-patient predictions, patient-level ledgers, channel-level ledgers, run-argument files with local paths, or cache manifests are included. Large or exploratory intermediate tables with patient-level rows were also excluded. The unrelated `TotalP` EEG/CRCICLR experiment tree was not copied into this DRE archive.

The copied P2/V3 metrics are reported as exploratory/ablation results, not as an independently replicated confirmatory result. The source report itself notes that the fusion was locked after observing the current seed-42 development OOF results.
