# DRE / NeuroEZ archive

This repository is now a DRE-only code and results archive. It is organized as a chronological evidence archive, not as a single runnable release: the server held multiple copied snapshots and experiment branches, so use the source manifest before comparing or reusing any result.

## Archive map

- [`dre/SNAPSHOT_TIMELINE.md`](dre/SNAPSHOT_TIMELINE.md) indexes the dated and named server snapshots.
- [`dre/code/server_e_drive/`](dre/code/server_e_drive/) contains screened code/document snapshots recovered from the Windows server's DRE-nips tree. Files are grouped by snapshot and have short hash-prefixed names to avoid Windows path-length limits; consult the manifest to recover each original relative path.
- [`dre/results/server_e_drive/`](dre/results/server_e_drive/) contains small candidate aggregate tables and reports from those snapshots. The archive filter excludes tables with identifier/path columns, concrete subject IDs, or oversized/patient-level layouts.
- [`dre/server_e_drive/SERVER_FILE_MANIFEST.csv`](dre/server_e_drive/SERVER_FILE_MANIFEST.csv) maps each archived file to its source-relative path, snapshot, modification time, size, and SHA-256.
- [`dre/results/nips_temp/`](dre/results/nips_temp/) contains the previously curated nips-temp result archive, including its own [`timeline`](dre/results/nips_temp/TIMELINE.md) and interpretation limits.

## Interpretation and exclusions

The server snapshots are historical copies, not independent replications. Results from different dates, seeds, splits, or protocols must not be pooled as if they were one evaluation. The nips-temp timeline labels known smoke checks, failed audits, incomplete work, and exploratory ablations separately.

No raw EEG/iEEG, patient records, caches, predictions, per-patient or channel-level ledgers, checkpoints, weights, virtual environments, or Git metadata are included. The scan summary and aggregate exclusion counts are under `dre/server_e_drive/`. The source tree itself was left unchanged.
