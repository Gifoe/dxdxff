# Server DRE archive scan

- Source: the Windows server's DRE-nips tree. Original paths are stored relative to the source root in the manifest.
- Snapshot labels and file-time ranges are indexed in `dre/SNAPSHOT_TIMELINE.md`.
- Code/document files included: 3,832.
- Candidate aggregate result files included: 1,262.
- Code/document candidates excluded by the safety filter: 608.
- Raw data, `dataest`, patient records, checkpoints/weights, caches, predictions, logs, virtual environments, and Git metadata were not included.
- Aggregate tables were limited to <=2 MB and <=200 rows, with no identifier/path columns, subject-specific source paths, or detected concrete subject IDs/absolute paths.
- Code copies may retain the known server data-root constant; no data beneath it were copied.
- Every included file has SHA-256, source-relative path, size, and source modification time in `SERVER_FILE_MANIFEST.csv`.
