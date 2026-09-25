# Server snapshot timeline

Source: the Windows server's DRE-nips tree. The archive retains separate snapshots rather than merging versions. Dates in snapshot labels follow source folder names; the manifest's UTC modification times are the per-file provenance record and can extend beyond a folder label.

| Snapshot label | Archived files (code/docs + aggregate candidates) | Source-time evidence | Original source folder |
| --- | ---: | --- | --- |
| `legacy` | 8 | 2026-03-27 | `pipeline/` |
| `early-5fold` | 11 | 2026-03-30 | `new-pipeline/5fold/` |
| `new-5fold` | 12 | 2026-03-30–2026-04-01 | `new-pipeline/new-5fold/` |
| `simple-baseline` | 12 | 2026-04-03–2026-04-22 | `new-pipeline/simple_baseline/` |
| `new-nips` | 8 | 2026-04-10–2026-04-16 | `new-pipeline/new-nips/` |
| `oral-nips` | 15 | 2026-04-20–2026-04-21 | `new-pipeline/oral-nips/` |
| `TeChEZ-mat-branch` | 16 | 2026-04-20–2026-04-22 | `new-pipeline/TeChEZ_mat_branch/` |
| `2026-04-24` | 16 | 2026-04-23–2026-04-27 | `new-pipeline/new-4-24/` |
| `2026-04-27` | 13 | 2026-04-23–2026-04-27 | `new-pipeline/new-4-27/` |
| `2026-04-28` | 30 | 2026-04-28 | `new-pipeline/nips_4_28/` |
| `2026-05-07` | 22 | 2026-04-28–2026-05-07 | `new-pipeline/new-5-7/` |
| `2026-05-11` | 10 | 2026-05-10–2026-05-12 | `new-pipeline/new-5-11/` |
| `2026-05-18` | 23 | 2026-05-18–2026-05-19 | `new-pipeline/new-5-18/` |
| `2026-05-29` | 85 | 2026-05-29–2026-05-30 | `new-pipeline/new-5-29/` |
| `2026-06-15` | 54 | 2026-06-15–2026-06-17 | `new-pipeline/6-15/` |
| `2026-06-18` | 16 | 2026-06-17–2026-06-20 | `new-pipeline/6-18/` |
| `2026-06-20` | 97 | 2026-06-15–2026-06-21 | `new-pipeline/6-20/` |
| `2026-06-21` | 190 | 2026-06-22–2026-07-07 | `new-pipeline/6-21/` |
| `2026-07-11-latest` | 4,456 | 2026-06-22–2026-07-27 | `new-pipeline/7-11/` |

`new-pipeline/new-4-24`, `new-4-27`, and similar names are folder labels, not proof that every contained result was produced on that date. The `2026-07-11-latest` label denotes the latest consolidated source folder; some files in it were modified through July 27.

Other generated names such as `5fold`, `new-5fold`, `new-nips`, `oral-nips`, and `TeChEZ_mat_branch` are retained with their labels because no single date can be inferred from the directory name alone. Exact file timestamps remain in the manifest.

## Scan totals

- 3,832 screened code/document files.
- 1,262 candidate aggregate result files.
- 608 code/document candidates excluded by the safety filter; the published exclusion table contains counts only, not source paths.
- Raw data, `dataest`, patient records, checkpoints, caches, predictions, and virtual environments were not archived.

See `server_e_drive/SERVER_FILE_MANIFEST.csv` for file-level provenance and `server_e_drive/SERVER_SCAN_SUMMARY.md` for filter details.
