# DRE / NeuroEZ results timeline

Dates below follow the source files' recorded modification dates in `nips-temp`. “Not uploaded” means there was no usable aggregate result, the run was only a smoke check, or the artifact contained paths or patient-level material that should not be published.

| Date | Stage found in `nips-temp` | What the available evidence supports | Archive status |
| --- | --- | --- | --- |
| 2026-06-06 | `neuroez_c_dry_parser` | Parser/cache exploration only; source patient-record and cache artifacts, no compact performance result. | Not uploaded; source/cache files excluded. |
| 2026-06-15–17 | BioDynFormer source audit and C0 ablation setup | Source metadata audit and C0 manifests/configuration were present, but no C0 aggregate performance table was found. | Not uploaded as results; no performance outcome to report. |
| 2026-06-21 | PGC smoke run | Task-1/Task-2 metrics exist, but the run is explicitly a smoke test: one epoch, batch size 2, and `n_splits=2` despite a `split_strategy=5fold` field. | Compact aggregate metrics included in `summary_tables/PGC_SMOKE_AGGREGATE.csv`, prominently marked non-inferential. Per-patient predictions, cache, and checkpoints excluded. |
| 2026-06-23 | B0/M1 Step-1 ablation tooling check, seed 42 | Two M1 configurations have aggregate patient-macro metrics. The source directory is explicitly named `tooling_check`; this is preliminary, not a confirmatory comparison. | Sanitized two-row table included in `summary_tables/M1_TOOLING_CHECK_SEED42.csv`; machine-specific paths removed. |
| 2026-06-30–2026-07-02 | A10 RankSimple and A9v11 BurstListwise | A10 has an empty smoke summary; A9v11 has an empty grid summary and no selected configuration. | No result tables uploaded; no usable experiment result was present. |
| 2026-07-04 | NeuroEZ four-center read audit | Audit summary reports zero loaded patients for LZU/HUP and one skipped multicenter row. The available CSV records a missing source-root category; the absolute path is intentionally omitted here. | Sanitized status table included in `summary_tables/NEUROEZ_FOUR_CENTER_READ_AUDIT.csv`. This is an input/read failure, not a model result. |
| 2026-07-17–19 | CANE-path sensitivity / P23 regression audit | Sensitivity protocol-audit files and run arguments exist, but no complete model outputs were available. `P23_LITE_REPORT` says `not_admitted`; its regression audit lists all expected model outputs as missing. | Sanitized P23 status included in `summary_tables/P23_LITE_STATUS.json`; no run arguments, manifests, or patient-level audits copied. |
| 2026-07-20 | P2-Q10 / V3-QBC seed-42 ablation | Aggregate results cover 80 patients and 7,635 channels across five folds according to the source report. Reproduction checks are reported as passed. The report labels the locked fusion exploratory and ablation alternatives as not for model selection. | Aggregate CSVs, source report, and aggregate-only figures included under `p2_v3_seed42/`. Patient-level/channel-level tables and local-path configs excluded. |

## Reading order

For the most complete quantitative result, start with `p2_v3_seed42/report/P2_V3_AAAI_ABLATION_REPORT.md`, then inspect the accompanying overall, by-fold, by-center, and paired-bootstrap aggregate CSVs. Earlier entries are smaller development checks or incomplete attempts and should not be combined with the seed-42 result as if they formed a single homogeneous evaluation.
