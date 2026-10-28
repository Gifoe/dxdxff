# PR ranking, robustness and seizure sufficiency

Supplementary ictal analyses on the existing cohort. `protocol.md` fixes models,
metrics, comparison families and seizure sampling before new analysis.

Execution: `scripts/run_abc.py` for A-C; D has separate frozen-inference scripts.
Server inputs stay in their canonical read-only locations under
`D:\Codex_Research`. New runtime:
`D:\Codex_Research\experiments\drechan_pr_ranking_robustness_v1_runtime`.
Only anonymous release payloads enter local `results/`; patient CSVs and
predictions remain in the trusted runtime's private directories.

The source is committed before execution. A-C must independently verify all
original baselines and all paired calculations. D must reconstruct the All
condition before interpreting seizure-number differences. Incomplete stages
are always labelled pending; A-C completion does not stand in for D completion.

## Verified completion

A-D completed and independently verified on 2026-10-29 (Asia/Shanghai).
A-C retain all 80 patients/7,635 channels. D uses the same 57 eligible patients
and 5,735 common channels at every K; 128 non-common channels were removed from
the eligible cohort without removing any eligible patient.

Frozen scientific commits: A-C `3eac5b44073a9c18de54d74f7e2f32b1f08f2a0f`,
D `4d45a5b02524e4c0eb2d01c45454e780e261bfb9`. In the trusted environment:

```
python scripts/run_abc.py --source-commit 3eac5b44073a9c18de54d74f7e2f32b1f08f2a0f
python scripts/run_d.py --source-commit 4d45a5b02524e4c0eb2d01c45454e780e261bfb9 --stage all
```

Use the frozen scientific checkout for an existing runtime's resume; generated
reports and later archive-checking utilities do not replace that source lock.
The parent experiment trees, checkpoints, source/data hashes and private seals
are checked at admission and remain read-only.

Validate and extract only the anonymous allowlist, then format:

```
python scripts/release_check.py --package <public_abcd_export_v1.zip> --extract results/public_abcd
python tables/build_tables.py --public results/public_abcd
python figures/build_figures.py --public results/public_abcd
python logs/build_report.py --public results/public_abcd
python logs/verify_public.py results/public_abcd
```

`report.md` answers the scientific questions. `manifest.json` links checkpoint,
score and archive hashes. `tables/` contains Markdown/LaTeX and `figures/`
contains editable PDF/SVG and 600-dpi PNG. `logs/SANITY_CHECKS.md` records actual
coverage, exclusions, numerical parity and execution deviations.

Per-patient endpoints and paired differences remain on the trusted server in
`runtime/private/patient_endpoints.csv`, `patient_paired_differences.csv`,
`patient_budgets.csv`, `D_repetition_endpoints.csv`, `D_patient_endpoints.csv`,
and `D_patient_paired_differences.csv`. None is included in the local release.
The separate three-model/seven-condition normalization experiment is also
complete; see its `results/RESULTS_ZH.md`.
