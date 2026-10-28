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
