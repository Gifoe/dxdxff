# Frozen Outer case analysis: execution record

Status on 2026-10-30: **COMPLETE_VERIFIED_FROZEN_OUTER_REPRESENTATIVE_CASES**.

Scientific source and selector were committed as
`b3a1cb4acfa200901312ab3307e07022633d3d53` before candidate computation.
Source archive SHA256:
`f3d6f44c9fad68fb00f8f24ab96966032ef7e4ac644d0c69efce2b866b873bc2`.
Admission passed all 40 sealed prediction cells: SVM No PR/Legacy PR at five
folds; original complete DREChanNet WO_BOTH_PR/FULL at seeds 42/43/44 and five
folds. This DRE model has 11,425 parameters with early FiLM and late Gate;
it is not the later no-guidance model or LOCO. The original 80 patients,
7,635 channels and 1,743 EZ labels, roles, checkpoints and score binding passed.
No fitting, forward pass, calibration, threshold selection or ensemble occurred.

## Preserved engineering failure and recovery

Initial runtime:
`D:\Codex_Research\experiments\drechan_pr_outer_representative_cases_v1_runtime`.
Its private source lock and input binding remain preserved. Application event
1000 at 2026-10-30 11:39:01 +08:00 recorded a native `0xc0000005` access
violation in `python313.dll`. Stderr was empty; no low-memory System event 2004
occurred in the checked interval. The underlying cause was not established.

Synthetic NumPy/sklearn/Torch AUROC/F1 checks passed with and without the
read-only legacy adapter, as did a one-patient diagnostic. The server then
went offline. After user-restored connectivity, process inspection preceded
execution. A foreground compute-only diagnostic completed all 640 endpoints
and parent metric replays, with no selection or export. The native failure was
not reproduced; this does not establish that its root cause was fixed.

Successful foreground execution used a new output directory:
`D:\Codex_Research\experiments\drechan_pr_outer_representative_cases_v1_runtime_run02`.
The original immutable scientific source was executed with only runtime output
paths overridden. A private engineering binding records its SHA and the failed
run's input-binding SHA. Models, scores, thresholds, formulas and selector were
unchanged. The old binding was rehashed unchanged. Foreground owner exited 0.

## Verification, export and outcomes

Server independent PASS: 40 cells, 640 patient-condition rows, 400 change rows,
3,200 endpoint values, all 640 historical metric replays, two cases, 50 case
metric rows. Independent selectors agreed; protected parent hashes unchanged.

Exactly 12 anonymous payloads plus manifest were downloaded. Whole archive SHA:
`7920a35660cd4d6b5ce93ae3a8e07307e60ce3793788621dd7c3b5e3b0e689dc`.
`release_check.py` authenticated the exact allowlist, byte counts, hashes and
privacy fields BEFORE extraction. Local `verify_anonymous.py` reconstructed all
16 case/model/seed/condition groups, every rank, all 50 metrics including DRE
seed means, and all 1,904 Recall@K rows from 1,888 anonymous score rows.
All five synthetic selection tests passed. `build_case_report.py` rechecks all
payload hashes and metrics before generating readable tables and the report.

- CASE-P01: LZU, fold 4, 114 channels / 32 EZ. Nine stable improvement
  candidates; minimum joint median-percentile distance. Two DRE seeds improve
  both selection endpoints; seed42 AUROC declines. Full details retained.
- CASE-N01: HUP, fold 3, 122 channels / 14 EZ. Two tier-1 decline candidates.
  Both models' mean ranking endpoints decline while DRE Macro-F1 improves.

Original identities, contact names, all80 patient-level results and reversible
mappings remain server-private. Only the two authorized anonymous cases have
channel-level exports. Following the user's explicit upload request, the GitHub
release scope is the source, protocol, verification, anonymous two-case tables,
cohort aggregate distributions and Chinese report. Original names, mappings,
all80 individual results and transport archives are excluded.

## Local transport artifacts

The small source transfer ZIP remains after automatic approval review rejected
its deletion without a specific reason; this was not bypassed. The authenticated
result transfer ZIP is retained as a verified transport artifact, outside the
Git deliverable. Unrelated experiments and previous figure previews unchanged.
