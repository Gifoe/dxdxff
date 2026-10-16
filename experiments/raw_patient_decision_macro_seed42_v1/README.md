# Frozen Raw score decisions, patient Macro-F1, seed42

Target: **80-patient equal-weight Macro-F1 >0.65**. The exact frozen Raw-only
reference is 0.6304930529317453, not its EZ-positive F1 or a pooled score.

This prospective experiment freezes the Native and PRECOMPAND_RESIDUAL Raw CNN
score bodies from `raw_precompand_macro_seed42_v1`. No neural weights, manual
physiological features, D1 branch, data, patient roles, or raw windows change.
The legal internal evidence motivating it is higher ranking AP but flat Macro-F1
in the preceding study. Gaussian posterior/prevalence inference is not used.

Four decision rules are compared: unchanged probability threshold; a raw-margin
quantile threshold; patient-standardized margin threshold; and fixed patient
top-fraction with indivisible equal-score groups. `PROTOCOL_LOCK.json` defines
the finite grids, selection, role separation, precision controls, and final plan.

Internal execution: `run_internal.ps1 -RunTag 01` on the trusted Windows server.
It runs the changed critical-path tests, freezes all 120 CAL configurations and
HELD score cells, then independently verifies calibration and patient metrics.
There are 90 new scalar calibrations and **zero neural fits**. All five procedure
choices are fixed within their own FIT before original Development calibration.

Ranking uses the natural unbounded NEZ margin, with FP32-probability ranking
reported separately as a numerical precision audit. A decision rule cannot
improve rankings of its fixed body. The optional internal per-patient oracle is
label-using and nondeployable; it cannot establish the target or choose rules.

No original Development or Outer outcomes are read by the internal scripts.
Final calibration/inference will be a separately source-frozen extension without
altering these registered sources. Historical Outer has been viewed repeatedly:
all subsequent Outer results are exploratory repeated-test, not independent
confirmation. No best-Outer adoption, threshold rescue, or grid expansion.

Public delivery is limited to code, hashes, anonymous aggregate results and
technical reports. Raw data, identities, truth, checkpoints, individual logits,
decisions, caches and runtime logs stay on the trusted server.

Execution is complete and independently verified: 120 internal cells, 45 final
cells, nine reported procedures, all original 80 patients and valid Raw windows.
**No procedure exceeded patient-equal Macro-F1 0.65.** Highest nominal Outer
point: 0.633879; legal ownFIT-selected policy: 0.628030. Neither replaces the
reference via best-Outer selection. See [RUN_REPORT.md](RUN_REPORT.md),
[all metrics](results/final_stage_v1/OUTER_PATIENT_MEAN.csv) and
[paired CIs](results/final_stage_v1/OUTER_PAIRED_CI.csv).

Final reproduction uses `run_final_execution_v2.ps1`; this execution-only
extension invokes the unchanged scientific stages and corrected reporting
verifier. The original frozen verifier and failed log are preserved. The repair
changes no model/configuration/prediction and repeats no inference. Public
artifacts can be checked with `python verify_public.py` without private data.
