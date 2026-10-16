# Raw-only optimization: mechanism evidence and limits

This is a technical evidence ledger, not a new clinical or independent
validation claim. All numbers below are from earlier actually completed public
artifacts. They are not outcomes of the currently running temporal-adapter
experiment. The target is complete 80-patient equal-weight Macro-F1 >0.65.

## Four requested components

The matched Macro-first readout study held the learned encoder fixed, refit the
same head and deleted one aggregation ingredient. It used all 15 legal ownFIT
paths, with 255 assessment appearances / 79 unique patients. Patient-ID-cluster
bootstrap carried repeated appearances together.

| Deleted component | Macro-F1 difference vs matched REFIT | Nominal paired 95% CI | Supported conclusion |
|---|---:|---|---|
| Pre-latent reference | -0.050031 | [-0.070167, -0.029965] | Strong conditional utility for this frozen representation/readout |
| Two temporal phases | -0.009622 | [-0.018222, -0.000827] | Conditional benefit from retaining the two phases |
| Seizure SD beyond mean | -0.002415 | [-0.010424, 0.005268] | Additional spread benefit is not established |
| Patient-wise Z | -0.092557 | [-0.120807, -0.066008] | Strong conditional benefit; numerical conditioning also changes |

These are not four end-to-end retraining proofs or universal necessity claims.
In particular, the SD result does not justify declaring between-seizure spread
useless. The Z result does not isolate a physiological cause from optimization
conditioning. Source:
[paired differences](../raw_macro_f1_head_screen_seed42_v1/results/cpu_bridge_execution_v2/INTERNAL_PAIRED_INTERVALS.csv),
[scope and verification](../raw_macro_f1_head_screen_seed42_v1/RUN_REPORT.md).
The experiment actually fit 105 readouts over 3,105 epochs, rather than filling
these comparisons with untrained or test-selected proxies.

## Earlier temporal-context experiment is not hidden

A preceding information-retention study already tested a longer local
receptive field (535 samples) from scratch. It used **EZ-first**, not the now
clarified Macro-first, training/threshold objective. Actual reported internal
Macro-F1 was 0.582751 for its short/window control, 0.578788 for long/window,
0.577473 for long/pre-normalized and 0.584837 for long/pre/context. It did not
establish a useful Macro gain. Its EZ-optimized oracle is not a Macro ceiling.

The new experiment therefore is not described as the first attempt at longer
temporal interactions, nor as a renamed successful version of that study. It
tests the presently fixed two-waveform Raw body, Macro-oriented warm training,
zero-initialized local interaction residual and equal-984-parameter pointwise
control. Those differences are prospective scientific configurations, not a
claim that a longer receptive field must improve F1. Source:
[actual first-path table](../raw_information_retention_internal_seed42_v1/results/deterministic_execution_v2/FIRST_PATH_MAIN_RESULTS.csv),
[original scope](../raw_information_retention_internal_seed42_v1/README.md).

## Other executed negative and partial results

| Completed study | Actual evidence | Consequence |
|---|---|---|
| Matched Macro readout refit/conditioning | Full80 Native 0.630493; refit 0.627736; ownFIT-selected 0.624314 | A more elaborate readout was not enough |
| Mean vs learned-response SD residual warm adaptation | 15 end-to-end fits / 77 epochs; SD arm selected epoch0 in allfive folds | No retained gain in that protocol; no universal dispersion-irrelevance claim |
| Shared frozen CNN two-waveform fusion | Full80 precompand 0.631432; AUROC/AP gain but Macro difference CI crossed zero | Ranking improved without established Macro improvement |
| Frozen-score decision rules | All9 complete procedures; highest nominal Macro 0.633879; selected policy 0.628030 | Do not adopt a best-Outer arm; simple calibration did not establish >0.65 |

The actual internal precompand AP gain was +0.014757 with nominal paired CI
[0.001063, 0.029497], while Macro gain was +0.000065 with CI crossing zero.
That legal internal mismatch motivated a calibration control; it did not prove
that patient-specific thresholds can be learned without labels. The internal
label-using oracle around 0.70 is nondeployable and never used as goal evidence.

Sources:
[readout report](../raw_macro_f1_head_screen_seed42_v1/RUN_REPORT.md),
[dispersion report](../raw_latent_dispersion_macro_seed42_v1/RUN_REPORT.md),
[waveform report](../raw_precompand_macro_seed42_v1/RUN_REPORT.md),
[decision report](../raw_patient_decision_macro_seed42_v1/RUN_REPORT.md).

## What remains unproven

No actually completed full-cohort Raw-only procedure in this sequence exceeds
the requested patient Macro-F1 0.65. Neither better AP, a favorable fold,
pooled metrics, an internal oracle, green source tests nor a CI upper limit
above 0.65 can replace that missing result. Source-specific and patient-paired
uncertainty are descriptive because historical Outer and internal paths have
been used repeatedly. Later experiments must not alter any old sealed result
or infer component necessity solely from the failure of one warm optimizer.

The active temporal-adapter protocol has a finite internal gate. If no fold
passes, its actual negative internal results will be published and no new
Outer will be run merely to search for a favorable winner. This is a stop for
that registered experiment, not a declaration that the broader Raw-only goal
has been completed or is impossible.

## Exact public source bytes used for this ledger

SHA256 values are for the currently inspected workspace artifacts; historical
Git text-normalization may differ from raw server-byte hashes. They are not
substituted for private data/model freezes.

- Macro readout `INTERNAL_PAIRED_INTERVALS.csv`: `b5d3b9d3bf917070559a0c8624dbbdc377dbd884e35c8c110b9bf853aea6acf3`
- EZ-first information retention `FIRST_PATH_MAIN_RESULTS.csv`: `6722f4d499982478adb2cca8b6abaad11687f7f01f865c3711ebb92108d8e05a`
- Precompand `OUTER_PATIENT_MEAN.csv`: `031ca22fb2c3f022bb815ac0c10400cf6c5a250af4a843b5a837170c60113a07`
- Decision study `OUTER_PATIENT_MEAN.csv`: `f20cbc93597e3749b26bc8a93d9c11b133bd9f7a5fdfb3d83184937711948acb`
