# D-MIL Adapter seed42 final report

Terminal: **`STOP_DMIL_TRAIN_GATE_FAILED`**

This is a TRAIN-only five-fold patient-disjoint OOF result. The preregistered gate failed, so no new official TEST evaluation was run, no final adapter was frozen, and no TEST metric is claimed.

## Frozen baseline replay

The already-frozen official TEST predictions replay exactly:

- expected AUROC: `0.7987673466324111`
- replayed AUROC: `0.7987673466324111`
- absolute error: `0.0` (required `<1e-5`)
- checkpoint SHA-256: `442b6a01b60994197fbc0e146f3abab7897b0b71e794259014d3eca663de3852`
- 8,104 labeled EDF-channel units, 96 patients
- CNN rerun: no
- predictions adjusted: no

This replay is provenance for the frozen baseline, not a new TEST evaluation.

## Official TRAIN OOF results

| Variant | New params | AUROC | AP | Macro-F1 | Pathological F1 | BA | Patient-equal AP | MRR | Top1 | NDCG | Delta AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 Mean | 0 | 0.957155 | 0.829912 | 0.870792 | 0.766680 | 0.856619 | 0.871055 | 0.949253 | 0.928 | 0.931264 | +0.000000 |
| V1 Tail excess | 1 | 0.956120 | 0.820689 | 0.866437 | 0.758567 | 0.849447 | 0.862990 | 0.938870 | 0.912 | 0.925483 | -0.001035 |
| V2 Q90 excess | 1 | 0.956171 | 0.821867 | 0.866437 | 0.758567 | 0.849447 | 0.864582 | 0.938853 | 0.912 | 0.926188 | -0.000984 |
| V3 Heterogeneity | 1 | 0.956295 | 0.823954 | 0.866437 | 0.758567 | 0.849447 | 0.866934 | 0.943270 | 0.920 | 0.927986 | -0.000860 |
| V4 Full D-MIL | 3 | 0.956201 | 0.822382 | 0.866437 | 0.758567 | 0.849447 | 0.865855 | 0.938870 | 0.912 | 0.926742 | -0.000954 |

Full D-MIL also reduced AP by `-0.007530`. Its fold-wise AUROC deltas were `-0.000313`, `+0.000366`, `-0.000049`, `-0.000315`, and `-0.001095`; only `1/5` was nonnegative. The gate required delta AUROC at least `+0.003`, nonnegative AP delta, and at least `4/5` nonnegative folds. It failed all three requirements. Full stayed within `0.001` of the best component, but that condition is immaterial because every learned variant was worse than mean pooling.

## Coefficients and mechanism audits

The diagnostic fold medians for Full D-MIL were:

- `beta_T = +0.344322`
- `beta_Q = +0.078531`
- `beta_S = +0.184481`

All three coefficients were positive in all five folds. These are OOF diagnostic coefficients, not final adapter parameters: the failed gate prohibits constructing `FINAL_ADAPTER_PARAMETERS.json` or `MODEL_FREEZE_BEFORE_TEST.json`.

Unconditionally, pathological channels had higher patient-equal tail excess (`+0.131080`, 95% cluster-bootstrap CI `[0.102845, 0.160920]`), Q90 excess (`+0.108010`, `[0.086642, 0.129888]`), and heterogeneity (`+0.113673`, `[0.094029, 0.133426]`). That association did not translate into incremental held-out ranking performance once the CNN mean score was retained.

The distribution features were not near-duplicates of the mean (`r=0.320` to `0.451`), but they were strongly redundant with each other: tail-Q90 `r=0.962` and Q90-heterogeneity `r=0.980`. Within mean-score quintiles, differences were small or unstable; the highest quintile showed patient-equal differences of `+0.001873` (tail), `+0.031774` (Q90), and `+0.044633` (heterogeneity), yet OOF AUROC still declined.

Segment count was not a plausible driver of the result. Counts were almost fixed: 13,082/13,350 units had five segments, 201 had two, and 67 had four. Correlations of count with the three features were `0.045`, `0.041`, and `0.035`. Both estimable count strata had negative AUROC deltas; a high-count stratum did not exist.

## Runtime and complexity

- frozen CNN parameters: `11,302,241`
- D-MIL trainable parameters: `3`
- feature aggregation plus loading 296 frozen TRAIN NPZ files: `1.989 s`
- all 25 fold×variant fits, train-threshold selection, and scoring: `5.344 s`
- Full D-MIL across five folds: `1.734 s`
- three-scalar formula over all 13,350 TRAIN units: median `0.000071 s` (`0.0053 us/channel`)
- added waveform inference cost: none

## Required questions

1. Frozen baseline exact replay: **yes**, AUROC `0.7987673466324111`.
2. TRAIN OOF baseline AUROC: `0.957155`.
3. Full D-MIL OOF AUROC: `0.956201`.
4. OOF delta AUROC: `-0.000954`.
5. Improved folds: `1/5` (using the preregistered nonnegative criterion).
6. Best single component: heterogeneity, AUROC `0.956295`; it still lost `0.000860` to mean pooling.
7. Fold-median diagnostic coefficients: `(+0.344322, +0.078531, +0.184481)` for `(T,Q,S)`; no final parameters were frozen.
8. Sign stability: all three Full coefficients were positive in `5/5` folds, but the performance effect was consistently non-beneficial.
9. Conditional evidence: some Q90/variance differences remained in the highest mean bin, but they did not improve held-out discrimination; the sparse-tail claim is not supported operationally.
10. Redundancy: modest versus mean, severe among the distribution features themselves.
11. Segment-count confounding: unlikely, but count robustness is intrinsically limited because 98.0% of units had five segments.
12. D-MIL TEST AUROC: not run.
13. Above frozen `0.798767`: not evaluated because the TRAIN gate failed.
14. Above published `0.8061`: not evaluated.
15. TEST paired bootstrap: not run.
16. Does evidence support that uniform mean pooling discards useful localization information? **No.** The fixed distribution statistics correlate with label but add no generalizable ranking information beyond the mean.

Final scientific conclusion:

> **Simple score-distribution aggregation is not the bottleneck.** Every preregistered D-MIL variant reduced TRAIN-OOF AUROC and AP relative to frozen mean pooling. Designing more pooling variants from these same five-segment score distributions is not justified by this result.
