# A0 + PMA, seed42 — development complete, continuation gate failed

**Terminal: `DEVELOPMENT_GATE_FAILED_PMA_INCREMENT_NOT_ESTABLISHED`.**
No new outer evaluation was performed; no B1/B2 outer access occurred.
All four fixed configurations, five-fold development selection, paired
bootstrap and frozen mechanism interventions were completed on the server.
No feature extraction, label change, additional configuration or rescue
experiment was performed.

## Primary development performance

Patient-equal within each original validation fold, then equal across the
65 patient-fold appearances (13 per fold), representing47 unique patients.
These are **validation-selected development** results, not outer TEST.

| Model | Params | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| B0: exact source-free A0 | 8,817 | 0.710667 | 0.518235 | 0.638080 | 0.431198 | 0.686550 |
| B1: A0 w/o LayerNorm | 8,641 | 0.736420 | 0.550088 | 0.641982 | 0.441803 | 0.757306 |
| B2: equal-parameter residual | 15,185 | 0.710281 | 0.517418 | 0.636782 | 0.425711 | 0.692423 |
| B3: A0 + PMA | 15,185 | 0.711474 | 0.526137 | 0.639305 | 0.432845 | 0.719243 |

| Model | Top1-EZ | Balanced accuracy | Sensitivity | Specificity | NEZ-F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.584615 | 0.665436 | 0.465904 | 0.864968 | 0.844961 | 0.773174 |
| B1 | 0.707692 | 0.677646 | 0.500859 | 0.854432 | 0.842161 | 0.769958 |
| B2 | 0.584615 | 0.662529 | 0.452118 | 0.872940 | 0.847852 | 0.775765 |
| B3 | 0.630769 | 0.667614 | 0.467630 | 0.867599 | 0.845765 | 0.774442 |

Each arm/fold uses its own legally selected checkpoint and threshold. This
experiment did not replace the original Macro-F1 checkpoint selection with
AUROC selection. B0 frozen checkpoint/scaler/protocol identity and validation
selection were replayed exactly. Original A0 outer values in the task are
historical context, not a comparison against the table above.

The declared secondary aggregation first averages appearances within each
identity and then averages47 IDs: AUROC B0/B1/B2/B3 =
0.721746/0.745147/0.721704/0.721521. These numbers are not substituted into
the primary gate.

## Matched effects and uncertainty

10,000 seed42 bootstrap draws resample47 unique patient identities with
replacement, retaining all their validation appearances. No channels or
65 appearances are independently resampled. Thresholds are not reoptimized.
Intervals are unadjusted percentile95% and describe development evidence,
not fresh-test significance or selection-corrected uncertainty.

| Comparison | ΔAUROC [95% CI] | ΔAUPRC [95% CI] | ΔMacro-F1 [95% CI] | ΔEZ-F1 [95% CI] |
|---|---|---|---|---|
| B3 − B0 | +0.000807 [−0.006380,+0.007362] | +0.007902 [−0.001465,+0.017745] | +0.001225 [−0.007568,+0.009940] | +0.001647 [−0.012327,+0.015450] |
| B3 − B2 | +0.001193 [−0.010249,+0.011632] | +0.008720 [−0.006107,+0.023323] | +0.002523 [−0.008640,+0.013922] | +0.007134 [−0.010336,+0.024073] |
| B3 − B1 | −0.024947 [−0.044397,−0.006735] | −0.023951 [−0.048206,−0.001695] | −0.002677 [−0.020512,+0.015308] | −0.008958 [−0.038425,+0.019670] |
| B2 − B0 | −0.000386 [−0.006570,+0.005462] | −0.000817 [−0.009670,+0.009145] | −0.001298 [−0.008704,+0.006327] | −0.005487 [−0.017445,+0.006795] |

## Fold and center consistency

| Fold | B0 AUROC | B1 AUROC | B2 AUROC | B3 AUROC | B3−B0 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.697122 | 0.737973 | 0.697845 | 0.694170 | −0.002952 |
| 2 | 0.716466 | 0.751103 | 0.715265 | 0.719349 | +0.002883 |
| 3 | 0.696716 | 0.706962 | 0.697879 | 0.696239 | −0.000476 |
| 4 | 0.751080 | 0.771005 | 0.752938 | 0.754776 | +0.003696 |
| 5 | 0.691951 | 0.715059 | 0.687479 | 0.692835 | +0.000885 |

PMA improves AUROC in3/5 folds, not the required4/5. B1 exceeds B3 AUROC
in every fold; this is diagnostic evidence, not authorization to run a new
B1 outer test.

| Center | Appearances / unique IDs | B0 AUROC | B3 AUROC | B3−B0 |
|---|---:|---:|---:|---:|
| HUP | 28 /21 | 0.790285 | 0.786432 | −0.003853 |
| LZU | 14 /11 | 0.627506 | 0.642134 | +0.014627 |
| Multicenter | 11 /9 | 0.796512 | 0.801960 | +0.005448 |
| Pediatric | 12 /6 | 0.543221 | 0.534524 | −0.008697 |

Center effects are not consistently positive. Centers are descriptive
subgroups, not independently tuned or selected pipelines. All11 metrics
per center and fold are in the corresponding aggregate CSVs.

## Frozen mechanism tests

Same B3 checkpoint and threshold; no retraining/reselection.

| Intervention | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|
| B3 full | 0.711474 | 0.526137 | 0.639305 | 0.432845 | 0.719243 |
| B3 statistics shuffled within patient | 0.707127 | 0.502250 | 0.619545 | 0.394711 | 0.668619 |
| B3 residual disabled | 0.707430 | 0.511753 | 0.617087 | 0.385780 | 0.694893 |

Full minus shuffled AUROC = +0.004347 [0.001130,0.007889]; Macro-F1 =
+0.019760 [0.010309,0.030894]. Full minus disabled AUROC = +0.004044
[0.000548,0.007368]; Macro-F1 = +0.022218 [0.008551,0.037930].

The branch is active: patient-appearance mean correction norm0.179521,
mean correction/hidden norm ratio0.042858, near-zero-channel fraction0,
mean residual-element std0.025667. Between-appearance std of patient mean
correction norm is0.149538. Disabling changes2.53% of channel-pair rank
relations on average; shuffling changes3.39%. Patient-appearance Top1
changes are15.38%/33.85%, respectively. These are equal-patient averages,
not pooled-channel percentages. Mean correlation of trained B3's backbone
logit with projected residual logit is0.657539; this is not correlation
with the independently trained B0 predictions. Full distribution/fold
statistics are in `PMA_CORRECTION_AUDIT.csv`.

The trained classifier depends on its correctly aligned magnitude branch.
However, an intervention removing part of a jointly trained network is not
the same comparison as training a simpler baseline. Dependence cannot be
equated with incremental predictive benefit. B3−B2 has no supported AUROC
or AUPRC gain and B1 has better ranking metrics. The magnitude-specific
**improvement** explanation is not validated.

## Frozen continuation decision

| Criterion | Observed | Requirement | Result |
|---|---:|---:|---|
| B3−B0 AUROC | +0.000807 | ≥+0.010 | FAIL |
| B3−B2 AUROC | +0.001193 | ≥+0.005 | FAIL |
| B3−B0 AUPRC | +0.007902 | ≥0 | PASS |
| B3−B0 Macro-F1 | +0.001225 | ≥−0.005 | PASS |
| B3−B0 EZ-F1 | +0.001647 | ≥−0.005 | PASS |
| Positive AUROC folds | 3/5 | ≥4/5 | FAIL |
| Engineering/integrity/numerical checks | PASS | All pass | PASS |

**Gate failed. Outer TEST was not accessed for PMA.** No new outer rows are
created, and no architecture/hyperparameter/threshold changes follow these
results. The machine-readable mechanism terminal documents an exploratory
correspondence-dependence signal; it does not override this failed predictive
continuation gate or establish a validated main model.

## Answers to the ten research questions

1. **AUROC over A0:** point gain+0.000807, CI crosses zero; no supported gain.
2. **AUPRC/Macro-F1:** tiny positive point differences, both CIs cross zero.
3. **Over equal-capacity residual:** no supported improvement; AUROC gain is
   only+0.001193 versus the predeclared minimum+0.005.
4. **Over removing LayerNorm:** no. B3 trails B1 AUROC by0.024947 and AUPRC
   by0.023951; their development intervals exclude zero in B1's favor.
5. **Consistency:** no.3/5 folds and2/4 centers improve AUROC versus A0.
6. **Magnitude usage:** yes, nonzero residuals and disabling losses show use,
   but do not prove usefulness relative to separately trained controls.
7. **Correct correspondence:** shuffling reduces frozen B3 metrics and
   changes ranking; paired AUROC/AP/Macro-F1 intervals favor correct alignment.
8. **Magnitude versus capacity attribution:** model dependence is present,
   but incremental superiority over B2 is unsupported; clinical magnitude
   relevance is not established by the intervention.
9. **Continuation gate:** no; the two AUROC-gain and fold-consistency criteria fail.
10. **Main ictal model for JBHI:** insufficient evidence. PMA adds6,368
    parameters without established ranking/F1 benefit over A0 and underperforms
    a simpler diagnostic control. It should not replace A0 as the main model
    on this evidence. No automatic next experiment is run.

## Verification and provenance

All123 engineering checks passed. Independent sklearn recomputation checked
390 patient-model/intervention cells and66 metric intervals; maximum metric
drift2.22e-16. Hash checks verify all selected models, scalers, thresholds,
feature/split/order identities and sealed predictions. Original scientific
runtime: PyTorch2.11.0+cu128; analysis runtime: PyTorch2.11.0+cu128,
NumPy2.3.5, sklearn1.7.2. The analysis environment only verifies frozen scores.

An initial engineering-test launch exited natively; Windows Application
Error recorded `python312.dll`/`c0000005`. A same-code unit-test retry passed
all123 checks before scientific training. A detached training launch produced
no epochs/checkpoints and exited without a captured cause; it was not counted
as a completed scientific run. The foreground same-command trainer completed
all15 new cells, with no model/data/loss/runtime/selection changes and no
completed epoch discarded. All existing files were preserved.

Training/evaluation artifacts and detailed patient/channel records remain
private on the authorized server. Only source, hashes and anonymous compact
aggregates are included here. No installed skills were used.

## Subsequent B1-only outer extension

The above report is the completed PMA-development snapshot. Afterwards, the
user explicitly authorized frozen B1 outer evaluation, recorded separately
in `../a0_b1_no_layernorm_outer_seed42_v1/FINAL_REPORT.md`. B1 outer AUROC
is0.740841 versus reused B0's0.739768; paired difference+0.001073 with95% CI
[−0.016461,+0.017722]. This does not establish a generalizable B1 ranking
gain. The original PMA gate remains failed; B2/B3 outer remains unperformed.
