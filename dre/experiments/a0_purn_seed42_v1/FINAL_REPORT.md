# A0 + PURN seed42 — complete development and authorized outer extension

## Original required development table

| Model | Parameters | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| B0 Original A0 | 8,817 | 0.710667 | 0.518235 | 0.638080 | 0.431198 | 0.686550 |
| B1 RMSNorm-88 | 8,817 | 0.716300 | 0.517188 | 0.637331 | 0.433727 | 0.701388 |
| B2 Unique-LayerNorm | 8,817 | 0.709542 | 0.521807 | 0.639085 | 0.428739 | 0.701693 |
| B3 PURN | 8,817 | 0.719820 | 0.526426 | 0.638263 | 0.436343 | 0.730679 |

B3 has the highest development AUROC/AP/EZF1/MRR point estimates; B2 has the
highest Macro-F1. Development terminal remains **PURN_DEVELOPMENT_GATE_FAILED**:
B3 AUROC−B0 is0.009153<0.010, AUROC−B1 is0.003519<0.005, and specificity
declines0.012257>0.010. No criterion is rounded up or relaxed. A small positive
ranking signal does not constitute full continuation success.

After this failure was reported, the user explicitly requested outer completion.
That separate amendment is not a passed gate, rescue training or new blind test.
All four arms' outer results are complete and independently verified; the
primary outer comparison worsens for PURN. See the detailed [OUTER_REPORT.md](OUTER_REPORT.md).

| Frozen outer model | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|
| B0 reused A0 | 0.739768 | 0.537887 | 0.624919 | 0.419405 | 0.728620 |
| B1 RMSNorm-88 | 0.726821 | 0.527317 | 0.618997 | 0.418595 | 0.693256 |
| B2 Unique-LayerNorm | 0.739365 | 0.539176 | 0.625620 | 0.416186 | 0.752043 |
| B3 PURN | 0.728407 | 0.528789 | 0.615286 | 0.409093 | 0.686701 |

PURN outer AUROC−A0 is−0.011361[−0.023761,−0.001255]. Macro-F1 declines
0.009634; its interval[−0.019680,+0.000026] still crosses zero. PURN does not
demonstrate a useful transferable gain and should not replace A0 on this evidence.

## Estimands and scope

All development entries are actual equal-appearance means over65 original
validation-fold appearances from47 unique IDs,13 per fold. Frozen80-patient
dataset contract is7,635 canonical channels,1,743 EZ/5,892 NEZ. Patient Z,
feature order, FIT-only imputer/scaler, labels(EZ0/NEZ1), original linear layers,
dropout, weighted BCE/optimizer, patient order, seed and selection are unchanged.
B0 is replayed; exactly15 new model/fold cells train independently from identical
original linear initialization. All four retain176 input affine parameters and
8,817 total; the experiment tests statistics, not added capacity.

The secondary estimator averages each ID's appearances before averaging47 IDs:

| Model | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|
| B0 | 0.721746 | 0.532438 | 0.649973 | 0.455464 | 0.681414 |
| B1 | 0.726978 | 0.534168 | 0.650549 | 0.458978 | 0.699596 |
| B2 | 0.718067 | 0.535798 | 0.650701 | 0.451110 | 0.702209 |
| B3 | 0.730242 | 0.540264 | 0.650958 | 0.461985 | 0.722359 |

It is separately reported, never substituted into the registered gate. Primary
development bootstrap resamples47 IDs retaining every appearance, not65
independent rows or channels; common10,000 seed42 draws, frozen numeric thresholds.
Intervals are unadjusted and descriptive, not selection-corrected or independent
clinical confirmation. Outer separately uses80 IDs, each once. No development
value is paired with an outer value as if they were matched observations.

## All development metrics

| Metric | B0 | B1 | B2 | B3 |
|---|---:|---:|---:|---:|
| NEZ-F1 | 0.844961 | 0.840936 | 0.849431 | 0.840183 |
| Top1-EZ | 0.584615 | 0.600000 | 0.600000 | 0.646154 |
| Balanced accuracy | 0.665436 | 0.671686 | 0.667941 | 0.672920 |
| Sensitivity | 0.465904 | 0.489438 | 0.461555 | 0.493129 |
| Specificity | 0.864968 | 0.853933 | 0.874327 | 0.852711 |
| Accuracy | 0.773174 | 0.768451 | 0.777226 | 0.769008 |
| Predicted EZ fraction | 0.198578 | 0.212054 | 0.190461 | 0.213833 |

B3 sensitivity rises0.027225 while specificity falls0.012257. Mean TP per
appearance rises8.107692→8.584615, FP11.769231→12.800000, TN66.215385→65.184615,
FN10.415385→9.938462. These are mean counts, not a pooled confusion matrix.
This is a false-positive tradeoff, not uniformly improved classification.

## Descriptive paired development intervals

Delta[95% percentile CI],10,000 paired patient-cluster draws:

| Contrast | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
|---|---|---|---|---|---|
| B3−B0 | +0.009153[0.000721,0.017659] | +0.008191[−0.002777,0.020345] | +0.000183[−0.007006,0.008082] | +0.005145[−0.006762,0.018155] | +0.044130[−0.003979,0.094743] |
| B3−B1 | +0.003519[0.000143,0.007067] | +0.009238[−0.000246,0.020659] | +0.000932[−0.005535,0.008208] | +0.002616[−0.007428,0.014303] | +0.029292[−0.013217,0.071385] |
| B3−B2 | +0.010278[−0.002628,0.026246] | +0.004618[−0.010362,0.020827] | −0.000822[−0.010503,0.009668] | +0.007604[−0.006142,0.022619] | +0.028987[−0.013398,0.071664] |
| B2−B0 | −0.001125[−0.012713,0.007672] | +0.003572[−0.005472,0.012826] | +0.001005[−0.005373,0.007459] | −0.002459[−0.011248,0.005707] | +0.015143[−0.017779,0.050222] |
| B1−B0 | +0.005633[−0.004060,0.014650] | −0.001047[−0.016223,0.013890] | −0.000748[−0.007597,0.006808] | +0.002528[−0.009959,0.016094] | +0.014838[−0.037388,0.069170] |

B3−B0 AUROC interval excludes zero positively, but it does not override the
effect-size/control/specificity gate. Its AP/F1 intervals cross zero. B3−B0
sensitivity interval[0.003085,0.055319] is positive and specificity interval
[−0.020407,−0.005036] negative. Both are development-selected descriptive
evidence. PAIRED_BOOTSTRAP.csv supplies all80 metric effects, including counts.

The independently calculated2x2 AUROC interaction is+0.004644
[−0.005923,0.017437], AP+0.005666[−0.009835,0.022594], Macro-F1−0.000074
[−0.010260,0.011033], EZ-F1+0.005075[−0.009796,0.021635], MRR+0.014149
[−0.041815,0.066966]. It is compatible with redundant or complementary effects;
no supported physiological interaction is established. FACTORIAL_COMPARISONS.csv
contains all five matched contrasts plus the interaction for every metric.

## Development consistency

| Fold | B0 AUROC | B1 | B2 | B3 | B3−B0 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.697122 | 0.710221 | 0.698636 | 0.711315 | +0.014193 |
| 2 | 0.716466 | 0.732143 | 0.719421 | 0.734462 | +0.017996 |
| 3 | 0.696716 | 0.693505 | 0.699872 | 0.698947 | +0.002231 |
| 4 | 0.751080 | 0.754076 | 0.738345 | 0.754858 | +0.003777 |
| 5 | 0.691951 | 0.691556 | 0.691436 | 0.699516 | +0.007566 |

B3 improves over B0 in5/5 folds, B1 in5/5 and B2 in4/5; fold3 B2 is higher.
B3 Macro-F1 versus B0 improves only folds2/4 and falls folds1/3/5, illustrating
that consistent AUROC increases do not guarantee classification improvements.

| Center | Appearances/IDs | B0 AUROC | B1 | B2 | B3 |
|---|---:|---:|---:|---:|---:|
| HUP | 28/21 | 0.790285 | 0.791746 | 0.781155 | 0.791822 |
| LZU | 14/11 | 0.627506 | 0.639397 | 0.632290 | 0.646661 |
| Multicenter | 11/9 | 0.796512 | 0.804563 | 0.805512 | 0.803368 |
| Pediatric | 12/6 | 0.543221 | 0.549073 | 0.544600 | 0.560580 |

B3 beats B0 at4/4 centers but both component controls at only3/4; Multicenter
favors both B1 and B2. Center samples are small and no center-specific tuning
is performed. All12 requested metrics plus mean confusion counts are preserved
by fold/center/channel-count strata in the CSVs.

| Fixed channel stratum | Appearances/IDs | B0 AUROC | B3 AUROC | B0 MF1 | B3 MF1 |
|---|---:|---:|---:|---:|---:|
| <64 | 17/12 | 0.778121 | 0.770631 | 0.697813 | 0.700336 |
| 64–127 | 39/30 | 0.667447 | 0.683678 | 0.623885 | 0.623618 |
| ≥128 | 9/5 | 0.770541 | 0.780457 | 0.586762 | 0.584475 |

The <64 stratum's ranking worsens, whereas larger-channel strata gain AUROC.
No stratum is selected/excluded. Counts are unique within stratum but repeated
validation appearances remain linked in bootstrap.

## Frozen selection and gate

Selected epoch / NEZ threshold by fold:

| Arm | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 |
|---|---|---|---|---|---|
| B0 | 3/0.465 | 1/0.275 | 2/0.340 | 2/0.355 | 2/0.445 |
| B1 | 2/0.510 | 2/0.300 | 2/0.340 | 1/0.410 | 7/0.350 |
| B2 | 3/0.465 | 1/0.270 | 2/0.350 | 8/0.345 | 6/0.360 |
| B3 | 2/0.510 | 2/0.265 | 3/0.335 | 1/0.415 | 6/0.395 |

B1 completed[8,8,8,7,13], B2[9,7,8,14,12], B3[8,8,9,7,12] epochs under
original stopping.138 total,12.968455 seconds measured trainer calls, not
end-to-end task time. Every selected FIT/VAL eval-mode weighted BCE and full
training history is reported; it is not an alternative checkpoint criterion.

| Requirement | Actual | Pass |
|---|---|---|
| B3−B0 AUROC≥0.010 | 0.009153 | No |
| B3−B1 AUROC≥0.005 | 0.003519 | No |
| B3−B2 AUROC≥0.005 | 0.010278 | Yes |
| AP≥B0 | +0.008191 | Yes |
| Macro-F1≥B0 | +0.000183 | Yes |
| EZ-F1 loss≤0.005 | +0.005145 | Yes |
| Specificity loss≤0.010 | −0.012257 | No |
| AUROC improves≥4 folds | 5/5 | Yes |
| No fold feature/preprocessing mismatch | PASS | Yes |
| All identity/parameter/numeric/leakage checks | PASS | Yes |

Original gate fails. Subsequent user-authorized outer does not retroactively
change this decision, criteria, stored protocol or development status files.
No OUTER_READINESS.md is claimed or automatic v2 begun.

## Answers to all14 research questions

1. **Duplicates overweighted?** Numerically yes in all88 moments, selected
   unique coordinates count twice. This is real weighting, not proof of harmful
   physiological bias. Duplicate VAL energy share averages21.20%, varying by channel.
2. **Zeros' effect?** RMS dilution is almost constant sqrt(88/72)≈1.105542;
   mean/variance effects are channel-dependent. Zero centered-energy share
   averages5.17%. They remain at their88 positions, never removed or reset.
3. **Ordinary RMS improves?** Development AUROC+0.005633, CI crosses zero;
   AP/MF1 point estimates decline and specificity worsens. Outer AUROC declines.
4. **Unique56 better than88?** Under centered normalization B2−B0 AUROC is
   −0.001125; under RMS B3−B1 is+0.003519 with positive descriptive CI but below
   the minimum increment. Improvement is not unconditional; outer is unsupported.
5. **PURN beats both controls?** Highest development AUROC/AP, not highest MF1;
   B3−B2 AUROC CI crosses zero. Outer trails B2 and has no supported gain over B1.
6. **At least +0.010 AUROC?** No: +0.009153 development, −0.011361 outer versus A0.
7. **AP/MF1 preserved?** Numerically yes on development, but both difference
   intervals cross zero. Both outer point estimates decline.
8. **Sensitivity without FP cost?** No: development sensitivity+0.027225,
   FP+1.030769 per appearance, specificity−0.012257. Outer has no sensitivity
   benefit and70 extra pooled false positives without extra pooled EZ TP.
9. **Consistency?** Development AUROC gains5/5 folds and4/4 centers but loses
   a channel-count stratum and Multicenter to controls. Outer gains1/5 folds
   and only a tiny Pediatric point increment; not stable transport.
10. **Factor/rank correlations?** Modest descriptive associations exist; B3
    RMS-ratio mean vs pair-rank changes Pearson−0.266668/Spearman−0.270760.
    Repeated appearances and multiple probes preclude causal/significance claims.
11. **Incremental architecture?** A limited development ranking signal is
    present without added capacity, but the predefined increment/control/
    specificity criteria fail and outer AUROC worsens. No useful replacement
    advantage is demonstrated.
12. **JBHI sufficient?** No. This is a completed negative controlled study
    with an exploratory development signal that does not survive this outer
    comparison. A new normalization name is not mathematical novelty.
13. **Full gate?** No; three criteria fail and remain failed after outer.
14. **Missing evidence?** Genuinely independent patients/prospective evaluation,
    robust operating-point transport, multiple fixed-seed uncertainty and
    external/source generalization. Current repeated cohort intervals do not
    provide those. No such extra experiment is performed here.

## Integrity and source provenance

Baseline bitwise replay, feature mapping, original preprocessing,476 numerical/
initial/resume tests and independent260-cell development/320-cell outer metrics
checks PASS (maximum discrepancy3.33e-16). All15 training cells run once; no
scientific retries, source changes, new feature extraction or retuning. U/affine/
backbone/threshold/source hashes remain bound. Exact mappings are public by
feature name/index; all identifiers and individual arrays remain private.

Historical findings are from reviewed existing code/reports; this run's values
are exclusively actual results. No new historical experiment is re-executed.
NORMALIZATION_MECHANISM_AUDIT.md supplies actual weighted RMS/LN/duplicate/zero
moments, factor frequencies, gradients, activation and affine summaries. Public
delivery contains code, hashes, protocols and anonymous aggregate reports only,
never EEG/features/checkpoints/individual predictions/logs. No installed skills.
