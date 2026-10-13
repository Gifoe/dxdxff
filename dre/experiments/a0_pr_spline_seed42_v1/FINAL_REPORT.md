# A0 + PR-Spline seed42: completed development, gate failed

| Model | Params | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| B0 Original A0 | 8,817 | 0.710667 | 0.518235 | 0.638080 | 0.431198 | 0.686550 |
| B1 Positive Feature Scale | 8,873 | 0.710654 | 0.518284 | 0.638403 | 0.432043 | 0.686841 |
| B2 Generic Input Residual | 9,259 | 0.711366 | 0.519250 | 0.637548 | 0.430486 | 0.685867 |
| B3 PR-Spline | 9,265 | 0.710482 | 0.518106 | 0.638123 | 0.431487 | 0.679149 |

**Scientific terminal: `PR_SPLINE_DEVELOPMENT_GATE_FAILED`.** The proposed
spline does not improve the exact A0 development AUROC, does not beat either
control, and improves AUROC in only 1/5 folds. Its near-zero Macro-F1 change
does not offset declining ranking metrics. No new Outer TEST was accessed,
no new raw/features were extracted, and no rescue variant was introduced.

## Scope, measurements and verification

These are actual completed server measurements: 5 historical B0 replays plus
15 new B1/B2/B3 train/validation cells, 120 completed new epochs under the
original stopping rule, and frozen identity/permutation interventions. All
new arms share the exact original A0 constructor/initial backbone, original
Patient Z/FIT preprocessing, source-free loss, optimizer and selection.
No installed skill was used. Baseline/data admission and 293 pretraining
engineering checks PASS. Independent sklearn metric and alternate bootstrap
verification PASS: 390 metric cells, 112 intervals, maximum drift 2.22e-16.

Primary numbers average each of the **65 validation-fold appearances** equally,
representing **47 unique patients**, with 13 appearances in every fold. This
exactly replays the historical development estimator; 65 is not 65 independent
patients. `DEVELOPMENT_UNIQUE_ID_SUMMARY.csv` additionally averages appearances
within each ID before averaging 47 IDs (B0/B3 AUROC 0.721746/0.721646). That
secondary estimator is not substituted into the registered gate. Center
summaries use the same appearance-based convention.

All bootstrap draws resample the 47 IDs with replacement, retain all their
appearances, use common seed42 draws for contrasts, and keep checkpoints and
thresholds fixed. Intervals are descriptive, unadjusted and development-selected;
they do not account for repeated cohort use or selection uncertainty. This
is not fresh independent clinical confirmation. No old outer metric is used
as a new development comparator.

## All primary metrics

| Metric | B0 | B1 | B2 | B3 |
|---|---:|---:|---:|---:|
| AUROC | 0.710667 | 0.710654 | 0.711366 | 0.710482 |
| AUPRC | 0.518235 | 0.518284 | 0.519250 | 0.518106 |
| Macro-F1 | 0.638080 | 0.638403 | 0.637548 | 0.638123 |
| EZ-F1 | 0.431198 | 0.432043 | 0.430486 | 0.431487 |
| NEZ-F1 | 0.844961 | 0.844762 | 0.844609 | 0.844760 |
| MRR | 0.686550 | 0.686841 | 0.685867 | 0.679149 |
| Top1-EZ | 0.584615 | 0.584615 | 0.584615 | 0.569231 |
| Balanced accuracy | 0.665436 | 0.665838 | 0.664844 | 0.666354 |
| Sensitivity | 0.465904 | 0.467303 | 0.464697 | 0.468585 |
| Specificity | 0.864968 | 0.864374 | 0.864992 | 0.864122 |
| Accuracy | 0.773174 | 0.772909 | 0.772610 | 0.772837 |
| Predicted EZ fraction | 0.198578 | 0.199403 | 0.197967 | 0.199792 |

Full precision is in `DEVELOPMENT_SUMMARY.csv`. TP/FP/TN/FN there are mean
counts per validation appearance, not a pooled confusion matrix. Against B0,
B3 TP rises 8.107692→8.138462 and FP 11.769231→11.815385; sensitivity rises
0.002681 while specificity falls 0.000846. Predicted EZ fraction rises
0.001214. The small sensitivity change comes with more false positives,
not improved discrimination.

## Paired cluster bootstrap: 10,000 draws, seed42

Cells show delta [95% percentile interval].

| Contrast | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
|---|---|---|---|---|---|
| B3 − B0 | −0.000185 [−0.000600, 0.000189] | −0.000130 [−0.001054, 0.000739] | 0.000044 [−0.002097, 0.002401] | 0.000289 [−0.003394, 0.004257] | −0.007401 [−0.024280, 0.000819] |
| B3 − B1 | −0.000171 [−0.000468, 0.000143] | −0.000179 [−0.001008, 0.000625] | −0.000279 [−0.002023, 0.001305] | −0.000556 [−0.003493, 0.002137] | −0.007692 [−0.024590, 0.000000] |
| B3 − B2 | −0.000883 [−0.001540, −0.000291] | −0.001144 [−0.002154, −0.000240] | 0.000576 [−0.002237, 0.003746] | 0.001001 [−0.003880, 0.006620] | −0.006718 [−0.023622, 0.002086] |
| B2 − B0 | 0.000699 [0.000115, 0.001369] | 0.001015 [0.000109, 0.002060] | −0.000532 [−0.003547, 0.002230] | −0.000712 [−0.006054, 0.004046] | −0.000683 [−0.002292, 0.000502] |
| B1 − B0 | −0.000013 [−0.000372, 0.000327] | 0.000049 [−0.000209, 0.000314] | 0.000323 [−0.000891, 0.002103] | 0.000845 [−0.001049, 0.003857] | 0.000291 [0.000000, 0.000911] |

B3−B0 AUROC interval crosses zero. B3 is below B2 in both AUROC/AP with
descriptive intervals entirely negative. B2's own very small AUROC/AP increase
has intervals above zero, but is far below the spline's registered research
target and does not establish clinical usefulness or warrant rescue testing.

## Consistency across folds and centers

| Fold | B0 AUROC | B1 AUROC | B2 AUROC | B3 AUROC | B3 − B0 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.697122 | 0.697638 | 0.697375 | 0.696836 | −0.000286 |
| 2 | 0.716466 | 0.716774 | 0.717069 | 0.716522 | 0.000056 |
| 3 | 0.696716 | 0.696726 | 0.697043 | 0.696620 | −0.000096 |
| 4 | 0.751080 | 0.751142 | 0.753042 | 0.751072 | −0.000008 |
| 5 | 0.691951 | 0.690987 | 0.692300 | 0.691361 | −0.000589 |

| Center | Appearances / unique IDs | B0 AUROC | B3 AUROC | B3 − B0 | B0 AP | B3 AP |
|---|---:|---:|---:|---:|---:|---:|
| HUP | 28 / 21 | 0.790285 | 0.790225 | −0.000060 | 0.555377 | 0.555498 |
| LZU | 14 / 11 | 0.627506 | 0.626222 | −0.001284 | 0.515798 | 0.513669 |
| Multicenter | 11 / 9 | 0.796512 | 0.797075 | 0.000563 | 0.631022 | 0.632637 |
| Pediatric | 12 / 6 | 0.543221 | 0.543344 | 0.000124 | 0.331026 | 0.331045 |

Only one fold improves and center changes are mixed and small. Fold4 MRR
falls 0.839250→0.800789; the corresponding LZU aggregate MRR falls
0.630357→0.595918 and Top1-EZ 0.428571→0.357143. No center-specific model,
threshold or knot rule was fitted. These subgroups are descriptive, not
independent center-specific claims. Full metrics are in the fold/center CSVs.

## Frozen mechanism interventions

| Frozen B3 variant | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR | Top1-EZ |
|---|---:|---:|---:|---:|---:|---:|
| Trained curves | 0.710482 | 0.518106 | 0.638123 | 0.431487 | 0.679149 | 0.569231 |
| M1 identity | 0.710579 | 0.518096 | 0.637281 | 0.429935 | 0.686566 | 0.584615 |
| M2 whole-curve permutation | 0.710577 | 0.518407 | 0.638253 | 0.431781 | 0.686566 | 0.584615 |

Same selected B3 backbone and numeric threshold, no retraining/reselection.
M1 is **not** independently trained B0. M2 uses the same seed42 permutation
of entire curve assignments (knots/slopes/masks) across folds, retaining
duplicate-to-main sharing. It changes assignment, not input feature order.

Removing curves slightly lowers MF1 (full−identity +0.000843, CI
[−0.001056,0.003476]) but raises AUROC and MRR. Permuting does not reduce
AUROC/AP/MF1. Full−identity AUROC −0.000097 [−0.000472,0.000250];
full−permuted −0.000095 [−0.000515,0.000294]. Hence numerical use is
demonstrated, beneficial feature-specific assignment is not.

Within-patient pair-order changes average 0.001556 against M1 and 0.002166
against M2; ties are included as ordering-state changes. Top1 channel identity
changes in 3/65 appearances against either intervention (this is not the
Top1-EZ accuracy delta). Full vs M1 gains mean 0.030769 EZ TP and adds
0.092308 FP while removing 0.061538 FP. Ranking and confusion attribution
are in `RANKING_ATTRIBUTION.csv`.

All 56 curves are algebraically active (448 extra parameters), but learned
slopes remain 0.956295–1.040381. Mean absolute deviation is 0.007924, with
outer/interior deviations 0.009941/0.007251. Mean channel-pooled VAL transform
is 0.003411 over representative feature/fold rows; the patient-appearance mean
over all88 input columns is 0.002872. VAL outer-interval fraction averages
0.033419. These support near-identity behavior, not a useful nonlinear gain.
Near-coincident knots and precise weighting are discussed in the feature audit.
No tail-only intervention was run, so tail causality cannot be concluded.

## Selection, efficiency and fixed gate

All B0/B1/B2/B3 selected epochs by fold are [3,1,2,2,2], individually selected
through the original Macro-F1/EZ-F1/earlier-epoch procedure, not manually tied.
Numeric NEZ thresholds by fold:

| Arm | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.465 | 0.275 | 0.340 | 0.355 | 0.445 |
| B1 | 0.465 | 0.275 | 0.340 | 0.355 | 0.445 |
| B2 | 0.460 | 0.265 | 0.345 | 0.350 | 0.445 |
| B3 | 0.465 | 0.275 | 0.345 | 0.355 | 0.445 |

Each new arm completed [9,7,8,8,8] epochs. The total recorded trainer time
across15 invocations is 13.128539 seconds (excluding audits/bootstrap/delivery).
FIT/VAL selected-checkpoint eval-mode weighted BCE is reported for every cell
in `TRAINING_EFFICIENCY.csv`; it did not select models. First pretraining
attempt corrected a manifest directory before any new clinical training;
all formal clinical training ran once without a crash. No epoch budget cut,
checkpoint change or alternate threshold optimization occurred.

| Predeclared B3 requirement | Observed | Pass |
|---|---|---|
| AUROC−B0 ≥0.010 | −0.000185 | No |
| AUPRC≥B0 | −0.000130 | No |
| Macro-F1≥B0 | +0.000044 | Yes |
| EZ-F1−B0 ≥−0.005 | +0.000289 | Yes |
| AUROC−B1 ≥0.005 | −0.000171 | No |
| AUROC−B2 ≥0.005 | −0.000883 | No |
| Positive AUROC folds≥4 | 1/5 | No |
| Specificity−B0 ≥−0.010 | −0.000846 | Yes |
| All admission/integrity checks | PASS | Yes |

## Answers to the 15 research questions

1. **Exact previous implementation?** Not found in inspected available refs,
   archives and relevant checkouts. Monotonic splines and numerical embeddings
   are established prior mechanisms, not a new mathematical invention. Search
   scope and historical/report-derived results are in the duplication audit.
2. **Initial A0 reproduction?** Yes: exact original backbone, zero theta,
   CPU/GPU random/extreme/full legal-bank parity within1e-6; no alternate A0.
3. **FIT-only knots?** Yes, label-free inverse empirical CDF with equal patient
   mass; original preprocessing unchanged and independent final buffers verified.
4. **Active DOF?** 448 per fold, all56×8 active; total9,265. Near-coincident
   boundaries mean this count is not proof of statistical identifiability.
5. **At least +0.010 AUROC?** No, −0.000185 versus B0.
6. **AUPRC improved/stable?** Numerically lower by0.000130, descriptive CI
   spans zero; the strict nondecrease gate fails even though drift is tiny.
7. **Macro-F1 preserved?** Yes numerically, +0.000044, but CI spans zero.
8. **Beats scaling?** No: AUROC/AP/MF1/EZF1/MRR all below B1.
9. **Beats matched residual?** No: AUROC/AP below B2; descriptive paired
   intervals are negative. B3's tiny MF1 advantage has no robust interval gain.
10. **Consistent folds/centers?** No: 1/5 folds, mixed center changes, LZU
    ranking deterioration. No consistent generalization increment is established.
11. **Uses nonlinear curves?** Yes computationally (nonidentity slopes/input
    and ranking changes), but identity/permutation do not establish benefit or
    useful feature-specific correspondence.
12. **Mainly extremes?** Not established. Tail slopes change slightly more
    than interior slopes, only3.34% mean VAL observations are outside knots;
    region-specific causal attribution was not tested. Do not invent it.
13. **Sensitivity versus FP?** Small sensitivity gain +0.002681 accompanies
    mean FP +0.046154 and specificity −0.000846 versus B0.
14. **Spline-specific or ordinary capacity?** No positive spline-specific
    advantage. Generic B2 ranks slightly better; this does not establish a
    clinically meaningful capacity benefit either. Near-identity behavior is
    not evidence that feature geometry improves localization.
15. **Ictal JBHI candidate?** No sufficient efficacy evidence from this study.
    The method fails its explicit minimum gain/control/consistency criteria.
    Treat it as a completed negative controlled experiment, not a demonstrated
    localization contribution. No v2, wider caps, alternate knots, rescue
    feature subset, extra seed or new outer evaluation follows.

## Delivery boundary

Only source, test code, protocol/hash locks and anonymous aggregate statistics
are published. Original banks, all selected checkpoints, patient/channel
predictions, clinical records and logs remain private on the trusted server.
Every value above is traced to actual result CSV/JSON or clearly marked
historical evidence; no unrun clinical outcome is filled with zero. No
`OUTER_READINESS.md` is produced because continuation fails and this task
does not authorize new outer testing in any case.
