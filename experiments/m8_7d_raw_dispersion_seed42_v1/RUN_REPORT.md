# Actual Raw dispersion study: negative internal gate

## Outcome

All10new fits/117epochs and independent verification completed. Neither new
architecture passes the frozen Development Macro improvement gate. The
selected reference remains C2; **no new Outer evaluation was run**. This
study does not establish the user's>0.64full80 patient-Macro target.
The previous actually verified C2 full80 value remains0.6396579900303192,
below the target; these internal numbers cannot replace it.

## What changed and what did not

Original7D measurements and14processed coordinates were retained, including
variance and partial high-gamma. Same80patients/7635channels/256seizures,
1471965valid Raw250 channel-windows,original patient-held-out roles,masks,
complete differentiable aggregation,Gate and one C2 scalar classifier.
No new physiological features,trained D1,logit residual or second classifier.

The new Raw pool preserves the learned temporal response mean and adds
0.25*tanh(W statistic+b). R0 uses the same response mean; R1 uses its
within-window population SD. Both add600zero-initial parameters,have10978
total parameters,and share exact original initial predictions. The retained
C2 has10378. This tests conditional incremental value and dispersion versus
matched nonlinear capacity,not a claim of generic pooling novelty.

All new weights were freshly initialized,not copied from trained C2.
AdamW,objective,max30/min6/patience6,seed42,patient order and201threshold
rules match the earlier C2. Only legal own-FIT updates and original
Development selections were used. Five exact old C2 fits are reused with
zero new epochs; actual prior epochs are retained in DEVELOPMENT_CELLS.

## Actual Development results

65patient-fold appearances/47unique people,not80Outer patients. The same
Development selects epoch,threshold and architecture; estimates are
selection-conditioned,not unbiased nestedCV or independent confirmation.

| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ frac.bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 reference | .673076 | .762169 | .561626 | .484762 | .861390 | .794168 | .790300 | .707692 | -.016861 |
| R0 mean-only extra capacity | .673594 | .764880 | .563017 | .487085 | .860102 | .792731 | .780147 | .692308 | -.011226 |
| R1 learned Raw dispersion | .672001 | .763928 | .558274 | .490736 | .853265 | .785604 | .780371 | .692308 | +.007184 |

Eligibility requires Macro≥C2+.003 and AUROC/AP≥C2−.01. R0 gains
.0005174195423195092;R1 loses-.0010755714002915803. Both fail Macro.
The gate was not loosened because the previous full80 result nearly reached
.64. No favorable fold,metric or secondary arm is relabeled the primary.

## Paired uncertainty and stability

Nominal95% percentile intervals use10000seed42patient-cluster draws,keeping
repeated appearances together. They do not account for seed,selection or
historical adaptation uncertainty and are not multiplicity-adjusted.

| Patient Macro contrast | Difference |95% CI |
|---|---:|---|
| R0−C2 | +.000517 | [-.005809,+.006937] |
| R1−C2 | -.001076 | [-.009891,+.007636] |
| R1−R0 | -.001593 | [-.010791,+.007253] |

Fold sampleSD of Macro:C2.019543,R0.019867,R1.020774. Dispersion does not
show improved stability in this study. Its higher EZ-F1 accompanies lower
NEZ-F1,ACC and AP; patient-Macro remains lower. All54absolute/paired
intervals for all9metrics and full fold/source/coverage tables are released.
Sources with fewer than5unique people have metrics suppressed,not patients
excluded. These are inherited source groups,not leave-center-out evaluation.

The result rejects this tested learned-response-dispersion addition as a
sufficient improvement. It does not prove clinical variance is useless or
that every Raw architecture has reached its ceiling. Learned temporal SD is
not the same quantity as the unchanged physiological variance measurement.

## Execution,verification and publication

Admission5CPUchecks and all5actual C2 validation replays passed with maximum
logit error0 and identical201threshold decisions. The largest complete legal
FIT patient contains164channels/48380validwindows. All3fresh admission
models passed mask,channel-order and3discarded-step full-gradient checks.
No formal epoch was discarded; initial full-patient logits were exact.

Actual new epochs:R0=62,R1=55,total117. Every15real retained checkpoint/
control cell was replayed;195patient rows/1755independent metric checks and
54independently reconstructed intervals passed. Maximum replay and interval
differences were0. No model or threshold was modified after selection.
The2pre-admission guard/deployment errors and preserved original sources/
logs are disclosed in ENGINEERING_LEDGER; they occurred before scientific
input freeze or any training,not after an outcome-driven change.

Exact terminal: RAW_DISPERSION_INTERNAL_PIPELINE_COMPLETE_NO_OUTER.

Model freeze SHA256:
2442250873dc3e10fc173edafc1c0ce848917924212607a084475f218093c042.
Candidate/failed-gate SHA256:
336086e5fa8f063d12e7a1baf77010cad4b19213810a4947c5b8ec942f2dc7f3.
Anonymous13file delivery ZIP SHA256:
a5b736cd910a0c9b8c2e305fdeb0552224a2a6558fbac9bef346ec08039f8768.

Source was uploaded before formal fits at17b021ae948a4fc2361683404137bf24becb6486;
actual admission was uploaded atf4ca6ec4f3347812cca4bbefdb2bdc055db78528.
Only source,hashes and anonymous real aggregates are published. Raw,labels/
IDs,individual scores/embeddings,weights,scalers and logs stay on the server.
The broader architecture target remains unproven,not completed by this study.
