# Frozen Raw patient decision experiment — seed42

## Outcome

**The target was not achieved.** The required quantity is the mean of each
patient's binary Macro-F1 across all original 80 patients. It is not EZ-positive
F1, pooled/weighted F1, fold-equal averaging, internal performance or an oracle.
The exact original Raw-only reference remains **0.6304930529317453**.

All eight preregistered fixed procedures and the ownFIT-selected policy were
evaluated, including unsuccessful procedures. The highest nominal Outer point
is **0.6338794718916507**, below 0.65. It was not selected by the legal internal
policy and must not be adopted as a best-Outer winner. The selected policy is
**0.6280297919345083**. No rule or threshold was changed after reading these
results. All Outer results are exploratory repeated-test, not independent
confirmation.

## What was tested

Two previously frozen Raw score bodies: original Native, and the prior shared
CNN/window-plus-precompand residual. Both have one classification head, no D1
branch and no handcrafted physiological feature input. This study updates no
neural parameter and changes no waveform preprocessing, model, Raw window,
patient role, cohort, or source matrix.

Each body has four decision rules:

| Rule | Selected on legal CAL/Development only | Patient adaptation |
|---|---|---|
| ORIGINAL | Inherited frozen FP32 NEZ probability threshold | None added |
| MARGIN_GRID | One raw-logit cutoff, 201 candidates | None added |
| PATIENT_Z | One standardized-logit cutoff, 201 candidates | Complete-patient mean/population SD |
| TOP_FRACTION | One fraction, 101 candidates | Label-free rank cutoff; equal scores indivisible |

These are decision controls, not new learned representations or calibrated
patient-prevalence estimators. No Gaussian mixture or density posterior was
used. Patient/channel identity never determines a binary decision or splits a
score tie. Channel names are used only for the inherited ranking tie convention.

The 15 existing ownFIT paths give 120 logical internal cells and 90 new scalar
fits. All configurations and HELD scores were sealed before HELD metrics.
There are 255 HELD appearances / 79 unique patients per procedure. Reused and
overlapping paths are selection-conditioned evidence, not 255 independent
patients. Five procedure choices were frozen before original Development.
Original Development supplied 30 further scalar calibrations, with unchanged
fullFIT score bodies. The 40 fixed decision cells plus five selected aliases
were frozen before new Outer margin inference; all 45 score/decision cells
were sealed before the complete Outer truth archive was opened.

## Internal results

| Frozen body | ORIGINAL | MARGIN_GRID | PATIENT_Z | TOP_FRACTION |
|---|---:|---:|---:|---:|
| Native | 0.609084 | 0.609611 | 0.606234 | 0.602336 |
| Precompand residual | 0.609149 | 0.609369 | 0.602969 | 0.608788 |

Numbers are patient Macro-F1 over HELD appearances. Native margin-grid minus
original is only +0.000526, paired patient-cluster 95% CI
[-0.001365, 0.002746]. The evidence does not support a substantial calibration
gain. The ownFIT policy chose Native ORIGINAL in folds 1–3, precompand
TOP_FRACTION in fold 4 and precompand ORIGINAL in fold 5; no global best arm
was chosen using overlapping evaluation paths or Outer outcomes.

The internal label-using per-patient best-threshold diagnostic is 0.699887 for
Native and 0.704430 for precompand. This **nondeployable oracle** uses the
patient's labels. It cannot establish the goal, justify a deployable threshold,
or show that an unlabeled calibration method can recover that performance.
No Outer oracle was computed.

## Complete fivefold Outer results

Patient-equal means; each row covers 80 patients, 7,635 unique channels and
the same 256 seizures / 1,471,965 valid Raw250 windows. `P` denotes the frozen
precompand residual score body, not a newly trained network in this study.

| Procedure | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC |
|---|---:|---:|---:|---:|---:|---:|
| Native + ORIGINAL | 0.630493 | 0.741296 | 0.550114 | 0.426475 | 0.834511 | 0.757014 |
| Native + MARGIN_GRID | 0.630071 | 0.741296 | 0.550114 | 0.425469 | 0.834673 | 0.757295 |
| Native + PATIENT_Z | 0.626811 | 0.741296 | 0.550114 | 0.417909 | 0.835713 | 0.757688 |
| Native + TOP_FRACTION | 0.621462 | 0.741296 | 0.550114 | 0.404285 | 0.838639 | 0.759301 |
| P + ORIGINAL | 0.631432 | 0.759384 | 0.570215 | 0.420221 | 0.842644 | 0.766018 |
| P + MARGIN_GRID | 0.633582 | 0.759384 | 0.570215 | 0.424722 | 0.842441 | 0.766151 |
| P + PATIENT_Z | 0.633879 | 0.759384 | 0.570215 | 0.430713 | 0.837046 | 0.759703 |
| P + TOP_FRACTION | 0.627729 | 0.759384 | 0.570215 | 0.415982 | 0.839476 | 0.761437 |
| OwnFIT-selected policy | 0.628030 | 0.752208 | 0.555576 | 0.416587 | 0.839473 | 0.760199 |

All four Native decisions share MRR 0.784454 and Top-1 0.700000. All four P
decisions share MRR 0.802328 and Top-1 0.725000. Selected MRR is 0.795123,
Top-1 0.712500. Decoder changes cannot improve the frozen body's ranking.
The full CSVs include EZ-fraction bias, per-fold means and sample SD, source
aggregates (sources with <5 patients suppressed), all metric coverage and CIs.
Mixed-fold pooled results are separate and are not the primary endpoint.

Paired 10,000-patient-bootstrap Macro-F1 differences:

| Comparison | Difference | Nominal 95% CI |
|---|---:|---|
| P + PATIENT_Z minus Native ORIGINAL | +0.003386 | [-0.009469, 0.016367] |
| P + MARGIN_GRID minus Native ORIGINAL | +0.003089 | [-0.008077, 0.014737] |
| P + MARGIN_GRID minus P ORIGINAL | +0.002149 | [-0.000280, 0.004717] |
| Selected minus Native ORIGINAL | -0.002463 | [-0.013065, 0.007849] |

These intervals are descriptive and unadjusted for historical/multiple
comparisons. None establishes a reliable Macro-F1 gain. The highest nominal
point has absolute CI [0.605512, 0.662658]; an upper CI above 0.65 does not
satisfy a target requiring actual mean F1 >0.65. It remains 0.016121 short.
Fold Macro-F1 sample SD is 0.035327 for Native ORIGINAL, 0.030113 for
P + PATIENT_Z, and 0.039590 for the selected policy. Lower fold SD alone
does not constitute improved patient performance or source robustness.

## Precision and verification

Natural unbounded NEZ logits were recovered by one frozen joint two-body
forward, rather than inverse-transforming rounded probabilities. The maximum
Native probability difference versus previous sealed scores was
1.1920928955078125e-7; precompand difference was exactly zero. All original
binary predictions and binary patient metrics were unchanged. Outer has zero
saturated probability channels, and raw-margin AUROC/AP/MRR/Top-1 exactly match
the legacy FP32-probability rankings. Saturation/grid precision does not explain
the performance gap in this cohort.

Independent verification passed all 720 patient metric rows (nine procedures),
all 45 configuration/score cells, exact selected aliases, original role/cohort
identities, all nine metrics and 99 paired-CI checks. Maximum independent
metric difference is zero. Coverage is complete; no patient/window/channel was
dropped to improve metrics.

An engineering error occurred only in the independent verifier: it expected
`y` in a `labels=False` metadata bank. All predictions and reports had already
been sealed. `verify_final_reporting_v2.py` maps the already legally opened
post-seal truth archive to the CAL archive indices. The original verifier,
source freeze, models, scalar decisions, predictions and logs remain unchanged.
No training or inference was repeated for this repair. The repair is separately
audited in `REPORTING_VERIFIER_REPAIR.json`.

## Conclusion and next decision

This finite experiment does **not** support simple global threshold refinement,
patient-logit standardization, or fixed predicted EZ fraction as a route from
the existing Raw-only 0.63049 to >0.65. Precompand retains its previously seen
ranking gain, but the present decision rules do not turn it into a stable
Macro-F1 gain. The original baseline is not replaced by a nominal Outer winner.
No further threshold grid expansion or favorable subgroup selection follows
from these results. The broader Raw-only model goal remains active and unmet;
any subsequent representation/training experiment needs its own prospective
internal selection protocol rather than reusing this Outer to adjust rules.

## Reproduction and artifacts

1. Deploy the unchanged sources and their pinned predecessor dependencies.
2. Run `run_internal.ps1 -RunTag <fresh>` in a sustained foreground SSH owner.
3. Run `run_final_execution_v2.ps1 -RunTag <fresh>` for the frozen final stages
   and corrected reporting-only verifier. This runner does not change the
   source-frozen mathematical code. To verify already sealed final outputs
   alone, use `run_verification_v2.ps1` with a fresh tag.
4. On a public checkout, `python verify_public.py` validates delivery schemas,
   all nine weighted fold means and artifact hashes without private data.

The actual run used original `run_final.ps1` through report; its verifier error
was repaired with `run_verification_v2.ps1 -RunTag 02`. This is not represented
as a never-failed run. The execution-v2 runner is a reproducible entrypoint to
the same stages, not a second model experiment.

- Internal selection SHA256: `1dcdd3bd8555786c67b5c452d9e260da07fb869d38955eeb7c88ab4f4841cbdf`
- Final decisions SHA256: `29164b7b023595edfc270d1bf820a66069d61d2b300e934806c196f6528149f0`
- All-45 score seal SHA256: `fc5ac982885e57f01d68ed64d95669dc0e5b2920b7b6bd3422a08801572d1353`

Only code, hashes and anonymous aggregates are published. Individual scores,
labels, decisions, patient/channel identities, weights, Raw banks and runtime
logs remain on the trusted server.
