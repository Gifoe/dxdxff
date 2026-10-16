# Censor-aware qualification of six record-level physiological measurements

## Finding

Six partial-high-gamma and line-length record descriptors satisfy the fixed
coverage and own-TRAIN variation criteria on the original80-patient cohort.
They produce12 channel coordinates after equal-seizure aggregation. These
measurements extend the available candidate definitions; this stage estimates
neither EZ/NEZ differences nor predictive contribution.

The [preceding source audit](../m8_nonlocal8_eligibility_audit_seed42_v1/AUDIT_REPORT.md)
excluded two channel-order-dependent ranks. This study uses the other six
operators on existing valid Raw250 windows. High gamma refers to the retained
80–125 Hz band. It does not represent complete80–150 Hz high gamma.

## Measurement definitions

Early slope is the frozen least-squares slope over centers0–10 s. Post top20%
is the mean of the largest ceiling20% post-onset window-feature values. Each
is summarized by equal-seizure mean and population SD for a canonical channel.
An absent channel in a seizure contributes neither a zero nor a duplicated
observation. Values are aggregated in frozen record order.

The two crossing groups use the first post center whose feature value reaches
pre mean+2 pre SD. Its location is divided by the largest absolute retained
window center. This is a dimensionless baseline-crossing position, not a
clinical onset latency. Each group supplies mean observed position and the
fraction of eligible seizures with an observed crossing. Unobserved crossings
remain missing in the position coordinate. The source helper's no-crossing
fallback is never averaged into that coordinate. A zero pre SD makes the
record-channel baseline unavailable rather than implying an early crossing.

## Actual coverage and quality

All256 records,24,995 record-channel units,7,635 canonical channels and
1,471,965 valid channel-windows were retained. Every recomputed record scalar
exactly matched the preceding source audit. The original Raw mask was used
unchanged; the previously disclosed27 native/Raw mask disagreements were not
resolved by silently switching support or deleting observations.

| Group | Source finite rate | Eligible baseline rate | Coordinate0 coverage | Coordinate1 coverage | Lowest own-TRAIN nonconstant-patient fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Early partial-HG slope | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Early LL slope | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Partial-HG baseline crossing | 1.000000 | 0.999840 | 0.979437 | 0.999869 | 1.000000 |
| LL baseline crossing | 1.000000 | 1.000000 | 0.977472 | 1.000000 | 0.900000 |
| Post top20% partial HG | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Post top20% LL | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

The final column takes the better of a group's two coordinates within each
selection TRAIN and then the minimum over all20 original inner partitions.
The threshold was0.5. Coverage requirements were0.99, except that legitimately
censored crossing positions were separately reported rather than treated as
source corruption. Their companion crossing-rate coordinate still had to meet
the0.99 coverage requirement.

There are157 channels without an observed eligible high-gamma crossing and172
without an observed eligible line-length crossing. One high-gamma channel has
no eligible baseline across its available seizures. That channel is retained
with missing crossing coordinates. Four of24,995 high-gamma record-channel
baselines have zero pre SD. No line-length baseline has zero pre SD. No
patient, channel or seizure was excluded to satisfy a gate.

## Numerical result and disclosed repair

The first implementation failed the preregistered unit-equivalence tolerance
of1e-9. All failed patient-coordinates were exactly constant. Floating
reductions of identical non-binary values produced a tiny positive SD and
spurious normalized values of±1. Nonconstant coordinates passed. The failure
and [numeric amendment](NUMERIC_AMENDMENT.json) were published before the
corrected run; the failed implementation remains intact.

The corrected float64 patient-relative transform sets finite exactly constant
patient-coordinates to0, retains all-missing coordinates as missing, and leaves
the nonconstant calculation unchanged. Multiplying already-defined coordinates
by1e-12 or1e12 then changes the transform by at most9.547918e-14. This is a
coordinate-unit check, not a test of raw-voltage-unit invariance of log1p power.

With imputation and scaling fitted only on the corresponding selection TRAIN,
the maximum absolute transformed early-HG input across the internal roles was
11.40 for the unit-equivariant transform versus10,319.86 for the diagnostic
legacy absolute-floor transform. For post top20% HG the corresponding values
were13.44 and291.60. These are input-range diagnostics, not improvements in
classification or robustness. Earlier32-group inputs and all trained models
were left unchanged.

## Verification and interpretation

Independent computation reconstructed all91,620 aggregated coordinate cells
and checked450 group-by-role numeric entries across20 partitions. All160
channel/record permutation checks were exact after inverse mapping and
canonical record sorting. The three critical-path tests cover censoring,
TRAIN-only preprocessing and the exact-constant repair. There were zero
classifier fits, zero label-indexing operations and no new Outer access.

Six groups are computationally eligible for a separately registered matched
predictive comparison. That comparison must give the same preprocessing to
old and new physiology controls, preserve patient isolation and allow an empty
new-feature set. This eligibility result does not overturn the failed prior
nonlinear-readout continuation rule, adopt a new universal feature list, or
establish a Raw/M8 increment. Acquisition filters, exact historical creation
CLI and independent clinical onset remain outside the available provenance.

## Reproduction and artifacts

Initial protocol: Git commit `82bef22`; exact-constant amendment: `1402871`.
Run `python -m pytest test_measurement.py test_constant_safe.py -q`, then
`python run_constant_safe.py`, `python verify_qualification.py` and
`python export_public.py` in the pinned server environment. The original
`qualify.py` reproduces the disclosed failure and must not be mislabeled as an
accepted run. Trusted-source registries and individual measurements stay on
the server. `results/qualified/` contains only anonymous aggregate tables and
verification results; `results/UNIT_GATE_FAILURE_DIAGNOSIS.json` preserves the
failed gate. `python verify_release.py` checks the downloaded public bytes and
the protected previous releases locally.
