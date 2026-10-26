# EZ-conditioned PR gains and reference-composition sensitivity

Implementation of two exploratory supplements to the completed all80 reference-
transfer experiment. **Real-data execution is not yet complete.** The trusted
server recovered after initial implementation. Metadata-only access verified the
exact frozen reference SHA,all80 patients/7,635 channels,and40-213 channels per
patient before any new endpoint calculation.

The frozen plan covers all80 patients/7,635 channels,12 unchanged physiological
coordinates andfive reference arms. E1 reuses the original100 bidirectional halves
to evaluate EZ/NEZ-specific CRPS andthe primary differential gain H. E2 holds out
each channel,using25/50/75/100percent ofOTHER channels as nested references. Its
primary D comparesFull25 versusFull100 CRPS. A within-query centered composition
slope diagnoses association with reference EZ proportion.

The two primary endpoints receive nominal95percent andadditional97.5percent family
intervals;eleven secondary slots receive Bonferroni intervals. All curves andfour
sources remain descriptive. Original absolute-atypicality results retain their
original six-family correction;new H cannot replace that inconclusive result.

## Trusted execution

Set existing `PRM_RECONSTRUCTION`, `PRM_LABELS`, `PRM_FEATURE_PR`, `PRM_UNIT_PR`,
`PRM_INTERNAL_BANKS` aliases. Set `PRCOMP_PARENT_RUNTIME` to the completed trusted
parent runtime and`PRCOMP_SOURCE_COMMIT` to the pushed new source revision.
Use a new nonexistent runtime,one BLAS thread andCPU-only computation:

```powershell
python -m unittest discover -s experiments/drechan_pr_ez_composition_v1/tests -v
powershell -NoProfile -File experiments/drechan_pr_ez_composition_v1/run_all.ps1 -Runtime <new-private-runtime> -Python <trusted-python>
```

Stages admit/freeze/evaluate/verify/report/package bind parent,input andsource
hashes. All nested reference indices/moments freeze without labels before new
endpoints. Raw data,identities,individual statistics andreference estimates remain
trusted-side. Only an exact anonymous aggregate allowlist is published.

The patient mapping is unchanged:79 original-FIT paths andone predeclared original-
validation-only exception;every person absent from their own TRAIN/CAL. Single-
seizure patients andconstant coordinates remain. Statistical background is not a
proven healthy baseline;neither H nor composition slopes establish localization
performance or biological causality.
