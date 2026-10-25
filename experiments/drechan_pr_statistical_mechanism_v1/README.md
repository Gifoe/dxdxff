# Patient-relative normalization: statistical mechanism identification

Status: protocol and implementation prepared; real-cohort execution pending.
No new ASSESS endpoints have been calculated at this source revision.

This isolated, CPU-only study uses all 80 historical patients exactly once in
held-out internal statistical assessment. The scientific endpoint is signed
transferable standardized EZ-minus-NEZ separation, not classifier performance.
It compares global preprocessing, patient centering, patient scaling and both,
with matched-donor, TRAIN-label permutation and operational-PR diagnostics.

The frozen design is in [PROTOCOL_LOCK.md](PROTOCOL_LOCK.md) and its machine-readable
companion. Historical internal roles are audited but replaced for this study by
a label-free, pre-outcome 80-patient assignment restricted to each path's original
outer-FIT population. Original model roles, outputs and experiments are unchanged.
Each person is absent from TRAIN/CAL in their own assessment path; CAL is unused.

Execution requires environment-configured trusted inputs; no private server
locations or individual data are included in this repository. Set
`PRM_RECONSTRUCTION`, `PRM_LABELS`, `PRM_FEATURE_PR`, `PRM_UNIT_PR`,
`PRM_INTERNAL_BANKS` and `PRM_SOURCE_COMMIT` to authorized inputs and the pushed
source commit. Set OMP/MKL/OpenBLAS threads to 1. Use:

```powershell
python -m unittest discover -s experiments/drechan_pr_statistical_mechanism_v1/tests -v
powershell -NoProfile -File experiments/drechan_pr_statistical_mechanism_v1/run_all.ps1 -Runtime <private-runtime> -Python <trusted-python>
```

The launcher runs admission, freezes all 7,070 directions before ASSESS, evaluates
and seals all endpoints, generates anonymous tables, independently reconstructs
all results, and packages only an explicit aggregate allowlist. Individual
statistics, preprocessing, directions, identities and logs remain trusted-side.

The analysis is exploratory repeated-cohort cross-fitting. It is not external
validation, a causal study, or evidence that PR creates within-coordinate ranking
information. Centering identities are checked rather than presented as discoveries.
