# Patient-relative normalization: statistical mechanism identification

Completed with real computation and independent numerical verification. All80
patients receive one held-out assessment, retaining all7,635 channels and four
source counts. No classifier or neural model was trained.

## Main findings

The result is mixed (prespecified Outcome D), not a uniform benefit of PR.

| Direct mean-difference probe | Global | Center only | Scale only | Full PR |
| --- | --- | --- | --- | --- |
| Transferable separation T | 0.94424 | 0.94424 | 0.87515 | 0.87515 |
| Direction alignment A | 0.51790 | 0.51790 | 0.59712 | 0.59712 |
| Class-midpoint MAE, lower better | 1.32975 | 0.29801 | 19.20898 | 0.26445 |

Primary Full-minus-Global ΔT=−0.06909,95% CI[−0.13636,0.00063]; only29/80
patients improve. ΔA=0.07922 with additional25-slot-adjusted interval
[0.02230,0.14198]. Centering explains most midpoint stabilization; its T/A
equalities are mathematical identities. Full PR's extra midpoint benefit over
centering is inconclusive. Fisher sensitivity reverses the alignment effect.
Matched donors perform worse but severely distort input variance, so that
comparison cannot establish physiological reference specificity.

These results do not support claiming that PR increases transferable
discriminative signal-to-noise. They support the narrower distinction between
coordinate stabilization and discriminative separation.

See [RESULTS.md](RESULTS.md) for complete tables/intervals and controls,
[FINDINGS.md](FINDINGS.md) for the seven research answers, and
[the four-panel figure](figures/pr_statistical_mechanism.pdf).

## Design and80-patient coverage

This isolated, CPU-only study uses all80 historical patients exactly once in
held-out statistical assessment. The scientific endpoint is signed
transferable standardized EZ-minus-NEZ separation, not classifier performance.
It compares global preprocessing, patient centering, patient scaling and both,
with matched-donor, TRAIN-label permutation and operational-PR diagnostics.

The frozen design is in [PROTOCOL_LOCK.md](PROTOCOL_LOCK.md) and its machine-readable
companion. Historical internal roles are audited but replaced for this study by
a label-free, pre-outcome 80-patient assignment:79 within original outer-FIT and
one original-validation-only exception for the patient never in FIT in any path.
This exception was identified from metadata before any new endpoint computation.
Original model roles, outputs and experiments are unchanged.
Each person is absent from TRAIN/CAL in their own assessment path; CAL is unused.
A separately declared strict79 sensitivity cannot replace the primary80 analysis.
The strict79 sensitivity also does not support improvement. Its exact table stays
trusted-side because80/79 complementary means expose the omitted person's endpoint.
Its qualitative negative conclusion and independent-verification commitment are
public; the80-person primary interval remains the main inference.

## Reproduction and verification

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

The pre-inference source commit is`49db6da4084635bade1cf64e3da10a5ed890e8fb`.
Its eight scientific files remain byte-identical. The independent replay checks
717,440 endpoint values,1,112 nominal intervals and25 additional-family intervals;
maximum endpoint discrepancy1.14×10⁻10. Nine mathematical/isolation tests pass.

The authenticated release contains18 privacy-filtered payloads plus manifest under
`results/`. `release_check.py` authenticates the whole archive, source hashes,
allowlist, privacy boundary and schemas **before extraction**. Public figures
and full tables can be regenerated from those aggregates:

```powershell
python experiments/drechan_pr_statistical_mechanism_v1/report_public.py
python experiments/drechan_pr_statistical_mechanism_v1/plot_figures.py
python experiments/drechan_pr_statistical_mechanism_v1/verify_public_release.py
```

`figures/` includes vector PDF, editable SVG and600dpi PNG for the four-panel
main figure, signed patient-effect bin distribution and all four-source results.
The schematic panel is explicitly labeled, and all quantitative panels use real
anonymous aggregates. Figure/statistics definitions are in[FIGURE_QA.md](FIGURE_QA.md).

The analysis is exploratory repeated-cohort cross-fitting. It is not external
validation, a causal study, or evidence that PR creates within-coordinate ranking
information. Centering identities are checked rather than presented as discoveries.

`sanitize_public_export.py` is a mandatory publication-only step after the frozen
calculation. It removes the complementary79-person table and retains all80 primary
and negative-control outputs unchanged. Only its filtered archive may be exported;
the original frozen package is a trusted intermediate, not a public release.
