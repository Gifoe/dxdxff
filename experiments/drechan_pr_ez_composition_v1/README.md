# EZ-conditioned PR gains and reference-composition sensitivity

**Completed and independently verified on all 80 patients and 7,635 channels.**
These two exploratory supplements reuse the original 12 physiological coordinates
and 100 fixed channel rankings. Each person has 40-213 channels. No classifier,
new feature extraction or favorable subset selection was used.

| Primary endpoint | Estimate | Two-primary adjusted interval |
| --- | ---: | --- |
| H: NEZ gain minus EZ gain | 0.017255 | [-0.006148, 0.042374] |
| D: Full 25% minus Full 100% CRPS | -0.006701 | [-0.022525, 0.000690] |

Both primary intervals include zero. NEZ and EZ both receive lower CRPS under
Full PR; their differential benefit is unresolved. The reference-size comparison
does not establish either deterioration or equivalence. All eleven adjusted
secondary intervals also include zero, including the three within-query
reference-EZ composition slopes. The original absolute-atypicality result remains
inconclusive under its unchanged six-family correction.

See [RESULTS.md](RESULTS.md), [中文结论](CHINESE_SUMMARY.md),
[FINDINGS.md](FINDINGS.md) and the [four-panel figure](figures/pr_ez_composition.pdf).

E1 reuses the original 100 bidirectional halves and exact frozen reference moments
to evaluate label-conditioned CRPS. E2 holds out each channel, using 25%, 50%, 75%
and 100% of OTHER channels as nested references; the query is always excluded.
Within-query centering controls fixed patient/query differences in the reference
composition analysis. No verifiable anatomical coverage metadata exists in the
frozen cache, so channel counts are not interpreted as implantation range.

The two primary endpoints receive nominal 95% and additional 97.5% family intervals;
eleven secondary slots receive Bonferroni intervals. All four sources and all
coordinates are retained descriptively. Original absolute-atypicality tables are
copied byte-for-byte and keep their original correction.

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

## Verification and release

Scientific source `b12952d9b8abce3b280dda7fa704671bb78e60fc` was pushed before
actual evaluation. Nine scientific files and ten parent dependencies remain
hash-frozen. Independent direct replay passed 518,880 endpoint values, 8,793
interval rows, both primary intervals, all eleven secondary intervals, and
20,000 aggregate permutation values. Maximum endpoint discrepancy is 2.57e-13.

The authenticated archive contains 21 anonymous payloads plus manifest, SHA256
`0903b16a7269527d1d0f36cd076720a6a37e0dc462b5e371ee6491fa39f503f4`.
Local whole-archive, allowlist and source checks passed before extraction.
Thirteen tests include synthetic end-to-end statistics, query isolation and
watermarked temporary figure QA. Real PDF/SVG/600dpi PNG figures consume only
verified aggregates. Existing studies and individual trusted data remain unchanged.
