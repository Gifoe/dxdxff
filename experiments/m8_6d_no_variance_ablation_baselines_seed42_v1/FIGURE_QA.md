# Visible QA — six-feature experiment

Status: **PASS after actual inspection of all three final PNGs**, not inferred from rendering.

| Issue | Artifact | Severity | Fix | Final status |
| --- | --- | --- | --- | --- |
| AUROC tick labels too dense | Initial feature contribution panel | Medium | Uniform MaxNLocator(nbins=4) for every panel, without changing data or ranges | Resolved in final v2 |
| Code-style classifier names | Classifier PR figure | Low | Natural display labels; internal model IDs unchanged | Resolved |

Final previews inspected:

- FEATURE_CONTRIBUTION.png:2430×810, six additional feature deletions.
- CORE_PAIRED_INTERVALS.png:2430×972, nine mechanism/branch comparisons and one interaction.
- CLASSIFIER_PR_INTERVALS.png:2430×810, seven paired classifier comparisons.

All row labels, titles, ticks, intervals, axes and captions are legible and unclipped. Dashed zero lines and gray uncertainty remain visible in grayscale. Feature/core signs are Full-minus-ablation; classifier signs are PR-minus-noPR. The interaction is distinguished by a diamond and explicitly identified as a separate contrast. Fixed protocol ordering is retained, not sorted by favorable results.

Nominal95%paired intervals,80patients,10000draws and exploratory repeated-test status are stated. The separate37-comparison Macro-F1 intervals remain in tables, not misrepresented as the plotted intervals. SVG text is live and PDF is vector output. Source/renderer/export hashes are traceable.

All21numeric CSVs have identical hashes between the preserved initial package and the final layout revision. No numeric outcome, cohort, model, threshold, statistical family or inference was changed during visual repair. The main technical figures adapt the recipe's direct-label and semantic-color grammar to uncertainty-first plotting rather than decorative rankings.
