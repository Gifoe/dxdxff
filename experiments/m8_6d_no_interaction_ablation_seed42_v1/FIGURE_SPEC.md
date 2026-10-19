# Aggregate figure contract

Two deterministic full-width quantitative figures, generated only after actual fourteen-group independent verification. These are result figures, not architecture illustrations; no generative image model or private material is needed.

| Figure | Question | Rows | Panels | Source |
| --- | --- | --- | --- | --- |
| Core paired intervals | How do approved module removals and branch restrictions change the fixed main model? | Seven non-feature contrasts in protocol order | Patient Macro-F1, AUROC, AP differences | `results/INTERVALS.csv` |
| Feature contribution | What is each remaining measurement's conditional contribution with the same Raw branch? | Six single-feature deletions in protocol order | Same three differences | `results/INTERVALS.csv` |

All intervals are nominal paired95% patient-bootstrap,10000draws,80patients. Positive means main-minus-ablation; every row, including negative differences and intervals crossing zero, is retained. No significance coloring. The13-comparison Macro-F1 intervals remain separately available in the numeric table; nominal figure intervals do not imply corrected significance. No Early x Late interaction, isolated Late Gate deletion, interaction add-back or unsupported claim appears.

Use direct row labels, lowercase panel indices, natural English titles, zero reference lines and readable4-tick axes. Markers blue `#0072B2`, intervals gray, neutral grid and white background; meaning survives grayscale. Fixed row order, no ranking by observed result. EditableSVG/live text and vectorPDF, plusPNGpreview. Caption states cohort, contrast direction, uncertainty and exploratory repeated-test status. Traceability manifest records source CSV hashes and output hashes. Inspect both actual renderedPNGfiles for clipping/overlap and axis readability before release.
