# Figure QA

Both actual generated PNG previews were inspected after independent result verification, alongside the source interval table. SVG retains live text; PDF is vector; PNG is a preview. `results/FIGURE_MANIFEST.json` records exact source/script/output hashes and completed visible-review status.

| Artifact | Check | Result |
| --- | --- | --- |
| CORE_PAIRED_INTERVALS,2520×835PNG | Seven approved core contrasts; main-minus-ablation direction; correct three metrics and nominal95% caption | PASS |
| FEATURE_CONTRIBUTION,2520×827PNG | Six remaining measurements;80–125Hz and10–125Hz labels; no Variance/add-back/isolated-Late row | PASS |
| Both | Titles, direct labels, ticks, footer and all interval endpoints visible without clipping or overlap | PASS |
| Both | Four-tick axes; dashed zero line; blue points/gray intervals; marker/line/direct-label meaning does not depend on red/green discrimination | PASS |
| Both | Natural English casing and canonical Macro-F1/AUROC/AP/PR/RMS labels; editable SVG text and PDF export | PASS |
| Both | Fixed protocol row order, no significance coloring or observed-outcome sorting; repeated-test and80patient/10000draw disclosure | PASS |

The figures show nominal intervals, not the additional13-comparison Macro-F1 intervals; corrected-level values remain in the report/CSV. No numeric table, CI or prediction was altered for plotting. No image-generation service received private or public result data.
