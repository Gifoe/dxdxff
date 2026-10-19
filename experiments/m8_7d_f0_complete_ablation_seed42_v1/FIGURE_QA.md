# Visible figure QA — 2026-10-19

Status: **PASS after actual visible PNG inspection**, not inferred from successful rendering.

Inspected both exported previews:

- `results/figures/FEATURE_CONTRIBUTION.png` (2,430 × 810 pixels).
- `results/figures/PAIRED_CORE_INTERVALS.png` (2,430 × 1,038 pixels).

The seven feature rows and eleven core/interaction rows are legible, in the fixed protocol order, with no clipped labels or overlapping titles, axes or captions. Each panel has its own numeric metric-difference scale and dashed zero reference. All intervals, including those crossing zero, are visible; neither significance badges nor a favorable-outcome sorting rule was introduced. The final core row is visibly distinguished by a diamond and explicitly described as a factorial interaction, not Full-minus-ablation.

Both captions state patient pairing, 10,000 draws and exploratory repeated-test status. The figures show nominal 95% intervals; the report separately supplies approximate 18-comparison Macro-F1 family intervals. Source numbers were independently checked before rendering and remain bound by `RENDER_MANIFEST.json` and `PACKAGE_MANIFEST.json`. All source and figure hashes match their server export. Editable SVG and vector PDF are retained alongside PNG; SVG text remains live.

No plot changed a model, threshold, feature set, cohort or outcome. No private identities or individual scores enter these graphics.
