# Figure contract and visual QA

## Scientific contract

Core conclusion: PR's measured angular alignment gain does not imply improved
transferable separation; centering primarily stabilizes the midpoint coordinate.
Archetype: schematic-led quantitative composite. Main quantitative evidence is
the prespecified direct-estimator all80 comparison, not a selected subgroup or
the more favorable Fisher T point estimate.

- Panel a: analytic schematic of heterogeneous unlabeled patient coordinates and
  the PR transform. The curves are explicitly NOT patient observations, class
  densities or estimated physiology. No synthetic curve is used in quantitative panels.
- Panel b: all four factorial arms' patient-mean T with nominal95% intervals and
  the primary paired Full-minus-Global interval.
- Panel c: all four arms' A and the declared25-slot-adjusted paired ΔA interval.
- Panel d: all three conditional Global-minus-arm MAE reductions, including the
  unfavorable scale-only result. Solid lines are25-slot-adjusted patient intervals;
  dashed lines are central95% reference ranges across100 matched-H3 label-null
  repeats, not confidence intervals. Its symlog axis is linear within±2, needed
  to retain the large negative scaling-only effect and the smaller positive effects.

Supplement1 shows all80 paired T effects via fixed aggregate bins, with unequal
bin widths labeled. No patient points or identities leave the server. Supplement2
shows all four factorial arms in each of the four sources, with nominal arm-wise
intervals and actual n36/21/15/8. There is no source selection or pooled-channel inference.

## Statistical and export contract

n=80 unique patients, one assessment each;79 within original FIT plus one explicit
original-validation-only exception, absent from own TRAIN/CAL. Main estimator is
the equal-TRAIN-patient mean class-difference direction, epsilon factor1e−8.
All directions are fixed before assessment. Intervals use10,000 seed42 paired
patient draws, conditional on fixed directions/paths. Algebraic centering equalities
are not significance tests. The25-slot family uses percentile1−0.05/25 intervals.

Python/matplotlib is the saved selected backend and is used for every plotted
element. Native PDF and editable-text SVG accompany600dpi PNG. Figure width is
182.9mm before tight bounding; all text≥6.2pt. Arial/sans-serif, restrained color
families, direct labels and distinct line styles provide redundant encoding.
Tight-export margins can change the final page width slightly; scale proportionally
to183mm in the manuscript. Source data are the authenticated aggregate CSVs.

## QA outcome

The figure-source preflight reports13PASS,0FAIL,1WARN. The warning is PNG rather
than TIFF; PNG is the user-requested600dpi raster and vector PDF/SVG are the main
publication outputs. No journal-specific submission-policy claim is made.

All three rendered figures were inspected. The initial linear drift axis compressed
the centering/full intervals because scale-only was strongly negative; a labeled
symlog display retained all effects without clipping. No data or statistical
calculations were changed. SVG editable text, PDF embedded fonts, raster dimensions,
600dpi metadata and non-overlapping labels are checked in release verification.

## Caption

**Figure | Patient-relative coordinates do not guarantee greater transferable
separation.** a, Schematic of unlabeled patient-wise centering and scaling, not
patient observations. b,c, Patient-mean signed standardized separation T and
cosine alignment A for all four factorial representations; error bars show
nominal95% patient-bootstrap intervals. Insets report paired Full-minus-Global
effects, with the A interval additionally adjusted for the25-slot secondary family.
d, Reduction in mean absolute class-midpoint displacement relative to Global,
positive indicating less drift. Solid intervals use the25-slot adjustment;
dashed intervals show central95% reference ranges over100 matched TRAIN/ASSESS
label shuffles and are not patient confidence intervals. The symlog axis is linear
within±2. All quantitative panels use80 unique patients, one held-out assessment
each, fixed TRAIN-derived directions and10,000 paired bootstrap draws.79 assessments
lie within original FIT; the sole never-FIT patient uses a predeclared original-
validation-only path. Single-coordinate affine rank invariance and centering
equalities are identities, not physiological discoveries. Fisher sensitivity and
all four-source results are reported separately. Cohort and features were used
historically; this is exploratory statistical evidence, not independent validation.
