# Findings: centering, scaling and transferability

## Conclusion

The results are mixed (prespecified Outcome D). Patient-relative scaling increases
cosine alignment under the primary mean-difference probe, but does not increase
transferable standardized EZ/NEZ separation. Centering stabilizes the shared
class-midpoint coordinate. That stability partly reflects generic coordinate
recentering, not newly created within-coordinate ranking information. The Fisher
sensitivity reverses the alignment result, limiting its generality.

## Seven research questions

1. **Does a transferable physiological direction exist?** Within this historical
   probe, yes. Global T=0.94424 (95% CI0.74719–1.15027) and A=0.51790
   (0.42600–0.60417), with positive T in72/80 patients. Observed values exceed
   the central95% TRAIN-label-shuffle reference ranges. This is evidence of a
   label-dependent held-out association, not new-cohort confirmation.
2. **Does PR improve alignment or separation?** Direct alignment improves:
   ΔA=0.07922,25-slot adjusted interval0.02230–0.14198. The primary separation
   does not: ΔT=−0.06909,95% CI−0.13636–0.00063; only29/80 improve.
   Four-source T differences are all negative. A positive angular effect must
   not be substituted for the unsuccessful primary endpoint.
3. **Centering, scaling or both?** Centering cannot alter class-difference vectors,
   T or A in the fixed coordinate system; this is an identity. Scaling causes
   the direct angular change. Centering reduces midpoint MAE from1.32975 to
   0.29801; Full PR gives0.26445. Full versus center-only's additional reduction
   0.03357 has adjusted interval−0.05975–0.13668, so added scaling benefit for
   this endpoint is unresolved. Scale-only greatly worsens midpoint MAE to19.20898.
4. **Ranking geometry or shared decision coordinate?** The clearest stable role
   is coordinate recentering. Positive affine transformations preserve each
   coordinate's ranks. Direct joint angular alignment improves, but T does not,
   and Fisher alignment decreases (ΔA=−0.11456). No downstream ranking metric
   was used to select or reinterpret the result. Midpoints are statistical
   diagnostics, not clinical decision thresholds.
5. **Does the correct reference beat matched alternatives?** It beats the
   prespecified source/channel-count donor: ΔT=0.44513, adjusted interval
   0.16912–0.75817; ΔA=0.42862,0.24338–0.61390. All80 source matches are legal,
   with74 channel-bin matches. However donor mean input variance reaches
   618,239.51 in one fold, versus approximately1 for correct PR. This severe
   induced distribution shift prevents a physiological-specificity conclusion.
   These are source/count-matched alternatives, not variance-matched controls.
6. **What remains inconclusive?** Improved transferable T; alignment independent
   of the direction estimator; Full PR's extra drift benefit over centering;
   reference specificity independent of donor variance distortion; clinical
   threshold usefulness; and generalization to new cohorts. Centering reduces
   drift even under matched label shuffles. Observed MAE reduction exceeds their
   central reference range, but those ranges are not adjusted population tests.
7. **Is a core methodological contribution established?** Not as a claim that
   PR increases transferable discriminative signal-to-noise. The primary endpoint
   fails to support that claim. This experiment supports a narrower account of
   centering-driven coordinate stabilization and probe-dependent scaling geometry.
   It does not establish algorithmic novelty, causality, clinical effectiveness
   or necessity of PR. PR may be described as a preprocessing component with
   these measured effects, not promoted to a universal information-enhancement
   mechanism.

## Identities versus empirical findings

**Mathematical:** positive patient-wise affine transforms preserve single-coordinate
ranks; G0/G1 and G2/G3 have identical T/A for the specified estimators. Centered
unlabeled feature means being zero is not an EZ-related discovery.

**Empirical:** primary direct ΔA is positive, primary ΔT is negative but its80-person
interval includes zero, center-only/full reduce midpoint drift, and donor mismatch
causes substantial variance distortion. Fisher and source differences remain visible.

**Bounded interpretation:** patient-relative preprocessing changes shared-coordinate
geometry, but angular consistency and standardized discriminative separation are
different properties. Improved angular consistency alone does not show more useful
transferable EZ information.

## All80 and historical-cohort scope

Every patient is assessed once, retaining all7,635 channels and four source counts.
79 paths lie within original FIT; one never-FIT pediatric patient uses an explicitly
declared original-validation-only path. This exception was resolved from metadata
before outcomes, not selected after observing effects. Primary results remain80.
The predeclared strict79 sensitivity also does not support improved T and does not
replace the primary analysis. Its exact table stays trusted-side:80/79 complementary
means reveal the omitted patient's endpoint. The negative conclusion and private
verification commitment remain public. All findings are exploratory repeated-cohort
associations.

## Paper-ready bounded statement

> In a patient-held-out statistical probe of80 historical patients, patient-relative
> normalization increased cosine alignment under a mean-difference direction
> estimator (ΔA=0.079, multiplicity-adjusted interval[0.022,0.142]) but did not
> improve transferable standardized separation (primary ΔT=−0.069,95% CI
> [−0.136,0.001]). Centering accounted for most of the reduction in class-midpoint
> displacement. These findings distinguish coordinate stabilization from increased
> discriminative information; angular effects depended on the direction estimator.

The methods text must also disclose the79 original-FIT assessments plus one
predeclared validation-only exception, historical feature development and the
conditional nature of the patient-bootstrap intervals.
