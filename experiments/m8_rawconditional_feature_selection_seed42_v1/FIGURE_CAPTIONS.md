# Internal Raw-conditional selection figures

## Figure 1. Added physiological cardinality does not identify one stable subset

**a,b**, Patient-equal AUROC and Macro-F1 of fixed-alpha ridge heads conditioned
on 192 frozen Raw coordinates. The blue line weights the five fold-specific
means by their assessment patient counts. Each k contains255 appearances of79
unique patients. Grey lines are the five original folds; their assessment
patients overlap across folds. Open circles mark each fold's own-FIT cardinality
decision, not the maximum of this pooled curve. **c**, All32 physiological
measurements and all five folds are retained. Values are selection frequencies
over10 training-patient weight bootstrap re-selections at the chosen k. The
encoder, preprocessing, calibration set and feature-choice set remain fixed.
The all-zero second column reflects its k0 choice, not a failed run. One group
has two DELTA mean/SD coordinates. No error bars, test or multiple-comparison
claim is attached to these descriptive curves. Source data are
`SOURCE_CARDINALITY.csv`, `SOURCE_SELECTION_FREQUENCY.csv`,
`SOURCE_WEIGHT_STABILITY.csv` and `SOURCE_FINAL_FOLD_OVERLAP.csv`.

## Figure 2. Conditional internal increments vary by metric and source

**a**, Selected-minus-Raw-only patient metric means and unadjusted95% percentile
intervals from10000 patient-ID cluster resamples. Repeated patient appearances
are carried together. Cardinality was selected using these own-FIT assessment
sets, so intervals condition on the selected procedure and are not an unbiased
test of the complete selection algorithm. Raw denotes the192D frozen-encoder
ridge readout here, not the original end-to-end Raw-only Outer model. EZ-frac.
bias is signed predicted-minus-true EZ fraction; a positive contrast is not
universally an improvement. **b**, AUROC and EZ-F1 contrasts in all four known
source strata. Counts are unique patients, not appearances; they total79.
No source-specific threshold or model is used. Source data are
`SOURCE_CONDITIONAL_INCREMENT.csv` and `SOURCE_SOURCE_INCREMENT.csv`. There
is one encoder training seed. Neither panel reports new Outer inference or
leave-source-out generalization.

Both figures use Python/matplotlib,183 mm final width, editable SVG/PDF and
600 dpi PNG/TIFF. All observations come from actual verified aggregates; no
patient/channel identifiers or private predictions are exported.
