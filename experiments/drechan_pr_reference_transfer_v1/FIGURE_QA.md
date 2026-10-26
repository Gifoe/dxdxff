# Figure provenance and caption

Figure: `figures/pr_reference_transfer.pdf` (vector), matching editable SVG and
600 dpi PNG. All numerical values originate in the authenticated aggregate
package; no patient-specific values or invented measurements are drawn.

**Caption. Patient-specific reference transfer before classification.**
(A) Design schematic. Each patient's channels are partitioned into two disjoint
sets. One set estimates unlabeled location and scale; Gaussian marginal forecasts
are evaluated on the other set. Roles are exchanged over 100 fixed partitions.
Disjoint sets are not assumed statistically independent. (B) Held-out CRPS,
averaged within each patient before the equal-patient mean across all 80 patients.
Intervals are nominal 95% paired, source-stratified patient-bootstrap intervals.
All forecasts are evaluated in one TRAIN-derived coordinate system. (C) Paired
CRPS reductions with additional six-family Bonferroni intervals; positive values
favor the method on the right of each contrast. (D) Mean absolute relative
deviation in EZ and NEZ query channels, with nominal 95% intervals. The paired
EZ-minus-NEZ difference and adjusted interval are printed separately. Its adjusted
interval includes zero; marginal EZ/NEZ bars are not a significance test.

Panel A is explicitly schematic. Panels B-D contain real aggregate estimates.
Each person contributes once to inference after averaging partitions. All12
coordinates and all80 patients are retained. Zero-scale CRPS and unsupported PR
coordinate behavior are defined separately in the frozen protocol. The study is
exploratory on a historical cohort with fixed overlapping TRAIN sets, not external
validation or biological causal identification.

Bindings: B=`BACKGROUND.csv`/ALL/ALL12; C=`CONTRASTS.csv`/ALL/ALL12/secondary
family bounds; D=`ANOMALY.csv`/ALL/ALL12, nominal EZ/NEZ and adjusted gap.
The PNG is 600 dpi, PDF fonts are embedded, SVG text remains editable. A visual
inspection checks axis labels, uncertainty bars, panel captions and absence of
clipping. Final public release verification checks source and aggregate hashes.
