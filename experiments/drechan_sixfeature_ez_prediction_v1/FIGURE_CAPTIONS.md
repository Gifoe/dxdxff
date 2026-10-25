# Figure and table captions

## Figure B. Physiological feature predictive performance

Dots show equal-patient mean AUROC across 80 Outer patients in the original
five patient-held-out folds; horizontal lines show nominal 95% percentile
patient-bootstrap intervals from 10,000 seed42 draws. The dashed line marks
AUROC0.5. All inputs use current feature-wise patient-relative normalization
except All Six w/o PR (orange open square), which retains FIT-only global
standardization. Other rows are blue circles. Each individual feature contributes
the original across-seizure mean and population SD, and All Six uses all12
coordinates. Partial high gamma covers80–125Hz. C is selected only within FIT;
CAL determines thresholds. Primary paired comparisons use the separately reported
28-comparison family intervals, not these absolute nominal intervals. This
historical-cohort reanalysis is exploratory cross-patient prediction.

## Supplement. Training-label permutation control

The histogram retains all100 fixed all-six+PR training-label permutation
repetitions, evaluated on the true Outer labels of all80 patients. The grey
dashed line marks chance AUROC0.5 and the blue line true-label full+PR training.
FIT labels were shuffled within patients with class counts preserved and sample
weights recomputed. C and label-free preprocessing follow the legal true-FIT
selection; each null model has its own true-CAL threshold. The distribution
describes this conditional negative control, not patient-bootstrap uncertainty.
