# Experiment B: six-feature predictive informativeness

Frozen before new fits or new Outer scores. No DREChannNet/raw neural features,
feature search, EEG extraction or changes to other experiments. This is exploratory
cross-patient prediction on a historically viewed cohort and selected feature set.

## Inputs, roles and models

Reuse Experiment A's SHA-locked R14 and label/mapping archive. Keep R14 indices
0–7,10–13, exactly the six measurements and mean/SD definitions audited in A.
These are DELTA summaries averaged over all valid pre/post windows then seizures,
not redefined 0–20 s measurements. Native units/floors remain unchanged.
Archive NEZ=1,EZ=0 becomes EZ-positive y=1-y. All80patients/7635channels/
1743EZ/5892NEZ/256seizures/1471965valid windows remain. No individual is removed.

Use the exact original five-fold roles. FIT/CAL/Outer patient counts are
51/13/16,51/13/16,50/13/17,52/13/15,51/13/16. Outer channel counts are
1631,1498,1746,1425,1335. All80 are Outer exactly once, with disjoint patients
inside each fold. All14 configurations share roles and inner splits.

Main eight rows: six single-feature2D+PR models, all-six12D without PR, and
all-six12D+PR. Supplement: all six single-feature2D without PR models. Every
row is retained. Each uses sklearn LogisticRegression, L2,lbfgs,max_iter2000,
tol1e-6,class_weight=None,random_state42,warm_startFalse.

## Preprocessing and weights

FIT-only mean imputation; an all-missing FIT coordinate falls back to0 without
dropping it. Infinities fail input admission. Unit scale comes from a FIT-only
StandardScaler on raw imputed coordinates. Current DREChannNet PR uses each
patient's own complete unlabelled channels: float64 mean/populationSD; SD greater
than1e-8 times FIT unit scale is supported, otherwise finite values become0.
NaNs remain missing until a FIT-only post-PR mean imputer. Final StandardScaler
fits FIT only for **both** PR and NoPR. Apply original float32 bank rounding
then sklearn float64 in both versions. PR calculation never consumes labels,
EZ prevalence or cross-patient/source test statistics. All inner sub-FITs fit
their own entire preprocessing chain. No outer/global scaler reuse inside CV.

Operations are coordinate-wise; single-feature classifiers consume only their
two selected coordinates. Admission checks inactive-feature invariance and
bit-exact parity with current DREChannNet PR. The old absolute-epsilon PatientZ
is not substituted. Seven single-seizure patients' SD coordinates stay constant
zero under PR; variation/support counts are reported rather than filtering them.

For N FIT channels and P FIT patients, w_i=N/(2P*n_patient,class). Each patient
has equal total weight N/P, and each within-patient class has half that total.
Weights have mean1, keeping the C regularization scale fixed. Recompute weights
on every sub-FIT and on shuffled training labels. No additional class_weight.
Scaler fitting is unweighted and label-free, consistent with the current model.

## FIT-only C selection and CAL-only thresholds

Candidates C=[0.01,0.1,1,10]. Three patient-isolated inner splits come from
sorted FIT patient IDs shuffled by NumPy default_rng(42+1009*outerfold), then
array_split into three. Identical roles across all14 arms and all4 Cs.
Every C yields one held-out probability per FIT channel; selection averages
AUROC over **every held FIT patient equally**, not the three subfold means.
Within1e-12 of best AUROC choose smaller C. Refit the selected C on FIT only.
Do not refit on FIT+CAL. PR and NoPR have the same grid/budget but select C
separately under the same policy, not a matched fixed-C causal comparison.

Use CAL patient-average Macro-F1 to select a numeric probability threshold from
201points0..1 step0.005. Ties: round MacroF1 and EZF1 to12 decimals, then higher
EZF1, nearest0.5, lower threshold. Predict EZ when pEZ>=threshold. No Outer label
or metric may affect C, thresholds, direction, stopping or row selection.
Record all convergence warnings/n_iter; never silently raise the2000 budget.

## Fixed negative control and execution barriers

100 permutations b=0..99, all-six+PR only, five original folds each. RNG is
default_rng([42,20261025,fold,b]); shuffle labels within FIT patient preserving
each EZ count. Recompute balanced weights using permuted labels: retaining
original label-position weights would leak original labels and is forbidden.
Use that fold's legally selected full+PR C and the identical label-free FIT
preprocessor, with a separate threshold chosen only on true CAL labels. Do not
retune C or flip score direction. True Outer labels are never permuted.
This is a conditional negative-control distribution, not an exact unrestricted
randomization test because hyperparameters were selected on true FIT labels.

Formal unique LR fits:840inner(14x5x4x3)+70selected(14x5)+500null(100x5)=1410.
Unit-test synthetic fits are separate engineering checks, not cohort evidence.
Freeze all70selected and500null models/preprocessors/thresholds before any new
Outer prediction. Seal all570 score cells before calculating aggregate results.
Historical Outer exposure is disclosed; these barriers are not a blinding claim.

## Endpoints and intervals

Patient-average AUROC,AP,Macro-F1,EZ-F1,NEZ-F1,Accuracy,MRR,Top-1. AUROC/AP use
continuous pEZ; ranking is within-patient, descending unrounded pEZ with original
canonical-channel lexical tie breaking. Equal80-patient means are primary;
fold and source summaries are separate. No pooled-channel substitution.

10,000seed42 equal-patient bootstrap percentile95% intervals, with trained models
and split assignments fixed. Paired differences resample patients jointly.
Seven prespecified Full+PR-minus-control contrasts cover all six single-feature
PR arms and FullNoPR. Report all56endpoint comparisons with nominal95%CI.
The four primary metrics AUROC/AP/MacroF1/Top1 form one28-comparison family
(7x4): additionally report Bonferroni percentile intervals at1-0.05/28. The
adjusted tails use about9 of10000 bootstrap order statistics; extreme bounds
have limited Monte Carlo precision. Other endpoints and six single-feature PR
improvements are descriptive nominal intervals. No favorable-control selection.

Null AUROC should be described relative to0.5, with the full100-repeat distribution
reported regardless of outcome. AP is not expected to be0.5: give patient-equal
EZ prevalence as a reference; finite random rankings can have a different exact
expected AP. Null quantiles describe training randomization, not patient CIs.

## Independent checks and interpretation

Replay every840candidate selection and570CAL thresholds. Independently reconstruct
saved-model probabilities with coefficient dot products and expit; independently
compute all9120patient rows/72960endpoints using pairwise AUROC,tied-score AP,
confusion counts and canonical ordering. Recompute intervals with bootstrap
count matrices and audit allfold/source/paired/negative summaries. Freeze/source/
input hashes and exact prediction ledgers must match before release.

A measures association; B tests held-out-patient prediction by a simple linear
classifier. Neither historical reanalysis establishes external generalization,
causality or irreducible conditional contribution of each feature. Combining
features or PR is not required to improve every metric. Preserve any negative
or inconclusive comparisons. Publication tables bold only true unrounded maxima.
All per-patient/channel predictions,identities,models,preprocessors and logs stay
server-only. Public outputs are explicit anonymous aggregate allowlists.
