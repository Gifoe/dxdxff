# PR ranking, robustness, review efficiency and seizure sufficiency

All analyses are exploratory analyses of the already studied ictal cohort.
Reuse the one common historical five-fold run of the representation-versus-score
experiment; do not select runs or construct an ensemble. All 80 patients, 7,635
channels, 256 seizures and four sources remain in A-C. D uses one fixed >=3-valid-
seizure cohort and one common channel intersection per patient for every K/model.

Models: original 12D Linear SVM, ictal TimeConv-CNN (end-to-end NoPR/FullPR,
not the separate interictal experiment or frozen residual reader), final
six-feature no-guidance/no-head-interaction DREChanNet. A-C use NoPR, FullPR,
Score-Z. D uses NoPR/FullPR only. Existing checkpoints, preprocessors, folds,
validation thresholds, training precision and model math remain immutable.
The DREChanNet/SVM early outer package is a provenance-bound reuse; its Score-Z
native-logit inference is CPU FP32, and its NoPR/FullPR scores are original
sealed outputs. TimeConv's original NoPR/FullPR checkpoints and scores are
already completed; its ongoing additional Global/Center/Scale fits are not used
as final checkpoints here. Admission must prove these statuses and hashes.

Freeze all 45 selected logical cells and parent score seals before any new
analysis. Reuse sealed ranking anchors and class predictions. EZ is positive;
DREChanNet native NEZ logits are negated. Ranking ties use canonical channel
strings, ascending, matching the parent. Score-Z retains the exact NoPR ranking
anchor; all AUROC/AP/MRR/Hit@1/3/5 values must match NoPR per patient.

A metrics: AUROC, AP, MRR, Hit@1/3/5, Macro-F1 and EZ-F1. NEZ-F1, accuracy and
EZ fraction bias are retained for original result replay. Top-k hit means at
least one EZ, not EZ recall. B reports gains, median, positive/zero/negative
fractions (numeric zero tolerance 1e-12), source distributions and source means.
CM is the four-source equal mean; Worst is their metric-wise minimum. Ordinary
outer predictions grouped by source must never be labelled LOSO. Existing LOSO
experiments with unmatched arms do not establish a matched six-condition LOSO.

C fractional budgets: .05/.10/.20, max(1,ceil(q*C)); fixed budgets:1/3/5,
min(k,C). Report EZ recall, precision, hit, NEZ count, prevalence lift and actual
budget. Predeclare coverage curves q=.01,.02,...,1 plus the zero-review endpoint.
All metrics first aggregate per patient, then equal-weight patients.

D: retain every patient with >=3 valid frozen seizure records, fixing the
intersection of labelled channels with valid raw support and physiological
summaries in all candidate seizures. Record exclusions before outcome access.
For K=1 and2 enumerate combinations when <=50; otherwise select 50 unique
combinations without replacement using a patient/K-specific SHA256-derived
seed from fixed seed42. K=All is one unique subset. Subset generation never
reads labels or scores, is frozen before inference, and is shared across models.
Rebuild frozen physiological summaries (including zero population SD for K=1),
raw across-seizure summaries and patient reference statistics within each subset.
Keep original FIT preprocessing and validation thresholds fixed. Cache only
frozen per-seizure sufficient representations; prove their All-subset parity
against direct original inference before accepting cached subset inference.
No raw copies and no additional model training. Results average metrics across
subsets within each patient before bootstrap or cohort summaries. D endpoints:
AUROC, AP, Macro-F1, EZ-F1, Recall@10%, Recall@20%, MRR and Hit@1.

Statistical resampling: 10,000 paired patient bootstrap draws, seed42, within
sources retaining each source's patient count. D uses the same procedure with
its fixed subset counts. Nominal 95% percentile CIs for all prespecified
descriptive comparisons. Separate Bonferroni families, fixed before new results:

- Retain the existing SIX FullPR-minus-ScoreZ Macro-F1/AUROC comparisons.
- B: TWELVE FullPR-minus-NoPR comparisons:3models x4 endpoints (AUROC,AP,
  Macro-F1,EZ-F1).
- C: NINE FullPR-minus-NoPR recall comparisons:3models x3 fractional budgets.
- D: THIRTY-SIX comparisons:3models x2 endpoints(Macro-F1,AUROC) x6 contrasts
  (K1-minus-All and K2-minus-All separately for NoPR/FullPR; FullPR-minus-NoPR
  separately at K1/K2). Other D endpoints remain descriptive nominal intervals.

No significant-selection, outcome-driven stopping, best-seizure or best-source
filtering. Intervals spanning zero do not show equivalence. Channel review
efficiency concerns existing EZ labels, not postoperative outcomes. D is frozen
inference sensitivity, not matched low-seizure training performance.

Undefined AUC with one class and AP/recall/MRR/hit with no positive labels are
explicitly missing with endpoint-specific coverage; never coerce to zero or
silently drop a patient. Classification uses the original zero-denominator
convention. Paired endpoint CIs use the common finite patient set and report it.

All patient/channel identities, predictions, individual endpoints and subset
ledgers stay on the trusted server. Per-patient CSVs are private deliverables
there. Only anonymous aggregates, fixed-bin distributions/densities, tables and
figures are released. Independent endpoint formulas, direct-index bootstrap,
source reconstruction, ranking invariance, original baselines and frozen-parent
hashes must pass. Source/manifest freeze precedes calculations; score sealing
precedes endpoints. No interference with the running GPU trainer.
