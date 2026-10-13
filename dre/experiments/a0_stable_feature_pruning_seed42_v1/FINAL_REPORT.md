# Stable feature pruning: exact compression succeeds; predictive pruning fails the registered gate

| Arm | Physiological inputs | Parameters | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR | Top1-EZ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 original frozen F0 | 42 unique, original88 interface | 8817 | 0.716569 | 0.508559 | 0.642067 | 0.433516 | 0.676591 | 0.553846 |
| B1 folded complete | 42 | 4225 | 0.715611 | 0.514250 | 0.642360 | 0.443450 | 0.689708 | 0.584615 |
| B2 predefined spectral-DELTA deletion | 36 | 3649 | 0.719570 | 0.517331 | 0.642722 | 0.437631 | 0.691219 | 0.584615 |
| B3 FIT-selected deletion | 40 in every fold | 4033 | 0.713472 | 0.515217 | 0.638483 | 0.432658 | 0.694687 | 0.584615 |
| B4 matched random deletion | 40 in every fold | 4033 | 0.717236 | 0.518808 | 0.644092 | 0.445378 | 0.700064 | 0.600000 |

Actual completed Development: equal means over65 original validation appearances,
47 unique patients, original five folds, seed42. B0 is hash-verified historical
prediction reuse. B1–B4 are20 new scratch-trained cells. All330 nested FIT-only
screening cells ran; no missing values or placeholder models. **No new Outer
TEST was run.** Terminal:
`STRUCTURAL_COMPRESSION_VERIFIED_PREDICTIVE_PRUNING_NOT_ESTABLISHED`.

## 1. What is algebraically redundant?

The frozen manifest and all five legal banks verify42 unique nonconstant
coordinates,8 exact auxiliary DELTA copies,12 retained constant-zero positions,
26 masked ZDELTA positions. Copies change the original LayerNorm geometry
despite containing no independent measurements. They are therefore folded,
not naively discarded. The resulting42-input model physically has4225 trainable
parameters, a52.08% reduction from8817, and reconstructs original88-position
normalization, including zero-position affine contributions.

Frozen checkpoint parity on all actual FIT/VAL tensors: max float64 logit
discrepancy1.3323e-15; max float32 discrepancy8.3447e-7. All threshold decisions
and original metric replay are identical. Synthetic pruned/full models with
nondefault LN affine also pass. This is a verified **functional compression**.
It is not evidence that low-frequency physiology or copied features are
clinically irrelevant. Their information content and their normalization
weighting are separate questions.

## 2. What is numerically unreliable?

The six predefined candidates are mean/std of beta, low-gamma and high-gamma
**DELTA view** coordinates, not their ABS/RATIO counterparts. FIT-only measured
raw within-patient SD and incidence of the original Patient-Z SD≤1e-8 fallback
are in `NUMERICAL_RELIABILITY_AUDIT.csv`. These are real current measurements;
COVE's earlier validation-scale warning motivated the candidate definition,
but did not itself prove predictive uselessness.

`FIT_CORRELATION_AUDIT.csv` records all861 canonical-coordinate pairs per fold
using FIT patient-equal covariance. RMS/variance/line-length/entropy correlations
are descriptive, never an automatic deletion rule. Small coefficients,
correlation, small SD and near-zero corrections do not establish clinical
irrelevance. No raw feature or window re-extraction was performed.

## 3. Does conditional held-out utility support a stable deletion?

Three screening groups per original FIT cohort, with separate training and
checkpoint-selection patients inside each remaining set. Original88D imputer
and scaler refit only on inner training patients. Screening patients do not
choose epochs or thresholds; original Development patients do not choose
groups. Eligibility requires AUROC gain≥.002, AP nondecrease, Macro-F1 loss≤.002
and positive AUROC in at least2/3 screens. The maximum is3 groups, not a forced
deletion budget.

Exactly **one** group qualifies in every original fold:

| Original fold | Selected deletion, mean+std | Inner mean ΔAUROC | Matched random deletion, mean+std | Final B3−B1 AUROC |
|---|---|---:|---|---:|
| 1 | Variance ABS | +0.003441 | Entropy ABS | -0.003298 |
| 2 | Line length RATIO | +0.002461 | Low-gamma DELTA | +0.005209 |
| 3 | RMS ABS | +0.008933 | RMS DELTA | -0.013973 |
| 4 | High-gamma DELTA | +0.010952 | RMS RATIO | -0.000311 |
| 5 | High-gamma DELTA | +0.006900 | Line length DELTA | +0.001682 |

High-gamma DELTA is selected2/5 times; variance ABS, line-length RATIO and RMS
ABS1/5 each. No single group is selected in most folds. Beta/low-gamma DELTA
groups are not independently eligible in any fold. Do **not** infer that their
ABS/RATIO counterparts should be removed, or that a globally combined subset
is validated. The selector was frozen before final Development, with no
retrospective cross-fold union/intersection or rescue search.

All B3/B4 models have40 coordinates/4033 parameters:192 fewer trainables than
B1, a4.54% reduction. B4 uses the same number of two-coordinate groups by fixed
RNG without rerolling; group multiplicities need not match. This run has no
zero-deletion aliases, but the code handles them honestly.

## 4. Optimization change versus true information removal

At the same frozen weights, full42 folding preserves logits. Retraining the
folded parameterization does **not** preserve AdamW geometry, clipping or
weight-decay effects, nor the space of separately trainable LN affine weights.
B1−B0 AUROC is−0.000959; AP+0.005691; Macro-F1+0.000293. These changes are
optimization/checkpoint-selection effects, not information added or removed by
structural folding. No trained F0 checkpoint initializes any new competitive
model; original seeded untrained weights are folded for every cell.

## 5. Paired uncertainty and final gates

10,000 fixed-score paired patient-ID bootstrap draws, seed42;47 IDs resampled
with all their repeated appearances. Thresholds stay fixed. Unadjusted95%
percentile intervals are conditional on the frozen models, not a correction
for many historical searches, seeds, splits or centers.

| Comparison | ΔAUROC [95% interval] | ΔAUPRC [95% interval] | ΔMacro-F1 [95% interval] | ΔEZ-F1 [95% interval] |
|---|---|---|---|---|
| B1−B0 | -0.000959 [-0.014343,0.015257] | +0.005691 [-0.005987,0.020158] | +0.000293 [-0.012456,0.012976] | +0.009934 [-0.010463,0.030918] |
| B2−B1 | +0.003959 [0.000190,0.008407] | +0.003081 [-0.002150,0.008312] | +0.000362 [-0.009029,0.009845] | -0.005818 [-0.019538,0.008119] |
| B3−B1 | -0.002138 [-0.007929,0.002939] | +0.000967 [-0.005411,0.007289] | -0.003877 [-0.009873,0.002221] | -0.010791 [-0.019588,-0.002708] |
| B3−B4 | -0.003764 [-0.009892,0.001689] | -0.003591 [-0.013263,0.005730] | -0.005609 [-0.016227,0.002877] | -0.012720 [-0.030778,0.001437] |

B2 improves AUROC in5/5 folds and its unadjusted interval is positive. It is a
specific **small ranking gain** on this reused Development cohort, not a general
clinical claim. EZ-F1 decreases0.005818, breaching the prelocked.002 preservation
allowance; its interval spans zero but the registered point-estimate stability
rule still fails. The+0.010 research target is not reached. B2's combined deletion
does not prove each of the three groups individually useless. It also has no
three-group-count-matched random control: B4 matches B3's one group, not B2.

B3 improves AUROC in only2/5 folds, loses Macro-F1/EZ-F1 beyond registered
allowances, and does not beat B4. Its EZ-F1 loss versus B1 has a negative
unadjusted interval. FIT eligibility did not transport stably to Development.
The data do not justify promoting automatic selected pruning as a localization
improvement. No arm meets the full pruning-usefulness gate.

## Other metrics and weighting

| Arm | Sensitivity | Specificity | NEZ-F1 | Balanced accuracy | Accuracy |
|---|---:|---:|---:|---:|---:|
| B0 | 0.464334 | 0.875001 | 0.850618 | 0.669667 | 0.779216 |
| B1 | 0.499263 | 0.852371 | 0.841270 | 0.675817 | 0.769263 |
| B2 | 0.465446 | 0.869086 | 0.847813 | 0.667266 | 0.777119 |
| B3 | 0.477473 | 0.862754 | 0.844308 | 0.670114 | 0.770316 |
| B4 | 0.493346 | 0.856246 | 0.842806 | 0.674796 | 0.771012 |

Separate47-ID-equal AUROC values, after averaging repeated appearances within
ID: B0.723153, B1.721015, B2.727195, B3.719269, B4.723327. This secondary
estimand cannot replace the registered65-appearance primary mean. Full AP/F1/
ranking and sensitivity/specificity are supplied for both estimands and every
fold. These are patient means, not pooled-channel metrics.

## Executed integrity, resources and stop

- Original80-patient/256-seizure/7635-channel contract and five-fold identities
  are admitted by original feature/order/split/legal-bank hashes. The seizure
  contract is inherited; no new raw records are counted here. TEST memberships
  are metadata exclusion checks only; no outer feature/score evaluation occurs.
- Folded checkpoint parity, actual-input removed-coordinate independence and
  original metric replay all PASS. All initializers/loss/optimizer/lr/update/
  class weighting/clip/early-stop/threshold rules are unchanged apart from the
  explicitly requested compact parameterization and predeclared nested roles.
- 330 screening cells completed3759 epochs;20 final cells completed187 epochs:
  **350 competitive cells /3946 epochs**. A Windows status-file replacement
  permission error stopped the first invocation after52 complete screening
  cells. The same command resumed from exact complete checkpoint/optimizer/RNG
  artifacts. No completed epoch was discarded, no subset rule changed. Monitoring
  switched to read-sharing with delete permission. The exact competing cause
  of the brief file lock is not independently proven.
- Resource CSV seconds are per **latest invocation**, including reload time for
  reused cells, not total wall time since first launch. The first52 cells' original
  elapsed training times were not persisted in that aggregator. The screening
  `reused_complete_cell` field is only a short-invocation heuristic, not an
  authoritative reuse counter. No precise all-in training-time claim is made.
- Independent score-only checks PASS for260 new Development patient/model
  cells,44 paired metric/interval calculations and all330 screening cells /
  5610 screening patient/model cells. All315 group/screen contrasts, nested
  preprocessing, eligibility, ranking and random-control decisions independently
  replay. The additional verifier performs no model inference/training or
  threshold optimization and leaves original source/output freezes unchanged.
- A verifier upload initially had an SSH banner failure; its premature invocation
  found no script and performed no audit. Upload retried after direct connection
  verification; the actual verifier completed with exit0. Both logs remain private.
- Only executable code, hashes and anonymous statistics are published. All IDs,
  individual predictions, banks, scalers, checkpoints and runtime logs stay on
  the trusted server. No installed skill was used. Historical scientific
  results and recovery backups are unchanged.

Retain F0 for established predictive behavior; keep full42 exact folding as an
**efficiency candidate**, not an established localization improvement or proven
noninferiority of its retrained trajectory. B2 warrants descriptive attention
for ranking but fails the complete registered stability decision. Do not create
a new combined subset, global winner, rescue search or Outer TEST from these
outcomes. Clinical irrelevance cannot be concluded from this feature-pruning
experiment. Seed42 stops here.
