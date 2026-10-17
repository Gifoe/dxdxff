# Original7D hidden-anchor single head: complete Development result

## Decision

**No new candidate passed the registered gate. No new Outer evaluation was
performed.** The previous C2 reference remains selected. The overall goal of
full80 patient-equal Macro-F1 strictly greater than0.64 remains unmet; the
latest frozen full80 primary is still0.6396579900303192, not a result from
this Development-only study.

Five shared physiological warm fits and15Raw joint fits actually completed:
123warm epochs+156joint epochs=279epochs. Twenty real new training cells and
five exact replayed C2 controls were independently verified. There were no
training/verification stderr messages or failed/repeated scientific jobs.

## What changed, and what did not

The original seven measurements,14DELTA seizure-mean/SD coordinates, PatientZ,
FIT-only preprocessing, Raw250/500sample windows,24D CNN, pre-reference,
0–8/8–20phase pool, seizure mean/populationSD,96Dfull-patient RawZ and original
seven-input Gate are unchanged. Original80patients/7,635channels/256seizures,
fivefold roles, sources and all valid-window masks remain bound to their
original hashes. No Variance deletion, new feature or extraction was used.

The new single shared head computes:

`h = Linear14→96(original_duplicate_aware_normalization(f14))`

`h = h + tanh(Linear96→96_without_bias(raw96))`

`NEZ_logit = Linear96→1(Dropout0.15(GELU(h)))`.

The Raw projection starts zero. Physiological normalization uses the exact
original virtual88D LayerNorm algebra:14unique values with22multiplicity slots
and66zero slots, epsilon1e−5. This is a classifier readout-geometry change,
not additional physiological measurements or naive LayerNorm14. There is one
scalar classification readout and no D1 module/output, additive prediction
residual or ensemble. The untrained original physiological initializer and
Raw CNN/Gate are copied only to match initial weights; no trained D1/C2 state
initializes a new model.

- H0: jointly train this architecture from scratch.
- P: train the **same final head** on physiology only, with Raw projection/CNN/
  Gate fixed at their initial states. One warm model per fold is shared.
- H1: load that own-fold warm state, reset optimizer/seed and jointly train
  all parameters with Raw.
- H2: load the same warm state, fix its physiological projection/readout and
  train the Raw projection/CNN/Gate through the shared nonlinearity.

Every new full architecture has13,961parameters; H0/H1 train13,961 in the
joint phase, H2 trains12,424, and P trains1,537 during warming. P's unused Raw
parameters are not falsely described as active physiological-model capacity.
All three new architectures have identical initial tensors before warming.

Relative to the10,378parameter C2 reference, width, normalization, dropout,
initialization and head geometry differ. Warm variants also receive extra
cheap feature training. Therefore comparisons to C2 are not capacity-matched,
and warming versus scratch is a training-path-plus-budget comparison, not
pure architecture causality. No novelty or guaranteed improvement is claimed.

All phases retain original0.75weighted NEZ BCE+0.25soft patient Macro-F1,
patient-equal complete updates, AdamW(.0003,wd.001), clip1 and FP32/TF32-off.
The new joint phases retain max30/minstop6/patience6. Warm training uses a
predeclared max100/min15/patience15 because it is inexpensive; actual warm
epochs are disclosed separately. Epoch0 is ineligible. The original201NEZ
threshold grid and epoch/threshold tie rules remain unchanged.

## Legal Development results, not full80 test results

These are65patient appearances of47unique people from the original five
Development roles. The same Development selects warm/joint epoch, threshold
and architecture. These estimates and intervals are selection-conditioned;
they are not an unbiased nested-CV estimate or independent confirmation.

| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ frac. bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 reference | .673076 | .762169 | .561626 | .484762 | .861390 | .794168 | .790300 | .707692 | −.016861 |
| P physiology-only warm, diagnostic | .646064 | .722573 | .516849 | .452743 | .839385 | .769856 | .694708 | .569231 | +.004933 |
| H0 scratch | .670936 | .780241 | .575981 | .486516 | .855355 | .789677 | .753241 | .661538 | −.001033 |
| H1 warm then joint | .673146 | .770633 | .572529 | .492554 | .853737 | .788298 | .753901 | .646154 | +.001793 |
| H2 warm then fixed head | .674237 | .774072 | .579597 | .490208 | .858265 | .791858 | .744701 | .646154 | −.009909 |

P is diagnostic only, not eligible as the Raw+7D final candidate.

The unchanged eligibility rule requires candidate Macro≥C2+0.003 and
AUROC/AP≥C2−0.01. H0 gains−0.002140, H1+0.000069, H2+0.001160 Macro.
All fail the Macro gate. The highest new penalized score cannot override
eligibility. In particular, improved AUROC/AP cannot substitute for the
requested F1 target. The gate was not relaxed because C2's old Outer was
close to0.64, and no H0/H1/H2 Outer result has been generated.

## All ten paired Macro contrasts

Nominal95% percentile intervals use10,000seed42 patient-cluster draws,
retaining repeat appearances together. They are not multiplicity-adjusted
and do not account for historical adaptation or seed uncertainty.

| Contrast | Difference |95% CI |
|---|---:|---|
| P−C2 | −.027012 | [−.062018,+.007886] |
| H0−C2 | −.002140 | [−.019842,+.017840] |
| H1−C2 | +.000069 | [−.024287,+.023358] |
| H2−C2 | +.001160 | [−.021887,+.024083] |
| H0−P | +.024872 | [−.007600,+.057991] |
| H1−P | +.027081 | [+.006446,+.049957] |
| H2−P | +.028172 | [+.006939,+.051560] |
| H1−H0 | +.002210 | [−.016686,+.020117] |
| H2−H0 | +.003301 | [−.016278,+.021923] |
| H2−H1 | +.001091 | [−.006051,+.008083] |

All five absolute intervals and all ten paired contrasts for all9endpoints
(135rows) are included, not just these Macro comparisons.

## Mechanism interpretation and limits

Raw adaptation does contribute to the warmed head: H2's physiological
projection/readout remain bit-exact, while its Raw projection and backbone
adapt. H2−P has positive nominal Macro limits on these reused Development
data. This supports a conditional representation increment under the tested
procedure, not independent physiological causality or universal necessity.

Warming is not enough to improve the current target reliably. H1/H2 versus
H0 and C2 Macro intervals span zero. H0 improves AUROC byabout.0181 but loses
Macro; H2 improves AUROC/AP while losing NEZ-F1, ACC, MRR andTop-1 point
estimates versus C2. Better ranking is not equivalent to better F1 at a
validation-selected decision threshold. A larger hidden layer or simply
preserving a physiological anchor did not solve that decision problem here.

The results do not justify a new expensive Outer pass or a claim that the
independent D1 branch is necessary for every possible single-head architecture.
They reject this finite set as sufficiently improved candidates under the
predeclared rule. No post-hoc patient/center exclusion or threshold adjustment
is allowed to rescue them.

## Stability and actual cost

| Fold | C2 Macro | P Macro | H0 Macro | H1 Macro | H2 Macro |
|---|---:|---:|---:|---:|---:|
|1|.666792|.642661|.657501|.667319|.662482|
|2|.705112|.679452|.688206|.688035|.695060|
|3|.662655|.649135|.701293|.690374|.682515|
|4|.676270|.675128|.674635|.687562|.686561|
|5|.654551|.583944|.633043|.632437|.644565|

H2 improves Macro in only2/5folds. Its fold sampleSD is.020449 versusC2's
.019543. The pediatric source group's12Development appearances have Macro
.573225 versusC2's.604001, while HUP improves from.701646 to.717329 across
28appearances. Source groups are inherited provenance labels, not independent
equally sized hospitals; appearances are not unique patient counts. No group
was dropped or assigned a special threshold. Full source/coverage tables
retain the exact denominators and suppression rule.

Actual epochs: P123, H053, H147, H256. C2's five old fits were reused and
counted as zero **new** epochs; their actual prior training epochs and selected
epochs remain in the cell table. Warm models are not counted twice for H1/H2.

## Independent verification and delivery

All25actual checkpoint/control cells were replayed with maximum logit error
zero and identical201threshold decisions. All325patient rows/2,925metric
entries, all135intervals, stage transfer hashes, unchanged warm Raw tensors,
fixed H2 physiological/readout tensors, random initial states and actual
fit/epoch counts passed. Maximum interval difference was zero. All frozen
source files remain byte-unchanged from the pre-training source commit.

Exact terminal: `HIDDEN_ANCHOR_INTERNAL_PIPELINE_COMPLETE_NO_OUTER`.

Model freeze SHA256:
`354ffa9832b21f7e74983a00daa8b92871197cdfd9fcd51ea1bfd7fb409230a3`.
Candidate/failed-eligibility selection SHA256:
`5463d6ab08ecafaf75910859ed629c4ff69dd1d2efee113c3f669b87040a19f8`.
Anonymous13file delivery ZIP SHA256:
`db4878726aaae70b57e1619fc0f66618150cc5915b0ed6e105621013d9e24162`.

Only source, hashes and actual anonymous aggregates are published. Raw data,
labels/identities, weights, private predictions/embeddings, scalers and runtime
logs remain on the trusted server. This experiment is complete; the overall
>0.64goal remains active and requires a distinct prospective next study, not
alteration of this failed gate or substitution of its Development values.
