# Raw-only Macro-F1 readout experiment — actual results

## Target and result

The user's target is **patient-equal Macro-F1 >0.65**. The actual historical
Raw-only reference is **0.6304930529**, not EZ-positive F1. This experiment did
**not** reach the target. Neither readout refitting nor the predeclared internal
selection procedure improved that reference in the complete fivefold report.
All values below are from completed server runs and independent verification.

## Complete frozen fivefold evaluation

Each procedure covers the same80 patients,7,635 channels and original five
patient-held-out folds. These are patient means, not pooled channel metrics.

| Procedure | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ fraction bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original Raw-only Native | 0.630493 | 0.741296 | 0.550114 | 0.426475 | 0.834511 | 0.757014 | 0.784454 | 0.700000 | -0.041381 |
| Same readout refit | 0.627736 | 0.741656 | 0.549391 | 0.419630 | 0.835843 | 0.758238 | 0.782641 | 0.687500 | -0.048515 |
| Own-FIT-selected procedure | 0.624314 | 0.734363 | 0.543291 | 0.415227 | 0.833401 | 0.754767 | 0.777433 | 0.687500 | -0.045300 |

Paired10,000-patient-bootstrap Macro-F1 differences:

| Comparison | Difference | 95% interval |
| --- | ---: | --- |
| Refit − Native | -0.002757 | [-0.008149,0.002332] |
| Selected − Native | -0.006179 | [-0.012020,-0.001621] |
| Selected − Refit | -0.003422 | [-0.011409,0.004112] |

The selection rule chose Native in folds1,2,3,5 and NO_SD in fold4. This is a
fixed heterogeneous procedure, not the best arm chosen after seeing Outer.
Refit and Selected were both specified before evaluation; neither was dropped
from the report. The one-fold SD deletion does not establish a universal claim
that seizure SD is necessary or useless.

Historical Outer outcomes have been viewed repeatedly. These intervals describe
this exploratory repeated-test comparison, **not independent confirmation**.
No post-Outer model, feature, threshold or selection change was made.

## Internal matched comparisons

Stage1 used15 genuine cross-fitted encoder paths (three per ownFIT), excluding
each fold's original Development and Outer. Each arm has255 assessment patient
appearances from79 unique people. Repeated appearances are clustered together
in bootstrap draws; they are not255 independent people. These previously used
paths remain selection-conditioned.

| Arm | Patient Macro-F1 | Change relative to REFIT | 95% paired interval |
| --- | ---: | ---: | --- |
| NATIVE | 0.609084 | +0.002046 | [-0.000827,0.005089] |
| REFIT | 0.607038 | 0 | reference |
| PATIENT | 0.607982 | +0.000944 | [-0.000820,0.002753] |
| LOCAL | 0.606291 | -0.000747 | [-0.002257,0.000690] |
| NO_REFERENCE | 0.557006 | -0.050031 | [-0.070167,-0.029965] |
| NO_PHASES | 0.597416 | -0.009622 | [-0.018222,-0.000827] |
| NO_SD | 0.604623 | -0.002415 | [-0.010424,0.005268] |
| NO_Z | 0.514481 | -0.092557 | [-0.120807,-0.066008] |

PATIENT−LOCAL is +0.001691, interval[-0.000078,0.003512]. The two variants have
identical initialization and14,097 trainable parameters; this controls extra
capacity rather than comparing a larger context model to only a smaller head.
There is no convincing gain from the tested patient conditioner. Generic
DeepSets/FiLM components are not claimed novel.

The deletions reuse exactly the same frozen encoder and refit only the readout.
They support a conditional statement about these representations and this
optimization protocol, not end-to-end retraining or universal component
necessity. NO_Z also changes numerical conditioning. NO_SD tests spread beyond
the retained seizure mean. The earlier EZ-directed53-cell study remains
unchanged and is not relabeled as Macro-optimized evidence; its EZ-threshold
oracle is not a Macro-F1 ceiling.

## Implementation and isolation

- No new Raw CNN training, manual physiological features, D1 classifier, D1
  residual/logit, source ID or patient ID is a learned input. The96D summaries
  use the original0–20-second support, complete-patient Z, original valid-window
  masks and complete differentiable pooling from the frozen encoders. The
  historical160-slot readout retains64 literal zero slots.
- Stage1 performed105 small-head fits/3,105 epochs. Stage2 performed six actual
  head fits/135 epochs; Native and matching procedure aliases were reused.
- Internal TRAIN+CHOICE fits readouts; CAL alone selects epochs/thresholds;
  HELD compares candidates. Each ownFIT selects its own procedure with the
  registered source-stability penalty and complexity tie rule. No global best
  model from the overlapping assessment folds is imposed on allOuter folds.
- Final encoders are the actual historical fullFIT Raw-only checkpoints.
  Readouts use original FIT for loss and original Development for epoch and
  threshold selection. Reuse of Development for historical encoder selection
  and readout selection is not an independent validation set.
- All15 final logical models/thresholds are frozen before newOuter inference.
  All15 score cells are sealed before the fullOuter truth archive is read.
  Patient-private hash-bound caches permit exact engineering resume.
- Native validation logits replay with maximum error0 in all five folds.
  NativeOuter probability replay error is at most1.1920929e-7, with unchanged
  decisions, before reusing the exact historical sealed Native probabilities.
- Separate verification checks archive patient/channel order, all80 patients,
  all7,635 unique channels, role separation and allnine patient metrics. Its
  maximum difference from saved patient metrics is below1e-12.

## Numerical incident, preserved rather than hidden

The first admission failed: CPU FP32 versus historical CUDA FP32 head logits
differed by up to2.861023e-6 across the15 diagnosed paths. No formal fit had
started. Original frozen source files, input freeze and failed logs were kept.
The recorded pre-fit `CPU_BRIDGE_AMENDMENT.json` uses original CUDA arithmetic
for Native replay/verification (all15 paths bit-exact) and retains CPU FP32 for
new trainable heads. The CPU bridge had maximum probability error1.7881393e-7
and zero disagreements across every201-point CAL grid. This is an explicit
amendment; the failed CPU absolute-logit criterion is **not** reported as passed.

## Artifacts and interpretation

`results/cpu_bridge_execution_v2/` contains all120 logical internal cells,
aggregate metrics, paired intervals, source/fold summaries, selection and
verification. `results/final_stage_v1/` contains patient means, pooled results,
fold/source results, foldSD, metric coverage, paired/absolute intervals,
selection and independent final verification. Source outcomes with fewer than
five unique patients are suppressed. Raw waveforms, identities, embeddings,
weights, labels, individual scores and logs remain on the trusted server.

Final terminal: `COMPLETE_VERIFIED_THREE_RAW_MACRO_PROCEDURES`;
independent verification: `PASS`; target established: **false**.

The strongest of these predeclared procedures remains the original0.630493
Raw-only model, still0.019507 below0.65. This bounded readout study is complete;
the broader performance goal is not. A further improvement would require a
separately registered, internally selected representation/training experiment,
not post-hoc repairs to these frozenOuter results.
