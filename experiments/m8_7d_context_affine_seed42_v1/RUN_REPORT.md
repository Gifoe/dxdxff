# Actual context-affine study: insufficient internal improvement

All10new head fits/275epochs completed withzero encoder/readout updates.
All15logical cells were independently replayed from actual Raw through the
assembled single-head models. The predeclared G1 gate failed;no newOuter.
The broader>0.64full80patient-Macro goal is **not achieved**. The current
actually verified C2 full80 result remains0.6396579900303192.

## Architecture and exact input scope

Ownfold trained C2 is frozen:Raw CNN,old7DGate,110→64GELU representation and
one64→1scalar readout. All original7physiological measurements and14processed
coordinates,Raw250 masks/times/500samplewindows,reference,phase/seizure
pooling and complete-patientZ remain unchanged.

G0 learns a global64gain/64shift vector. G1 uses mean/populationSD of the
64learned hidden coordinates over all channels of one patient,then128→8→128
to supply bounded gains/shifts. The same original readout classifies the
adapted hidden values. Only one scalar classification Linear exists. There
is no new clinical input,source/ID/label/prevalence input,trained D1,
independent output residual or ensemble. Unlike a constant decision bias,
the gains can alter within-patient ranking.

Full G0/G1 parameter counts10506/12562;trainable128/2184. They are not
capacity-matched. Extra head training versus original C2 is also a budget
difference;this is not a pure architecture-causality or novelty claim.
Global shift64 has only one effective readout-bias degree of freedom.

Actual ownFIT|Development hidden extraction covered320completepatient
appearances across5folds,no ownOuter. All caches were bound to exact body/
input/preprocessor/scale/archive hashes and sealed. They contained no
new label arrays. Body had already been selected on originalDevelopment;
that label reuse and mixed legacy files are disclosed,not called physical
isolation or independent nested validation.

Both new arms used the same original objective/optimizer/order/seed42 and
201threshold rules,with100max/15min/15patience cheap head epochs. Epoch0
ineligible. G0/G1actual stopping differs by legal Development selection.
All5old C2 references are reused withzero new epochs.

## Actual Development,not80patientOuter

65patient-fold appearances/47people. These same Development labels select
epoch,threshold and candidate,so estimates and intervals are selection-conditioned.

| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ frac.bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 reference | .673076 | .762169 | .561626 | .484762 | .861390 | .794168 | .790300 | .707692 | -.016861 |
| G0 global affine | .673541 | .762474 | .562000 | .487082 | .860000 | .793093 | .791070 | .707692 | -.012914 |
| G1 patient context | .674710 | .762468 | .562437 | .488164 | .861256 | .794735 | .791085 | .707692 | -.015242 |

G1 eligibility was fixed at Macro≥max(C2,G0)+.003 withAUROC/AP≥theirmax−.01.
G1−C2=.0016336990488679008;G1−G0=.0011688866511914988.
Neither reaches the required gain. No rounding,CI upper bound or favorable
secondary metric was used to relax it. C2 remains selected,no final-stage
checkpoint freeze or test inference was launched.

| Patient-Macro paired contrast | Difference | Nominal95%CI |
|---|---:|---|
| G0−C2 | +.000465 | [-.001424,+.002548] |
| G1−C2 | +.001634 | [-.000815,+.004211] |
| G1−G0 | +.001169 | [-.002250,+.004102] |

Fold sampleSD ofMacro:C2.019543,G0.019243,G1.020072. Context does not
demonstrate improved fold stability. All54absolute/paired intervals for
all9metrics and all fold/source/coverage tables are released. Nominal
95%percentile intervals use10000seed42patient-cluster draws retaining
repeat appearances;no correction for selection,seed,multiplicity or
historical adaptation. Source metrics<5unique patients are suppressed
without excluding patients.

These results provide a small uncertain F1 point increment,not evidence
that this context network is a reliably better architecture or that it
will exceed0.64 onOuter. They do not prove every patient-context approach
is ineffective. Do not label the new internal.674710 as a full80result.

## Actual verification and execution

Five distinct CPU checks passed. Every old C2 actualRaw validation replay
matched,with identical201decisions;all10initialheads matched old outputs.
LargestcompletelegalFIT164channels/48380validwindows:both Fusedmodels had
exact initial fullRaw/cache hidden/logits and channel-order equivariance;
2discarded head-only gradient steps each,body/readout frozen.

Actual head epochs:G0=133,G1=142,total275. All15retained cells were replayed
through realRaw;all320patient-cache hashes and both frozen readout/body
state checks passed.195patient rows/1755independent sklearn/ranking
entries and54independently rebuilt intervals passed. Maximum full-model
logit replay difference4.76837158203125e-7,within declared1e-6;all201
threshold decisions exact. Maximum interval reconstruction difference0.

Exact terminal:CONTEXT_AFFINE_INTERNAL_PIPELINE_COMPLETE_NO_OUTER.
Source was pushed before any fits:
86a47602de9ab7c23e7e0111b5a9b7a28be2549f.
Model freeze SHA256:
f32af4b55d91386abdec21d5c478601a45e281299bbbdb09d64e54da77d066f1.
Candidate failed-gate SHA256:
b79f87abff78ac22638bb1a2766e18d5b1d609b6175be8dabd63b5ceffcd1740.
Anonymous15file delivery SHA256:
4174c79cbfc64360dfe915a0284c0f32a7506a333bd9063caf1333469bc146ce.

All old/paused sources,seals,checkpoints and logs are retained. Only code,
hashes and anonymous real aggregates/reports are uploaded. Raw,identities,
labels,weights,scalers,individual hidden/prediction caches and logs remain
on the trusted server. A verified internal terminal is not goal completion.
