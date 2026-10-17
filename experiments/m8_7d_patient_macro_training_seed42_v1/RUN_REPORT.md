# Retained C2 + patient-Macro training: actual negative gate

All10new end-to-end fits/108epochs completed. After a disclosed native
verification-only failure and exact verifier-only resume,all15actual model/
control cells,195patient rows and54intervals independently passed.
Neither objective passed the fixed internal Macro improvement gate;C2
remains selected,no newOuter. The>0.64full80goal remains unproven.
Previously verified full80C2 patient-Macro is0.6396579900303192.

## What changed

Architecture was intentionally retained,not renamed as a new structural
method. All10378parameters,original7D/14coordinates,Raw/Gate/reference/
phase/seizure aggregation/patientZ and one scalar C2 classifier are unchanged.
No newband,feature,trainedD1,logitresidual,source/ID/labelpredictor orensemble.

The original C2 reference uses .75globally classweighted mean BCE+
.25softpatientMacroLoss. T0 uses .5globalclassweighted BCE normalized by
the patient's total classweight+.5softMacroLoss. T1 substitutes the equally
weighted mean of its two within-patient classmean BCEs. Both new versions
have the same mix,normalization scale,architecture,seed42,initial tensors,
optimizer,30/6/6budget,patient order and201threshold rules.

T1−T0 contrasts local versus global classbalance in that normalized mix.
Either versus C2 also includes the lossmix/normalization change;it cannot
be attributed to classbalance alone. Labels enter only original legalFIT
training loss,not model prediction or any test-time prevalence estimate.
All legalFITpatient groups were checked to retain both original classes;
no patient was filtered. LocalBCE classreplication invariance does not
imply softMacro or fullobjective invariance.

New models trained from scratch,not old trained weights. Five actual old
C2references are reused withzero new epochs;their prior training counts
remain in DEVELOPMENT_CELLS.csv.

## Actual Development,not80patientOuter

65patient-fold appearances/47unique people. SameDevelopment selectsepoch,
threshold and objective;estimates and intervals are selection-conditioned.

| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZfracbias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 reference | .673076 | .762169 | .561626 | .484762 | .861390 | .794168 | .790300 | .707692 | -.016861 |
| T0 normalized global | .669890 | .758296 | .559045 | .482541 | .857240 | .790082 | .803812 | .738462 | -.006050 |
| T1 patient balanced | .673135 | .759028 | .555161 | .489224 | .857045 | .789980 | .776798 | .676923 | -.003071 |

Both fail the predeclared Macro≥C2+.003 gate. T1−C2 is only
0.000058430287662641484. Better point estimates than the weakerT0 control
cannot replace the stronger retained C2 comparison. No criterion was
changed or another endpoint substituted to authorize Outer.

| Patient-Macro paired contrast | Difference | Nominal95%CI |
|---|---:|---|
| T0−C2 | -.003186 | [-.008835,+.002659] |
| T1−C2 | +.000058 | [-.005634,+.005973] |
| T1−T0 | +.003244 | [-.002897,+.009401] |

All9metrics,54absolute/paired intervals and fold/source/coverage results
are released.10000seed42patient-cluster draws preserve repeated appearances,
not multiplicity/seed/adaptation correction. Metrics for sources<5unique
people are suppressed without excluding patients.

Patient-local balancing increases EZ-F1 while reducing NEZ-F1;their
near-cancellation explains the negligible Macro pointchange. AUROC/AP/MRR/
Top-1 point estimates do not show a consistent improvement over C2.
The tested objective family is not supported as a replacement. This does
not prove every loss or the unchanged physiological features ineffective.

## Actual verification and source integrity

Five unique CPU objective/model checks passed. All5old C2 actualRaw
validationreplays and201decisionchecks passed. LargestcompletelegalFIT
164channels/48380validwindows:all3initialRawsummary/logits exact,padding/
channel-order checks and3discardedfullgradientsteps each. No formalepoch
was discarded. Actualnew epochs:T0=52,T1=56,total108.

All15retainedcells independently Raw-replayed,195patientrows/1755sklearn/
ranking entries and54intervals passed with maximum logit and interval
differences0. Model/source/data/role/checkpoint seals remain unchanged.
The first verifier crashed in pandas nativebootstrap processing after
all checkpoint replays;the exact verifier-only fresh-process recovery,
Windows event and protected hashes are detailed in ENGINEERING_LEDGER.md.

Exact successful terminal:
PATIENT_MACRO_TRAINING_INTERNAL_PIPELINE_COMPLETE_NO_OUTER
(RunTag02,verification only;training completed underRunTag01).

Training source pushed beforefits:
d5dbe1bbcecdba21d2e47ce92aa52ec948edf46b.
Model freezeSHA256:
9aeb0770c177993048dff7585ca23cb22acd06147766b4797f61eb607108d361.
Candidate failed-gateSHA256:
24eb24e9dcbdb38665a69b122b979cf7fc2e2e601d0b56c35b29ad3c40810658.
Anonymous13file deliverySHA256:
503874cfb46e39569935a3551b4312ef1532c61e0b16b7d3c98bf0fa66a9dc77.

The optional separate final source recipe was prepared but not executed or
deployed:its entry gate fails. No finalfreeze,newOuter score,full80metric
or targetsuccess is manufactured from that unrun code.
Onlycode/hash/anonymousrealaggregates are published. Raw,labels/identities,
weights,scalers,individualpredictions/embeddings,caches/logs remain trusted.
This experiment is complete;the broader model performance goal is not.
