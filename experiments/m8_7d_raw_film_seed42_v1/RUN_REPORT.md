# Actual Raw FiLM placement study: negative Macro gate

All10new fits/116epochs and full independent verification completed. Both
new architectures fail the prospectively fixed Development Macro gate;
selected C2 reference,**zero new Outer passes**. No>0.64full80 target is
established. The actually verified old C2 full80 Macro remains
0.6396579900303192;the internal values below are not substitute test results.

## Mechanism and fairness

All original7physiological measurements/all14processed coordinates,cohort,
patient-held-out roles,masks and complete Raw250 aggregation were preserved.
F0/F1 add the same7→32→24 zero-output conditioner to the first Raw
convolution after versus before GELU. Both retain the original lateGate,
reference,phase/seizure pooling,patient Z and one C2 scalar readout.
They reuse the oldGate's exact seven window inputs and FIT-only scale;
there is no new clinical measurement,trained D1 or second classifier.

Each new architecture has11426parameters versus C2's10378. Original shared
CNN/Gate/head tensors match exactly at initialization. New models start
from scratch,not old trained weights. F0/F1 are capacity-matched placement
controls;either versus C2 includes extra capacity/conditioning and is not
a pure causal test of physiological-feature necessity. Generic FiLM is not
claimed novel. Objective,seed42,patient order,optimizer,30/6/6budget and
201threshold protocol match original C2. Five actual old C2 fits are reused
as references withzero new epochs,not mislabeled new training.

## Actual legal Development results

65patient appearances/47unique people,not full80Outer. Development selects
epoch,threshold and architecture,so results and CIs are selection-conditioned.

| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ frac.bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 reference | .673076 | .762169 | .561626 | .484762 | .861390 | .794168 | .790300 | .707692 | -.016861 |
| F0 after GELU | .668477 | .775724 | .578870 | .474351 | .862602 | .796160 | .779723 | .707692 | -.027211 |
| F1 before GELU | .670502 | .777180 | .569715 | .484566 | .856438 | .789199 | .765768 | .676923 | -.005231 |

Eligibility requires Macro≥C2+.003,AUROC/AP≥C2−.01. Both fail Macro.
Higher AUROC does not replace the requested F1 endpoint. No gate was relaxed,
no favorablefold/secondary endpoint was selected through Outer,and no new
test metrics were generated.

| Macro paired contrast | Difference | Nominal95%CI |
|---|---:|---|
| F0−C2 | -.004600 | [-.023634,+.015221] |
| F1−C2 | -.002574 | [-.020289,+.014907] |
| F1−F0 | +.002026 | [-.010903,+.015237] |

All54absolute/paired intervals and all9metrics are released. Intervals use
10000seed42patient-cluster draws with repeated appearances together;they
do not correct selection,historical adaptation,multiplicity or seed effects.
Complete fold/source/coverage tables are included,with source metrics
suppressed for<5unique people rather than excluding those patients.

This study shows improved ranking point estimates without the required
patient-Macro gain. It neither establishes superiority of early versus
late placement nor rules out every Raw encoder. It is not justification
to reselect these failed candidates by AUROC or to run Outer until one wins.

## Verification and cost

Five CPU changed-path tests passed. All5actual C2 validation replays had
maximum error0 and identical201threshold decisions. Largest complete legal
FIT admission used164channels/48380validwindows,exact original initial Raw
summary/logits,padding and channel-order parity,and3discarded gradient
steps per arm. No formal epoch was discarded.

Actual new epochs:F0=55,F1=61,total116. All15real retained cells were
replayed through Raw;195patient rows/1755independent metric entries and
54intervals passed. Maximum replay and interval errors were0. Model/data/
role/source hashes and original protected scores remain unchanged.

Exact terminal:RAW_FILM_INTERNAL_PIPELINE_COMPLETE_NO_OUTER.
Source pushed before formal fits:41c478b0265e2c108da90bb1dfb695498605b2a6.
Model freeze SHA256:
83ffe0516ce3f7ea8b13448b498948f5b04b3a630ad538a0037134b1243721f7.
Candidate failed-gate SHA256:
778e4544c01bfccc55fb0a44a35a15ae235daa22f93947a6842401674657915c.
Anonymous13file delivery SHA256:
80dd90d4c974cf81f6f01ecf51652498db27d7705eada89a06c01149c680ea5d.

Only source,hashes and anonymous real aggregates/reports are published.
Raw,identities,labels,weights,scalers,individual scores/embeddings and logs
remain on the trusted server. The broader performance target is not complete.
