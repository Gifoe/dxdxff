# Channel-relational single head: actual negative gate

All10new end-to-end fits/107epochs completed and were independently verified.
Neither arm passed the unchanged internal Macro improvement gate;C2 remains
selected and no newOuter was run. The requested>0.64full80patient-Macro goal
remains unproven;the previously verified C2 point is0.6396579900303192.

## Mechanism and controls

All original7physiological measurements/14coordinates,Raw/Gate/masks,
completepatientZ,originalfivefold roles and exactlyone scalar classifier
were retained. P0 has a pointwise64→24GELU→64 hidden perturbation;P1 uses
query-dependent non-selfchannel contrasts with8Dquery/key and16Dvalues.
Each adds exactly3072parameters,total13450 versus C2's10378. Initial shared
CNN/Gate/head tensors and fullpatient logits were exact. New models trained
from scratch,not old trained D1/C2. Same seed42,original objective,optimizer,
30/6/6budget,patient order,valid windows and201threshold rules.

The relation is learned functional similarity,not anatomical distance,
electrode connectivity or a new clinical measurement. The contrast test
isolates cross-channel information versus a parameter-matched pointwise
alternative,not identicalexpressiveness/FLOPs orclinicalcausality.

## Actual Development,not full80Outer

65patient-fold appearances/47people,same original Development selects
epoch,threshold and architecture;selection-conditioned,not unbiasednestedCV.

| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZfracbias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 reference | .673076 | .762169 | .561626 | .484762 | .861390 | .794168 | .790300 | .707692 | -.016861 |
| P0 pointwise | .672011 | .761336 | .561546 | .485327 | .858694 | .791040 | .790584 | .707692 | -.010820 |
| P1 relational | .674225 | .765829 | .562818 | .493711 | .854739 | .787047 | .789028 | .692308 | +.004821 |

Both new arms fail Macro≥C2+.003. P1−C2 is+.0011488561257179248,
not enough. AUROC/AP/EZ-F1 gains do not substitute for the primary Macro.
No favorablefold,secondary metric or Outer outcome changed this decision.

| Macro paired contrast | Difference | Nominal95%CI |
|---|---:|---|
| P0−C2 | -.001066 | [-.005517,+.003166] |
| P1−C2 | +.001149 | [-.005764,+.009417] |
| P1−P0 | +.002214 | [-.004678,+.010461] |

All54absolute/paired interval rows for9metrics,fold/source/coverage and
actual epochs are published. Intervals use10000seed42patient-cluster draws
retaining repeat appearances,not multiplicity/seed/adaptation correction.
Sources<5unique patients have metrics suppressed,not patients excluded.

The relational arm shifts toward more EZ predictions:EZ-F1 rises while
NEZ-F1,ACC andTop-1 point estimates fall. Its small Macro increment and
relational-versus-pointwise interval do not establish a reliably better
architecture. This does not prove all channel-set learning is ineffective.

## Actual checks and delivery

Five changed-path CPU checks passed. All5old C2 actualRaw validation replays
were exact with201threshold decisions identical. Largest complete legal
FIT164channels/48380validwindows:all3initial Raw summaries/logits exact,
paddingignored,channelorderequivariant,and3discarded fullgradientsteps
including opening Q/K/V and output gradients. Formalepochdiscarded0.

Actual new epochs:P0=53,P1=54,total107. Five oldC2references count aszero
new epochs,not five new fits. All15actualretainedcells replayed;195patient
rows/1755independent metric entries and54intervals verified. Maximum logit
replay error0,maximum interval difference0. Original sources/data/roles/
checkpointandscoreseals remain unchanged.

Exactterminal:CHANNEL_RELATION_INTERNAL_PIPELINE_COMPLETE_NO_OUTER.
Sourcepushedbeforefits:5e9bd17ea24631d8f8aff75ea46fc0d37a619b2a.
Model freeze SHA256:
21c4a00f797fba9a881d8e24175f0babc86332c6a439a797caf1ca0c11400fc9.
Candidate failed-gate SHA256:
fcfc6537d69b3b70bf984fcf80b6bea5f5507ecd980f3140665285542c5e9b5f.
Anonymous13file deliverySHA256:
25cb0c1636e555ea2c76e8791d4d76229140e5db72f25266ec7f3a6a7c2e7d97.

Only source/hashes/anonymousrealaggregates are published. Raw,IDs,labels,
weights/scalers,individualscores/embeddings,caches/logs remain trusted.
This completed negative experiment does not complete the broader goal.
