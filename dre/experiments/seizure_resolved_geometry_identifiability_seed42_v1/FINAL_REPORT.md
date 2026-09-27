# Seizure-resolved geometry identifiability — development-only result

A1 replayed at all 150 checkpoints within 1e-6. The exact temporal-pooled seizure-channel tensor is the original aggregator input; canonical channels and masks align across seizures. No outer predictions/metrics were computed. The legacy cache initializer nonetheless materializes all 80 labels, so this is not strictly sealed.

There are 65 VLOO target cells but 47 unique patients. All target-metric intervals use 10,000 patient-ID cluster resamples.

## Geometry and matched contexts

Oracle stability mean/median 0.860/0.872. FIT r80 mean/median 18.9/19; r90 mean/median 25.2/25.
R80 reconstruction median cosine 0.621; fraction >=0.70 0.185; AP headroom retention 0.540. Full and r80 oracle scores use target labels and are NONDEPLOYABLE.
D3 r80-coefficient ridge, identical <=16D context preprocessing and predictor:

| Context | Cosine | EZ-AP | Delta AP vs static | Correct-wrong AP | True-shuffled cosine |
|---|---:|---:|---:|---:|---:|
| STATIC_R4_MARG | 0.169 | 0.5335 | +0.0000 | +0.0223 | N/A |
| SR_DIST | 0.108 | 0.4957 | -0.0378 | +0.0001 | -0.015 |
| SR_PERSIST | 0.167 | 0.5264 | -0.0071 | +0.0197 | N/A |
| SR_COMBINED | 0.129 | 0.4936 | -0.0399 | +0.0035 | -0.007 |

Shared FIT linear AP 0.5225; frozen A1 AP 0.5572. Best predeclared seizure D3 by AP: SR_PERSIST AP 0.5264, cosine 0.167.
FIT context-distance/oracle-direction-cosine correlations and FIT-ID bootstrap intervals are in CONTEXT_DIRECTION_ASSOCIATION.csv. Descriptive seizure-count strata, wrong-context and shuffled-membership controls are separate outputs.

## Required audit answers

1. Exact A1: yes, 150/150 checkpoint grids replayed, maximum error 0; five-fold mean Macro-F1 0.6260.
2. Captured tensor: source `seizure_channel_embedding`, shape `[batch, seizure, canonical channel, 32]`; original aggregator prehook input and replay agree exactly.
3. Channel identity: source canonical-channel name mapping and validity masks align; absent entries are excluded and pair statistics use observed channel intersections.
4. Seizure-count groups (distinct patients): one=7, two=11, three-or-more=29. These are 47 distinct development validation patients, not 65 independent patients.
5. FIT oracle variance requires median r80=19 and r90=25; the corresponding means are 18.9/25.2.
6. r80 is better than the prior rank-4 diagnostic (median reconstruction cosine 0.621 vs about 0.382; headroom retention 54.0% vs about 21.4%) but still fails all prespecified expressivity thresholds.
7. Matched static R4 marginal control remains weak: D3 cosine 0.169, AP 0.5335; the earlier different predictor's R4 marginal diagnostic was about cosine 0.197/AP 0.531, so these are directionally similar, not an exact numerical reproduction.
8. SR_DIST does not improve over static: cosine delta -0.061, AP delta -0.0378.
9. SR_PERSIST does not improve over static: cosine delta -0.002, AP delta -0.0071.
10. SR_COMBINED does not improve over static: cosine delta -0.041, AP delta -0.0399.
11. Therefore the measured seizure distribution/persistence summaries add no convincing patient-specific geometry information beyond the matched static marginal control; higher raw context dimension alone is not evidence.
12. FIT context-distance/oracle-direction Spearman rho is static -0.050, SR_DIST -0.005, SR_PERSIST -0.051, combined -0.037; seizure contexts do not strengthen the negative association. FIT-ID bootstrap intervals are in the association CSV.
13. Correct versus cyclic wrong context for the best seizure D3 (SR_PERSIST): cosine +0.036, AP +0.0197; neither reaches both fixed identification thresholds and patient-cluster uncertainty includes zero.
14. True minus shuffled seizure-membership cosine: SR_DIST -0.015, combined -0.007; real membership does not beat the destruction control. No fabricated persistence shuffle was used.
15. Descriptively the best seizure D3 minus D1 AP is -0.0383 (one), +0.0074 (two), +0.0095 (three-or-more); these posthoc groups neither establish a seizure-structure effect nor justify group-specific selection.
16. Best seizure D3 AP 0.5264 barely exceeds D1 0.5225 but is below frozen A1 0.5572; it does not meet the deployable readout gate.
17. No: r80 expressivity, context identification, and deployable ranking gates all fail. A seizure-conditioned patient-specific readout is not justified by these representations.
18. If pursuing this question, predeclare a separate few-shot patient-calibration study or obtain genuinely richer observations; another zero-label hypernetwork on the same A1 summaries is not supported.

A patient-specific oracle can exist without an identifiable zero-label proxy. Larger context dimensionality alone is not evidence that seizure membership helps; the shuffled-structure control must also improve. No further zero-label hypernetwork is justified unless the fixed gates pass. If they fail, few-shot patient calibration or genuinely richer observations are the coherent alternatives, subject to a new protocol.

`R80_ORACLE_SUBSPACE_INSUFFICIENT`
`SEIZURE_RESOLVED_CONTEXT_DOES_NOT_IDENTIFY_PATIENT_GEOMETRY`
`SEIZURE_CONTEXT_READOUT_SIGNAL_NOT_YET_SUPPORTED`
`ORACLE_GEOMETRY_NOT_WELL_REPRESENTED_BY_SHARED_LINEAR_SUBSPACE`
