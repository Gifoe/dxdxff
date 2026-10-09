# Implementation and execution audit

## Frozen evidence and order

Historical/source review and PROTOCOL_LOCK preceded new score analyses. Five sealed TabM FIT/VAL banks, original A0 checkpoints and canonical raw export were hash-checked. Prepared tensors, feature order, patient membership, imputer/scaler states, epochs and numerical thresholds matched. Fresh A0 evaluation was limited to VAL; stored frozen channel scores were then reused for all audit variants. Maximum replay probability drift is below 2e-7; every original patient metric matches within 1e-12.

All twenty existing PR-UAS teachers passed checkpoint/output hashes, once-only OOF coverage, teacher TRAIN-only preprocessing reconstruction, raw feature identity, and query exclusion from both training and checkpoint selection. One PCA4 is fitted on each outer FIT's original prepared features without labels. One P1/P2 lambda pair is chosen **separately within each corresponding outer FIT**, not by pooling other folds' FIT labels into that fold. Choices were sealed in PRE_VAL_FREEZE before new VAL probes. OOF MC-mean score semantics are explicitly disclosed.

P1 and P2 minimize mean support BCE plus lambda/2 squared parameter norm, including intercept, from zero using damped float64 Newton and gradient infinity tolerance 1e-9. No class-balanced fabrication, threshold retuning, backbone or clinical-label modifications. One-class supports use the same finite regularized solution. There were **zero solver fallbacks**.

## Tests and complete execution

23 synthetic tests passed on the server, followed by real FIT-only episode convergence/selection checks in all five folds. Integration gates cover fresh A0 checkpoint reproduction, exact identities and preprocessing, clinical export/index alignment, frozen feature-input exclusion, and all twenty teacher provenance checks. `audit/ENGINEERING_TEST_COVERAGE.json` maps the requested 22 requirements to tests/gates.

65 development cells/47 unique IDs, 2,015 support episodes (65 uncertainty B8, 1,300 random B8, 650 half-random) completed. Repetition means are computed within patient-fold cell before inference; unique-ID cluster resampling retains all repeated appearances. Each paired contrast uses 10,000 seed42 draws. AUPRC/AUROC and sensitivity/specificity have missing-value handling and denominators; no undefined AUROC is filled with 0.5. Bias-only ranking identity passed in all episodes. Cached private coefficients, labels, oracle intervals and predictions remain server-only.

All 2,015 completed episode predictions/metrics were independently deterministically replayed in isolated fold processes; each fold's maximum metric drift was **0.0**. Original checkpoint and bank hashes were unchanged. Completed per-fold integrity audits contain only non-identifying counts. Permutations preserve support class counts: 373/2,015 one-class episodes are explicitly degenerate; random mixed-class permutations can also coincidentally preserve the complete assignment, and those counts are reported without pretending the labels changed.

## Engineering deviations (no outcome tuning)

1. Initial synthetic testing crashed natively in Python/sklearn's repeated parameter-validation/inspect path before new data analysis. AP/ROC were implemented with their exact tied-score cumulative definitions, exhaustive threshold confusion counts were vectorized, and sklearn parity tests passed. This preserved metric definitions and reduced redundant work. Inactive original source snapshots are retained to resolve execution bindings.
2. The first complete analytical pass stopped at reporting because a DataFrame `query` column was accessed as an attribute, colliding with the method. It was changed to bracket column access. All model/protocol inputs stayed fixed; no scientific outcome had been used to change a choice. The completed corrected pass retains atomic private episode ledgers.
3. A reporting-only supplement added CIs for all oracle controls/centers, score-transport summaries, scarcity summaries and numeric-threshold multiplicity semantics; it did not make predictions or select parameters.
4. Additional all-fold verification attempts crashed natively in python312.dll. Five isolated fold checks then completed exact deterministic replay. This was environment/process isolation, not a model, score, threshold, label, feature or statistical change. One stalled local aggregate download was retried; only its empty incomplete destination was discarded.

The successful analytical run binding is in RUN_STATUS/PRE_VAL_FREEZE. Later reporting/verification utilities carry separate source hashes. No background training job, automation, external patient-data upload or next model was started.

## Requirement mapping

Requirements 1–2: five real frozen checkpoint/input replay gates; 3–6: orientation, threshold, tie/extreme and no-worse oracle tests; 7–8/10–11: label-free support API, changed-query-label coefficient identity and identical query partitions; 9: PCA invariance to changes outside FIT; 12: P1 exact ranking metric identity; 13–16: finite Newton gradients, strong L2, constant features and one-class support tests; 17: class-count-preserving permutation tests and observed degeneracy counts; 18: synthetic deterministic repetition/atomic content parity plus 2,015 exact real replays; 19: VAL-only forward indexing and zero outer predictions; 20: inspected actual source chain plus 7,635 export/index/run-union matches; 21: 88D feature/input audit excluding clinical/center/outcome metadata; 22: repeated-ID bootstrap test retaining both appearances of a sampled cluster.
