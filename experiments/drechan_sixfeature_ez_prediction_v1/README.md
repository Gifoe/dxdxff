# Experiment B: physiological feature predictive informativeness

Status: implementation and pre-fit verification; no new results claimed.

An independent CPU-only Logistic Regression experiment:14 fixed configurations
(eight main plus six NoPR supplements), original five patient-held-out folds,
FIT-only nested C selection, CAL-only thresholds, and100fixed within-FIT-patient
label permutations for full12D+PR. All80patients/7635channels retained. No neural
model, new extraction or modification of Experiment A/other running experiments.

See PROTOCOL.md and PROTOCOL.json for the frozen design, weight normalization,
current unit-consistent PR, exact fit budget, 28-comparison primary family and
historical-cohort interpretation. HISTORICAL_PROTOCOL_AUDIT.csv explains why
old LR models/results are references only, not reused experimental cells.

Expected formal LR fits:1410 (840inner CV,70selected final,500negative-control).
All570final models/thresholds freeze before new Outer scores; all570score cells
seal before aggregate metrics. Final delivery requires independent numerical PASS.
