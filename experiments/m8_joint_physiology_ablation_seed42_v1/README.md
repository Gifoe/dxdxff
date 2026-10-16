# DREChannNet/M8: joint physiological ablation, seed42

Prospective 2³ deletion experiment: Variance (V), Partial High-Gamma Power
80–125 Hz (H), Spectral Entropy 10–125 Hz (E). Config digits are **V,H,E**;
1 means deletion. `000` is original7; `011/101/110` retain5 groups; `111`
retains4: beta, low gamma, RMS, line length. Every measurement contributes
both seizure mean and SD. Both D1-only and Full M8 are evaluated separately.

Reuse20 verified model pairs (`000/001/010/100`, fivefolds) and their sealed
Outer scores. Train only20 new model pairs (40 models). Original sources,
data/masks, epoch/threshold selection, initialization and training are imported
unchanged. All input/model/protocol hashes and the internal-data candidate
decision must be frozen before new Outer inference. All80 head-fold cells
are sealed before reading full evaluation labels.

The validation-only stability allowances and tie rules are in
`PROTOCOL_LOCK.json`. They are prospective engineering margins, not clinical
noninferiority margins. Development is the original65 appearances/47 patients,
selection-conditioned. Outer is the same80 patients/7635 channels/256 seizures
and is **EXPLORATORY_REPEATED_TEST**, not independent confirmation. No feature,
threshold or rule may be changed from this study's Outer results.

Report main effects, context-specific and averaged pair interactions, three-way
interaction, M8-vs-D1 increments and deletion difference-in-differences. Such
effects are conditional model/procedure utility, not pure causal physiology:
retraining, LayerNorm geometry and changed frozen bases remain relevant.

Execution: original model runtime runs `run.py train`, `run.py verify`;
stable CPU runtime runs `report.py development`; original runtime runs
`run.py outer`; stable CPU runtime runs `report.py outer`, `verify_results.py`.
Public release includes only source, hashes and anonymous aggregate tables.
Raw data, IDs, private features/scores, weights, scalers and logs stay trusted.
No additional1–3D experiments are authorized by this protocol.
