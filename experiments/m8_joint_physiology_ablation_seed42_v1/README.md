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

## Completed evidence

See [REPORT.md](REPORT.md) for all16 D1/M8 rows, uncertainty, interactions and
selection boundaries. [Outer results](results/outer/MAIN_RESULTS.csv) contain
the nine reported endpoints, including EZ-fraction bias;
[paired contrasts](results/outer/PAIRED_CONTRASTS.csv) contain deletion, factorial
and Raw-compensation contrasts. Fold/source/stability/coverage and absolute
intervals are separate tables in the same folder. Development tables and the
candidate gate are under `results/development/`.

40 new models/434 epochs completed;40 historical models and their existing
Outer/base scores were reused. All80 validation models replay with logit error0.
No candidate passed the prospectively locked internal stability gate: both
head decisions remain000. The exploratory4D M8 has AUROC0.767551 and
Macro-F10.640621, but is not promoted from Outer. The strongest conditional
finding is a nonadditive M8 three-way AUROC interaction, not a minimal-subset
proof. There was no1–3D follow-up training or post-Outer threshold change.

Independent checks cover1280 Outer patient rows,1040 validation appearances,
all10,000 draws of1116 paired/84 adjusted/288 absolute intervals, and5508
grouped aggregate numeric fields. Additional read-only audits were executed
after the immutable production driver: `tools/verify_all_intervals.py` and
`tools/verify_grouped_aggregates.py`. Local release verification is
`python tools/verify_public_release.py`; local synthetic tests are
`python -m pytest test_analysis.py -q`.

Terminal:
`JOINT_2CUBED_COMPLETE_NO_INTERNAL_STABLE_4D_OR_5D_REPLACEMENT_EXPLORATORY_OUTER_ONLY`.
