# Implementation contract

Rank-configurable SourceA0 preserves shared architecture and R4 parameter keys
`network.*`, `V`, `u`, `b`. R0 has only b; counts 8821/8921/9021/9221 are tested.
Parts computes base plus (direction+bias), preserving historical association.
All unknown source values receive zero correction. Source constants are not
additional input columns. Same-fold shared parameters must match bitwise.

Training copies original D2 optimizer, patient order, objective, shrinkage and
selection semantics. Timing/finite assertions and source-rank argument are
engineering instrumentation, not changes to science. Exact epoch-boundary
resume covers optimizer/RNG/history; original references are read-only.

Preprocessing is reproduced from original helper and checked bitwise against
frozen bank; no new population fitting or EEG extraction is performed.
Protocol, input, source audit and all executed code hashes bind every formal
cell. A single sealed private validation ledger drives all paired comparisons.

Bias/direction interventions use selected checkpoints and selected numeric
thresholds. No intervention is another trained arm. Bias-disabled ranking
reuses the unchanged full-model ordering after verifying source-constant
translation; direction-disabled ordering uses the shared base. This prevents
reporting float32 sigmoid rounding ties as biological rank changes.

All public CSV rows aggregate folds, sources, methods or feature-free mechanism
statistics. Effective 4x96 matrices and individual logits/labels/IDs are private;
only their singular values, norms, source-bias scalars and pairwise source
cosines are released. Factors are not interpreted as physiological causes.

All 34 required engineering checks and 15 actual FIT-only smoke updates precede
formal training. Synthetic interrupted training is isolated in a private test
directory and is not another formal run or predictive result.
