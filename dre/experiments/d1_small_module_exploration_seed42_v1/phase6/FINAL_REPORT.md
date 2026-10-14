# Nonlinear conditional feature complementarity — completed, target not met

| Development model | AUROC | AP | Macro-F1 | EZ-F1 |
|---|---:|---:|---:|---:|
| Frozen D1 |0.725978|0.503601|0.644546|0.443537|
| Original M8 reference |0.756633|0.542226|0.666795|0.477249|
| M14 shallow complement |0.756633|0.542226|0.666795|0.477249|
| M15 depth3 complement |0.757947|0.542719|0.666807|0.480328|

All ten required candidate/fold cells actually fit200 boosting trees (2,000trees total), with selection among the eleven predeclared stages including frozenM8stage0. A cell selectingstage0 is not a skipped training run. Two synthetic tests passed locally/server before the new input freeze. Independent verification loads every saved model and replays its validation probability, checks130 patient metric cells against sklearn, reproduces FIT-only feature pruning and paired10000cluster bootstrap intervals, and checks original own-fold input/preprocessor identity/exclusion.

The nonlinear depth3winner's Macro-F1 gain overM8 is only0.00001236, not a useful improvement or0.70 result. M14recovers the originalM8mean because no validation stage warrants its learned correction. Additional processed88D evidence therefore does not establish reliable complementary value through these fixed residual trees. Do not infer that every feature is universally useless; this tests two specific bounded-capacity/prospectively chosen tree recipes on repeatedly viewed Development data.

The primary unit remains65patient/fold appearances and47uniqueIDs; all originalFIT/validation/TEST roles are preserved. This phase never accessesnewOuter. None of the three new phases reaches0.70, and earlier0.666795 is Development, not the completedM8Outer0.634097. All failed candidates/stages and their frozen artifacts remain retained rather than being erased or replaced by a label-derived threshold oracle.
