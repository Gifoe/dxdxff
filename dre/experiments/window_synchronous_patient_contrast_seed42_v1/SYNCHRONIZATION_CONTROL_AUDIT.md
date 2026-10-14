# Synchronization control — PASS

All quantitative gates in PROTOCOL_LOCK were fixed before table construction and label access. Context shifts are deterministic SHA256(seed42,patient,run,sample,canonical channel,stage), nonzero modulo eligible positions. Identical offset per feature when masks have identical lengths. Only finite context positions inside the same early/later stage move; self sequence, baseline, valid locations, patients and seizures do not move. No alternative shuffle was searched.

| Label-free check | Actual | Prespecified gate |
|---|---:|---:|
| Eligible context sequences |349,930|all eligible remain denominator|
| Nonzero shifted fraction |1.000000|>=0.95|
| Material-change fraction |1.000000|>=0.50; relative change>1e-6|
| Mean absolute zero-lag query/context correlation reduction |0.403042|>=0.02|
| Maximum cyclic lag1 correlation drift |7.77e-16|<=1e-10|
| Changed summary coordinates |56/84 (2/3)|>=10% at relative RMS>1e-6|
| Validity masks and context multisets |exactly identical|required|

Constant or short sequences were not filtered to improve these results; all sequences in this execution were eligible and materially changed. `SHUFFLE_DISRUPTION.csv` reports raw mean absolute change, relative change, context SD, real/shifted absolute zero-lag correlation and mean/SD preservation by feature, stage and center.

Circular rolling preserves temporal order **up to the boundary**, the value distribution and cyclic autocorrelation. It does not guarantee ordinary adjacent-window autocorrelation is identical; both ordinary versions and their difference are explicitly reported. These are overlapping2-second windows, not independent time samples.

Algebraically, the stage mean of d-m is unchanged when m is rolled over exactly matching valid positions. Thus28 context mean/across-seizure-mean-SD coordinates are effectively invariant; the identified contrast comes from the56 variance/slope summary coordinates. Merely adding84 context coordinates is not proof of synchronization; C1 versus identically sized C2 is the primary attribution control.

This gate passed before supervised fitting. No thresholds, feature subsets, stages, classifier or control were changed after predictive outcomes.
