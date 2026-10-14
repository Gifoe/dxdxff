# Macro-F1>0.70 exploration — Phase4 completed, target not met

| Development model | AUROC | AP | Macro-F1 | EZ-F1 |
|---|---:|---:|---:|---:|
| Frozen D1 |0.725978|0.503601|0.644546|0.443537|
| Frozen M8 |0.756633|0.542226|0.666795|0.477249|
| M9 patient-context shift |0.756633|0.542226|0.668015|0.474158|
| M10 context readout |0.757119|0.540293|0.666772|0.476561|
| M11 sharpened readout |0.756729|0.538633|0.666165|0.476654|

All15actual fold/candidate training cells and171epochs completed. Four synthetic tests passed locally/server before input freeze. Allfive frozen M8 validation logits are exactly reproduced in the FIT/VAL evidence-bank extraction. Independent verification replays all15adapter checkpoints, independently recomputes195patient metric cells and FIT-only feature normalizers, validates ownfoldFIT/validation/TEST separation and paired10000cluster-bootstrap intervals. Original scores/checkpoints/scales are not modified.

The common winner is M9, but its gain over M8 is only0.00122045. Ranking is unchanged and EZ-F1 declines slightly; these are not evidence of a useful0.70 breakthrough. Nonlinear frozen-evidence readouts/stronger softF1 did not improve the target. The label-derived threshold ceiling0.728583 is only a diagnostic upper bound, never legal model performance; fixed top20% yields0.637328 and is not adopted.

Primary unit remains65original validation appearances/47uniqueIDs, not80Outer patients. Every threshold/model is selected using the repeatedly used validation set. These outcomes are adaptive hypothesis generation, not independent confirmation. The full objective remains unachieved. No newOuter was read in this phase; no target/denominator is lowered. Complete precision, folds/centers, all training epochs and uncertainty are retained in `results/`.
