# Joint readout/cap fine-tuning — Phase5 completed, target not met

| Development model | AUROC | AP | Macro-F1 | EZ-F1 |
|---|---:|---:|---:|---:|
| Frozen D1 |0.725978|0.503601|0.644546|0.443537|
| Original frozen M8 reference |0.756633|0.542226|0.666795|0.477249|
| M12 joint physiology/readout |0.757303|0.542413|0.667720|0.476846|
| M13 learnable bounded cap |0.757325|0.542647|0.667791|0.476912|

Allten required fold/candidate training cells completed, with originalM8eligible at epoch0. Some folds legitimately select epoch0 after actual training fails to improve validation; this is not a skipped training cell. Two synthetic identity/permutation/gradient tests passed locally and server before freeze. Actual legalFIT starting-logit identity checks pass. D1weights remain exactly unchanged in allten selected checkpoints.

The new110Dcorrection readout jointly sees frozen-preprocessedR14and morphology96. M12has10,377 trainable correction parameters; M13adds one scalar for10,378. The optimizer fine-tunes the new model, never overwrites originalM8/D1checkpoints or changes their normalization/window/reference caches. Selection and class weights remain train/validation-only. Independent sklearn calculations check130patient metric cells, checkpoint/bank hashes, held-out role exclusion and paired10000bootstrap intervals.

Best Macro-F10.667791 does not reach0.70 and does not outperform the earlier phase's0.668015. The almost equal results do not establish a useful learnable-cap mechanism. Original model epochs, all new training histories and unfavorable folds remain published; no newOuter is accessed and no post-test rescue is made. The task remains active, with the prospective nonlinear feature-complementarity phase testing the next mechanism.
