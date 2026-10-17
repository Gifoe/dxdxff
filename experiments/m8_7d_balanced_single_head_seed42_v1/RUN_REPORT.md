# Original7D balanced single-head: complete fivefold result

## Outcome

The preselected primary **C2_INTERACTION** obtained full80 patient-equal
**Macro-F1=0.6396579900303192**, versus0.6334923520820233 for the exact
existing7D+Gate single-head control. The point gain is0.006165637948295843,
but the requested **strictly>0.64 target is not met**. The shortfall is
0.0003420099696808; rounding to0.640 does not turn this into a pass.

The paired95% Macro difference interval is[−0.004708959,+0.017371172].
This does not establish statistically confirmed improvement. This is an
exploratory repeated test on a historically viewed cohort, not independent
predictive confirmation. C2 was selected using Development before test;
neither C1 nor another endpoint is substituted after observing Outer.

## Architecture and unchanged inputs

All original seven measurements are retained: Beta Power, Low-Gamma Power,
Partial High-Gamma Power, RMS, Variance, Line Length and Spectral Entropy.
Their original DELTA, seizure mean/SD, Patient Z and FIT-only preprocessing
produce the same14D input. No feature deletion, new extraction, band repair,
resampling, source/identity predictor or quality-based exclusion was used.

Raw uses the exact original250Hz/500sample windows,24D CNN, pre-latent
reference subtraction,0–8/8–20s phase pooling, available-seizure mean and
population SD,96D complete-patient Z, original7→32→24 Gate and all masks.
Each new model covers80patients/7,635channels/256seizures and1,471,965valid
channel-windows. Partial High-Gamma is not claimed to be full80–150Hz.

There is one110→64→1 classifier with an unbounded NEZ logit. C0 is exactly
the original S3 network. C1 splits the first matrix into Raw/physiological
hidden contributions:

`a = W_raw [sqrt(110/192) r96]`

`b = W_phys [sqrt(110/28) f14]`

`hidden = a + b + bias`

C2 uses `hidden = a*(1 + gamma*tanh(b)) + b + bias`, where
`gamma=0.5*tanh(gain)` and gain starts at zero. A single GELU and one final
scalar Linear produce the prediction. These64D contributions are not
independent classifiers; no D1 logit, pretrained D1 output or prediction
residual enters the network. All initial CNN/Gate/head tensors match across
arms. C1/C2 start with identical classifier functions; the interaction has a
gradient immediately. CPU function/shared-gradient checks and actual complete
patient GPU Raw-path/gradient checks are documented separately.

C0/C1 have10,377parameters and C2 has10,378, all trainable. Dimension balancing
does not enlarge C1's function class relative to an ordinary concatenated MLP;
it changes parameterization, gradient scales and effective weight-decay
geometry. Thus a gain is not proof of a novel feature or uniquely necessary
fusion mechanism. The interaction is also generic, not a novelty claim.

## Actual training and selection

Two new candidates×five original folds completed10real fits and114epochs.
Five exact existing S3 controls were reused after source/input/checkpoint/
role/initial-state replay checks, not counted as new fits. No trained control
or D1 state was loaded into C1/C2. All arms use the original FP32/TF32-off,
AdamW(.0003,weight_decay.001), clip1, max30/minstop6/patience6, patient-equal
complete updates and0.75weighted NEZ BCE+0.25soft patient Macro loss.
Epoch/threshold selection uses the unchanged original rule and201NEZ grid.

Development has65appearances of47people and is reused for epoch, threshold
and architecture selection. It is not unbiased nested-CV. C1/C2 passed the
registered Macro+0.003/AUC−0.01/AP−0.01 gate versus C0; C2 had the larger
Macro−0.1fold-SD score. All15actual validation checkpoints replayed with
maximum logit error zero, exact threshold decisions,1,755independent metric
checks and54independently checked intervals. The choice was sealed before
the separate final protocol and source were published.

Before new test input access, all15models/epochs/thresholds, preprocessors,
Gate scales, source/input/cohort hashes and the selected primary were frozen.
Ten new model-fold inference cells and five exact old S3 control score cells
were sealed before the report process read full truth. All160new model-patient
cells have private atomic, hash-bound caches. No model or threshold was changed
after this evaluation. Original source, checkpoints, scores and paused Raw-only
experiments remain intact.

## Full80 patient-equal results

AP is average precision; ACC is ordinary accuracy, not balanced accuracy.
Macro-F1 is the mean of EZ/NEZ F1 within each patient, then averaged equally
over all80patients. MRR is reciprocal rank of the first true EZ; Top-1 is
the fraction whose first-ranked channel is EZ. All9metrics are finite80/80.
Pooled confusion counts and unweighted fold summaries are separate artifacts,
not substitutes for the primary denominator.

| Model | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ frac. bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 original7D single-head | .633492 | .753681 | .572050 | .438524 | .828460 | .754404 | .812191 | .725000 | −.028580 |
| C1 balanced | .639572 | .769705 | .582314 | .448818 | .830326 | .758699 | .803898 | .712500 | −.025711 |
| **C2 balanced+interaction, preselected** | **.639658** | **.769704** | **.582443** | **.448853** | **.830463** | **.758853** | **.803898** | **.712500** | **−.026113** |

| C2−C0 metric | Difference | Nominal patient-paired95% CI |
|---|---:|---|
| Macro-F1 | +.006166 | [−.004709,+.017371] |
| AUROC | +.016022 | [+.003609,+.030901] |
| AP | +.010393 | [−.003469,+.025235] |
| EZ-F1 | +.010329 | [−.008516,+.029497] |
| NEZ-F1 | +.002002 | [−.003230,+.007435] |
| ACC | +.004449 | [−.003052,+.012228] |
| MRR | −.008292 | [−.052455,+.035739] |
| Top-1 | −.012500 | [−.087500,+.062500] |
| EZ fraction bias | +.002467 | [−.007716,+.012755] |

Intervals use10,000seed42 paired patient draws, fixed models/thresholds and
nominal percentile limits, without multiplicity or historical-adaptation
correction. A positive signed bias difference is not inherently an improvement.
All3absolute and3paired contrasts×9endpoints are reported, not only this table.

C2−C1 Macro is only+0.000086 [−0.000573,+0.000832]. Almost all of the observed
point gain is already present in the balanced parameterization. There is no
evidence here that the added interaction is necessary. AUROC/AP and class-F1
point estimates improve versus C0, while MRR/Top-1 decline: this is not uniform
localization improvement or the previous eight-metric noninferiority claim.

## Fold/source stability

| Fold | C0 Macro-F1 | C1 Macro-F1 | C2 Macro-F1 |
|---|---:|---:|---:|
|1|.606152|.604546|.604546|
|2|.632722|.646472|.645517|
|3|.628679|.639837|.641141|
|4|.626451|.612202|.612202|
|5|.673319|.693075|.693075|

Macro fold sample SD increased from.024531(C0) to.034914(C2); AUROC fold SD
increased from.044040 to.053528. C2 improves Macro in3/5folds, not all folds.

| Historical source group | Patients | C0 Macro-F1 | C2 Macro-F1 |
|---|---:|---:|---:|
|hup|36|.655496|.666225|
|lzu|21|.566053|.570828|
|multicenter|15|.667190|.666519|
|pediatric|8|.648321|.650420|

These are inherited source labels, not four equally sized independent hospitals.
The smallest group has8people; no source-specific threshold or exclusion is
introduced. Exact complete tables, coverage and all endpoints remain in CSVs.

## Verification, limitations and next action

Independent final verification passed240patient rows/2,160metric entries and
all54intervals; maximum interval difference was1.1102230246251565e−16.
Both new arms independently cover the full1,471,965valid Raw windows and
the exact80patients/7,635channels. The old S3 scores reproduce all original
fullprecision metrics. Both coordinators ended normally; no new-stage stderr,
native recovery, checkpoint loss, retraining or duplicate scientific test
occurred in this trial.

Exact terminal:
`BALANCED_7D_SINGLE_FULL80_OUTER_PIPELINE_VERIFIED_COMPLETE`.

Final input/source/model freeze SHA256:
`c36dae333719f4c40497e13e9531e0afa65e5b4b46530ac4aad232da47da7201`.
All-score seal SHA256:
`9f5beef8b232cff825cf1a3b560d9a54f10bdfd4ee9bdf27f3275b3bb7774e6c`.
The12file anonymous final delivery ZIP SHA256 is
`da5b80c6cbb1e21983e1544436d91e4ff6df2fb02716e415d8afeca2553aa340`.

This experiment is complete, but the overall>0.64goal is not. Do not tune this
frozen model, threshold or selection to cross the remaining gap. A further
study must be separately registered, keep the same7D, and select architectures
inside legal training/validation data. A mechanistically distinct question is
whether physiology-first training of the **same final single head**, followed
by joint Raw adaptation, can preserve the physiological anchor without an
independent D1 prediction branch; this has not been run or claimed successful
here. Wider arbitrary modules or more feature coordinates are not justified
by the negligible C2−C1 difference.

Only code/hashes/anonymous aggregates and this technical report are published.
Raw signals, truth, identities, checkpoints, private predictions/embeddings,
scalers and runtime logs remain on the trusted server.
