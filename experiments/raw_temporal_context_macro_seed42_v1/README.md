# Shared Raw temporal-context adapter, Macro-F1, seed42

Prospective experiment, not a claim that the >0.65 target has been met.
The primary endpoint remains complete 80-patient equal-weight Macro-F1; the
exact existing Raw-only reference is 0.6304930529317453. Prior threshold and
pooling studies are preserved; their negative results are not relabeled here.

The original learned local response has a 33-sample receptive field before
global window averaging. This study compares a 984-parameter temporal
depthwise/pointwise residual to a 984-parameter pointwise channel MLP and a
matched no-added-module fine-tune. The temporal local receptive field is 289
samples. Both original and adapted models still globally pool all 500 samples;
expanded local interactions do not mean that only the new model sees the whole
window. No external novelty claim is made for this generic construction.

All models retain the single shared CNN/head and preceding frozen waveform
normalizations, phase/reference/seizure aggregation and full-patient Z. Initial
Raw CNN/head and precompand-fusion matrix come from the exact same legal
ownFIT path. Added residuals are zero, so all three models have identical
initial predictions. Training uses only complete TRAIN+CHOICE patients;
CAL chooses epoch/threshold; HELD selects within ownFIT. No D1, manual
physiological inputs, patient/source identity predictors or window sampling.

`PROTOCOL_LOCK.json` fixes optimizer, Macro-weighted loss, finite budgets,
capacity controls, original threshold grid and advancement gate before fits.
First path trains all three models in all five ownFITs (15 fits). Only folds
that pass the +0.01 Macro and AUROC safety gate advance to the two additional
paths (at most 45 fits). No unrun path will be filled with a proxy result.

Execution on the trusted server: `run_internal.ps1 -RunTag <fresh>`, held by a
sustained foreground SSH owner. It first admits the changed critical paths and
largest actual complete training patient; it then trains exact full-patient
forwards and independently replays all selected states and metrics. Atomic
last-complete-epoch optimizer/model/RNG checkpoints support diagnosed resume.

Original Development and Outer are not used in these internal scripts. A
final fullFIT/Outer stage is warranted only if the predeclared internal gate
passes; it needs its own frozen final implementation. All historical Outer is
exploratory repeated-test, not independent confirmation. Public delivery is
code, hashes and anonymous aggregates only; private data, identities, weights,
scores, raw banks and logs remain on the trusted server.
