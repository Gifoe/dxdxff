# M8 Raw-only matched ablation — seed42

This experiment removes the 14D/D1 base prediction and the7D physiological
conditioning gate from Full M8. The new network consumes only aligned Raw250
waveforms, valid masks and original window times. It does not receive14D,
7D physiology, D1 logits, patient IDs or center IDs. No installed skill is used.

## Kept and removed

Retain the original M8 CNN (1→12,k9,s4;12→24,k7,s4; GELU; adaptive mean pool),
channel-local waveform normalization, pre-onset same-channel latent reference,
early0–8s/late8–20s pooling, available-seizure mean/populationSD,96D full-patient
PR and96→64→1 GELU head. Only the CNN/head train:8,433 parameters.

Raw features exactly match the original M8 factory's identity-gate features
on a real legal FIT sample (maximum error0). Original fold-specific random CNN/head
initial states are copied **before training**, not from a trained Raw model.
Constructing the old factory is solely for RNG/initialization and feature-parity
checks: its D1/base and physiology gate are discarded and absent from the new
network. D1 is never called for a new Raw-only prediction.

Keep the original correction readout `tanh(head(z))` as the standalone NEZ logit,
with base logit removed. Thus the logit remains bounded±1 and its NEZ probability
lies in approximately[0.269,0.731]. This is deliberately the inherited M8 Raw-path
readout, not an unconstrained standalone CNN classifier. Do not remove the cap
after viewing results. It limits the interpretation of a negative Raw-only result.
The head's final layer starts at zero; both CNN and head learn end-to-end.

Both the14D anchor and physiology guidance are removed jointly. This is the
requested strictly Raw-input configuration, not a single-site causal ablation
of only14D versus only the conditioning gate. Full M8's base-plus-residual
optimization is necessarily replaced by standalone Raw classification.

## Data, training and selection

Reuse the original80patients,256seizures,7,635canonicalchannels and1,471,965valid
channel-windows, existing Raw250/500sample banks and genuine boundary/missing-channel
masks. The original pre-reference and0–20s post-onset summaries remain. All valid
windows are encoded as in M8, but times≥20s do not enter those summaries, as in
the original model. No new extraction, downsampling, selected patients or windows.

All five exact original FIT/validation/Outer patient splits; seed42+1009×fold.
Original M8 AdamW lr0.0003/wd0.001, gradient clip1, max30/min6/patience6,
FP32/TF32off, chunk4096, one equal-patient mean update per FIT patient. Preserve
the original objective:0.75 FIT-only NEZ-positive weightedBCE +0.25 patient soft
Macro-F1. Model/threshold selection uses the original validation-only M8 selector,
not an Outer-driven or newly chosen threshold rule. Report EZ=1 with
`pEZ=1−sigmoid(logitNEZ)` and `predEZ=(pNEZ<thresholdNEZ)`.

Three legal FIT admission-gradient steps are discarded and the same seeded model
is rebuilt before formal training. Atomic complete-epoch optimizer/model/RNG
checkpoints support resume. Source, protocol, dependency and original reference
hashes are locked. All five selected checkpoints/thresholds freeze before the
single new Outer pass; private per-patient logits seal before full Outer labels.

## Reference comparisons and outputs

Existing frozen D1 and Full-M8 score ledgers are hash-verified and reused only
for descriptive/paired reference evaluation; no new reference inference, fitting,
threshold changes or training on their predictions. Verify their primary
patient metrics against the prior private ledgers exactly.

Anonymous public outputs include patient-macro, pooled, fold/center, coverage,
confusion, validation selection/training aggregates, independent metric/source
verification and10,000seed42 fixed-threshold paired patient bootstrap intervals
for Raw-only minus D1/Full-M8. NaN coverage is explicit. Labels/IDs, raw data,
individual predictions/embeddings, weights, checkpoint paths and runtime logs
remain private on the trusted server. No GitHub PR is created.

Run `run.py` with the pinned parent artifacts and original existing environment;
set OMP/MKL/OPENBLAS threads1. Keep foreground SSH alive. A matching complete
final status prevents a second scientific evaluation. Original experiments and
unrelated jobs are not modified.

Scientific status: **EXPLORATORY_REPEATED_TEST**. This cohort and M8 architecture
were repeatedly viewed/adaptively developed historically; this is not blind
external confirmation. Do not infer that Raw is intrinsically uninformative from
one small inherited-cap model and seed, or attribute the joint ablation entirely
to either removed component.

Actual final metrics will be added after independent verification, not inferred
from smoke tests or earlier unrelated Raw backbones.
