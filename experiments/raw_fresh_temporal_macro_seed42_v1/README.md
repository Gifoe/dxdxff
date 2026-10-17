# Fresh temporal interactions for Raw-only Macro-F1, seed42

Prospective bounded internal study. **No result has been obtained.** Target:
full80 equal-patient Macro-F1 >0.65, not an internal, pooled, EZ-F1 or oracle
metric. The preceding studies remain published with their negative results.

The older long-RF from-scratch trial was EZ-oriented. The newer matched
temporal/pointwise Macro trial warmed an already trained two-waveform model,
with eligible epoch0 frequently retained. This study instead trains a compact
single-waveform CNN and its sole classifier from random initialization; the
new modules do not enter a frozen or warm-start body. This is not a novelty
claim for generic depthwise CNNs, and the older failures are not concealed.

| Fresh configuration | Information path | Parameters |
|---|---|---:|
| FRESH_BASE | Original two-convolution, global mean | 8,433 |
| FRESH_POINTWISE | 984-parameter per-position nonlinear residual before mean | 9,417 |
| FRESH_TEMPORAL | 984-parameter depthwise k17 + pointwise residual before mean | 9,417 |

All three use one96to64to1head and exact matching initial CNN/head tensors.
Both residual outputs initially equalzero, so initial logits must match the
base exactly. Adapter shapes differ: complete state dictionaries are **not**
claimed identical. Every retained checkpoint is a trained epoch>=1. The
longest local interaction path grows from33to289input samples before global
pooling; global pooling still aggregates every latent position, so the old
CNN was not receiving only33physical samples in total.

All original pre-latent reference, two phases, available-seizure mean/SD,
complete-patient Z and pre-only waveform normalization remain exact. There
are no manual physiological predictor slots, D1 branch/logits, old weights or
embeddings, window sampling, new extraction or changed patient roles. No
classification metric may be inferred from admission or synthetic tests.

The first path compares all three fresh models with Native, frozen
precompand, prior compact PRE_MEAN and the immediately preceding retained
pre-state model. The latter improved internal ranking but not Macro-F1 and
never qualified for a new Outer run. All four are hash-verified fixed
references, not inputs to new models. Missing old replication cells stay
missing; complete first-path scores are not mixed with unequal subsets.

There are15first-path fits and at most45including only prequalified ownFIT
replications. The fixed first gate requires >=0.01Macro gain against the
strongest reference and the AUROC safety condition. The replicated final
gate additionally requires >=0.015Macro gain with positive paired lower
limits and path consistency, as specified in PROTOCOL_LOCK.json. Procedure
selection is confined to each original ownFIT, not a global winner across
overlapping folds. A legal fixed fallback is chosen among the three
deployable references using their same first path, never viewed Outer.
If no fresh candidate qualifies, there is no new Development or Outer run.

Before training: five changed-path tests and actual largest legal complete
patient backward admission. Full initial logits and shared CNN/head weights
must match across all three models, with finite gradients and zero admission
optimizer steps. The original fresh fit engine is byte-pinned; only its
parameter-count reporting helper and progress labels are bound to this
experiment. Sources/protocol are pushed before formal training.

After training: all retained CAL/HELD predictions replayed,201threshold
decisions checked, nine independent patient metrics, source/fold coverage
and198absolute/paired interval rows. First-path assessment has86appearances
of59unique people, not a new80patient test. Repeated patient appearances are
kept together in10,000bootstrap draws. Original Development/Outer remain
unavailable to this internal program. A later final program would need a
separate model/threshold freeze, and its results would still be exploratory
repeated-test, not independent confirmation.

`run_internal.ps1` requires a sustained foreground SSH owner. Preserve every
log/checkpoint and diagnose nonzero exits before safe exact resume; do not
duplicate a running worker or automatically retry a Python error. Upload
only code, hashes and actual anonymous aggregates. Raw, truth, identities,
weights, individual predictions/latents and runtime logs remain private.

## Actual prebinding check, 2026-10-17

The deployed source manifest passed and all five unique CPU changed-path
tests passed (pytest:5 passed in1.22s). These are executable checks, not
classification results or proof of the full80target. Complete-patient GPU
admission remains mandatory before formal fitting. Source/protocol are
pushed before any formal optimizer update.

## Actual GPU admission and training start

Source/protocol commit `f266f04b224463ff02a34fdcaaf568d055ede568` was verified
at GitHub before formal fitting. Complete-patient GPU admission then passed
all three networks on164channels/48,380validwindows, the largest actual legal
training patient among79unique training patients. Initial logits were exact
across all three, shared CNN/head states matched in all five folds, all
parameter gradients were finite and both new output projections had nonzero
gradients. Admission performed zero optimizer updates; see the actual
[ADMISSION.json](results/internal_v1/ADMISSION.json).

At09:39Asia/Shanghai on2026-10-17, FRESH_BASE fold1 was actively training
through epoch5 at approximately1.9seconds/epoch, error log empty. This is a
dated progress snapshot, not a classification result or an overall runtime
estimate. The original Development/Outer were not used. No target achievement
is claimed from this admission or the still incomplete internal matrix.
