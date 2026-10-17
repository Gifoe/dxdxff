# Early learned Raw support x response pooling, seed42

Prospective bounded internal experiment. **No performance result has yet been
obtained.** The target remains full80 patient-equal Macro-F1>0.65. No other
F1, internal point, oracle, favorable fold or interval upper bound substitutes.

The preceding fresh late-temporal adapter failed against both controls. It
mixed latents after two stride4 reductions. This study instead learns extra
first-filter Raw support before the first reduction, crossed with the learned
response statistic. It is not another late adapter or a claimed novel
physiological feature. All earlier negative experiments remain published.

| Factorial cell | First filter support | Learned response pool | Parameters |
|---|---|---|---:|
| SHORT_MEAN | Original9tap | Mean | 8,433 |
| SHORT_ENERGY | Original9tap | log1p(mean squared response) | 8,433 |
| LONG_MEAN | 129tap center/outer parameterization | Mean | 9,873 |
| LONG_ENERGY | 129tap center/outer parameterization | log1p(mean squared response) | 9,873 |

The long stem is `Conv9(x)+Conv129(x, outer60/zeros9/outer60, pad60)`.
Its1,440outer weights startzero, while the center is newly sampled from the
same random initialization as the short model. It keeps exactly the original
123first-layer temporal positions; matching-pool initial outputs are identical.
Extra Raw support has a gradient immediately, rather than waiting for a
zero-output late projector. The structural local field before global pooling
is33versus153input samples; initial long functions still equal short. Global
pooling aggregates every latent position, not only33physical samples total.
All four share the same random core and compact96to64to1head. No trained
control weights are injected into a new variant. Wider support also adds
capacity and boundary terms, so its effect is not a pure time-only cause.

SHORT_MEAN is the **identical** already trained FRESH_BASE from the preceding
study, not its different old head160 PRE_MEAN. Its five first-path and two
fold4 extra cells may be reused only after exact lineage/role/initial-state,
actual checkpoint replay and threshold-decision checks. Reuse counts/old
training epochs stay distinct from new fits/epochs. Missing extra control
paths are actually trained if needed, not fabricated. Firstpath requires15
new fits plus5real reused controls, avoiding5redundant baseline fits.

All models receive only the exact Raw waveforms, masks and times. Pre-only
waveform normalization, pre-latent reference, two phases, available-seizure
mean/SD, complete-patient Z, original cohort/roles and full-window forwarding
remain unchanged. There are no fixed physiological bands, manual predictors,
D1 branch/logits, identity predictors or extra classifier. The energy is a
generic statistic of learned responses, not a hand-measured PSD feature.
No assertion of recovered low frequencies or full>125Hzpower is made.

The four fixed comparison arms retain Native, frozen precompand, legacy
PRE_MEAN and prior pre-state scores. They differ in parameterization/history,
are hash-verified and never enter a new model as inputs. No global winner is
chosen from overlapping folds. Every choice stays inside its original ownFIT.

The frozen race initially requires>=0.01Macro gain over all fixed references
and SHORT_MEAN, plus AUROC safety. Advancing ownFITs run all four cells on two
more paths. Final eligibility additionally needs>=0.015Macro gain with positive
paired lower limits and consistency against each of Native/precompand/SHORT_MEAN.
At most55actual new fits, with all reused cells disclosed separately. If no
new variant qualifies, there is no new Development or Outer evaluation.

All eight first-path arms and nine metrics, complete factorial main effects/
interaction, fold/source stability and261absolute/paired/factorial CI rows are
verified and reported from real runs. Assessment has86appearances of59people,
not a full80test. Repeated appearances are clustered together; these are
reused exploratory data, not independent confirmation.

Before training: six unique checks, seven actual old-control replays and full
backpropagation on the largest complete legal training patient for all four.
No optimizer update occurs in admission. Sources/protocol are pushed before
formal fitting. `run_internal.ps1` must retain a foreground SSH owner; never
duplicate a live worker, alter frozen sources, discard completed epochs or
blindly retry a Python error. Only code/hashes/anonymous actual aggregates go
to GitHub. Waveforms, truth, identities, weights, private predictions/latents
and all runtime logs remain on the trusted server.

2026-10-17 prebinding CPU preflight: six tests actually passed (0.95s),
with zero formal updates and no new Development/Outer. One ad-hoc launch
aborted from torch-before-numpy OpenMP initialization; the existing scientific
import order (numpy before torch) passed without an unsafe duplicate-runtime
override or source/math change. Actual GPU admission is still pending.
