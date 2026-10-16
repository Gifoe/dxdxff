# Fresh-start Raw response-energy factorial, Macro-F1, seed42

Prospective protocol. No new result or target achievement is claimed yet.
Goal: original complete80-patient equal-weight Macro-F1 >0.65. Existing exact
Native Raw-only reference: 0.6304930529317453. Earlier negative warm-adaptation
and EZ-first studies are preserved, not relabeled as successful Macro fits.

Four random-start equal-capacity networks form a 2x2 comparison: original
window normalization vs fixed pre-only common-scale asinh normalization, and
mean pooling vs `log1p(mean(learned_response^2))`. All accept only waveform,
valid mask and original times. Every filter and head is jointly trained; there
are no preset frequency bands, hand-computed physiological predictor slots,
D1 output or second classifier. The pooling operator is generic, not claimed
novel. Response energy must not be mislabeled as measured raw bandpower.

All four have 12,529 parameters, identical random CNN/head weights per fold,
and unchanged pre-reference/phase/seizure-statistics/full-patient-Z aggregation.
There is no CAL-selected initial checkpoint. A longer but finite Macro-aligned
budget tests a different training protocol; any gain over historical Native
alone does not isolate the pooling operator. Structural attribution requires
the matched fresh mean controls and factorial interaction.

`PROTOCOL_LOCK.json` fixes all choices before fitting. Screen1 trains all four
arms in allfive ownFITs (20 fits). Only registered gate-passing folds advance
the frozen candidate plus its mean controls to the other two paths (at most
50 fits total). First-path and advanced-subset results remain separate. CAL
chooses checkpoints/thresholds; HELD chooses procedures within ownFIT. Original
Development and Outer are excluded from this internal program. A separately
frozen final implementation is warranted only after its internal gate passes.

Trusted Windows execution: `run_internal.ps1 -RunTag <fresh>` under a sustained
foreground SSH owner. Changed-path tests and largest complete-patient gradient
admission precede all formal fits. Last-complete-epoch model/optimizer/RNG states
permit diagnosed exact resume, never blind Python/source/data retries.

Publish only code, hashes and actual anonymous aggregates. Raw, IDs, labels,
weights, individual predictions/latents and logs stay private. Any future Outer
is exploratory repeated-test because historical outcomes have been viewed.
An internal result, green admission, oracle or pooled metric cannot establish
the requested full80-patient target. No unrun cells will be filled with proxies.

## Completed internal stage, final stage pending

All42registeredfreshfits/1062epochs and independent replay finished. First-path
Macro-F1: WINDOW_MEAN0.600015, WINDOW_LOGENERGY0.599525, PRE_MEAN0.625949,
PRE_LOGENERGY0.624857 (86appearances/59unique people, not full80Outer).
The matched data do not establish an energy-pooling advantage over mean.
Legal frozen choices are FROZEN_PRE/NATIVE/PRE_LOGENERGY/PRE_LOGENERGY/
PRE_LOGENERGY for folds1–5. See [INTERNAL_REPORT.md](INTERNAL_REPORT.md) and
[actual internal tables](results/internal_v1/FIRST_PATH_MAIN.csv).

`FINAL_PROTOCOL_LOCK.json` and final code were frozen before original
Development access. `run_final.ps1` performs the20freshfinalfits and one joint
35-cell sealedOuter evaluation; the primary is the ownFIT-selected policy,
not the best secondary Outer arm. No final result is claimed until its actual
completion and verification. `python verify_public.py` checks public delivery.
