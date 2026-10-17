# Joint learned-response moments, Raw-only Macro-F1, seed42

Prospective bounded experiment. **No result or >0.65 achievement is claimed.**
The preceding complete fresh factorial failed the full80 target; its actual
negative results remain in
[the prior report](../raw_fresh_energy_macro_seed42_v1/RUN_REPORT.md).

The new hypothesis is information complementarity, not energy superiority.
One CNN supplies mean and log1p(mean squared) learned responses to one head.
Three14,577-parameter random-start networks compare (mean,mean),
(energy,energy), and (mean,energy). Duplicated controls have the identical
input/head shape and no additional statistic; all three have exact matched
initial CNN/head tensors. Generic moment pooling is not claimed novel or a
measured physiological band. No independent D1/physiological branch, static
predictor features, prior trained weights or old logits enter a new network.

All preserve pre-latent reference, two temporal phases, available-seizure
mean/SD and complete-patient Z. The aggregation's dimension generalization
is tested for exact24D output **and gradient** parity with the original core.
All valid Raw windows are forwarded; original pre/0to20 support is retained,
so late encoded windows are not claimed to affect final predictions.

`PROTOCOL_LOCK.json` fixes models, normalization, loss, optimizer, roles,
budgets and gates before training. Screen1 runs all3×5fold=15fits. Only
advance-gate-passing ownFITs run all three arms on the two additional paths,
at most45fits total. CAL alone selects epoch/threshold; HELD selects within
its own originalFIT. Original Development/Outer are excluded from this stage.
Complete first-path and selected replication tables remain distinct.

Five changed-path tests and actual largest complete-training-patient
backpropagation precede training. `run_internal.ps1 -RunTag <fresh>` requires
a sustained foreground SSH owner, retains all logs and atomically checkpoints
last complete model/optimizer/RNG state. No duplicate jobs or blind retries.
On completion, `summarize.py` independently replays CAL/HELD and all nine
patient metrics, clustered intervals, race decisions and ownFIT choices.

A final originalFIT/Development program is only warranted after actual
independent internalPASS and an advanced fold, and requires a separate
source/choice/checkpoint/threshold freeze before any new Outer inference.
All historical Outer has been reused: exploratory repeated-test, not blind
confirmation. No best-Outer secondary adoption or metric substitution.

Publish only source, hashes and actual anonymous aggregates. Raw, IDs,
labels, weights, individual predictions/latents, caches and runtime logs stay
private. Unrun results are not filled by templates or proxies. The broad
full80 equal-patient Macro-F1>0.65 goal remains active and unachieved.

## Actual execution snapshot, 2026-10-17

Scientific code/protocol were pushed at `c66da777960020766ab3293362bc9b7eff01f638`
before formal fitting. Five unique changed-path tests passed. The largest
complete legal training patient (164channels/48,380validwindows) passed full
backpropagation for allthree networks, with finite parameter gradients and
nonzero CNN/head gradients. Admission made zero optimizer updates; its actual
[aggregate artifact](results/internal_v1/ADMISSION.json) is included.

At07:54Asia/Shanghai, the first complete internal fit was saved and the second
was actively training, error log empty. This is a dated progress snapshot,
not a claim that the full internal matrix or any new Outer is complete.
No new performance or target achievement is claimed until actual verification.

## Guarded final implementation, not an executed result

`FINAL_PROTOCOL_LOCK.json`, `final_stage.py`, `verify_final.py` and
`run_final.ps1` implement the already registered conditional final plan. They
do not alter any live internal training source. Entry requires completed
independent internalPASS plus allfive ownFIT choices and an advancedfold.
Then actual complete originalFIT admission precedes15freshFIT/Development
fits,30model/thresholdfreezes,one joint sealedOuter pass,all480patientmetric
rows and126absolute/pairedCI checks. The primary remains SELECTED, not the
best fixed Outer point. A new post-seal Raw mask census is also mandatory.
`check_final_gate.py` checks denial while the real internal terminal is absent;
it performs no fitting, prediction or Development/Outer data loading.
That negative-entry check passed on the server while the real internal
terminal was absent. It is a workflow guard check, not a model performance
result. This final implementation is not automatically run by the internal
coordinator and must not be launched while an internal worker is active.

`handoff_after_internal.ps1` is a one-shot execution sequencer, not a new
selection rule. Under a sustained foreground SSH owner, it watches the exact
original coordinator PID plus creation time, waits for normal exit, then
requires independent internalPASS/126CI checks, frozen fivefold choice hash,
no remaining owned worker, and exact precommitted final source hashes. Only
then does it invoke the unchanged `run_final.ps1`; any missing gate or failure
stops without retry. It never terminates a process, changes a model or starts
a second active lane. Its private operational status/history is not clinical
evidence or a substitute for actual final results.
