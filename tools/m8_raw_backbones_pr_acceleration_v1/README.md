# Execution-only acceleration of the frozen eight raw baselines

The user explicitly requested faster completion. This operational amendment does
not change the scientific model/data/protocol files in
`experiments/m8_raw_backbones_pr_seed42_v1/`, any training chunk size, optimizer,
gradient order, RNG, BN state, initial seed, epoch budget, validation selector,
Outer cohort or decision threshold.

## Admission actually run

The original four-backbone CUDA smoke was rerun while the original TimeConv
trainer remained active, in a separate admission runtime. It passed without
modifying scientific checkpoint files or the original admission audit.

`fast_engine.py` retains the exact original tensor math but collects per-chunk
finite flags on GPU and checks them together before the head/loss instead of
forcing a Python host barrier for each chunk. The full-raw finite check is
retained. The archived encoder's redundant finite check on each normalized
input chunk is removed at runtime; its input-shape check remains. Replay,
autocast boundaries, BN buffers, PR, masks and all gradients are unchanged.
The frozen source files on disk are **not** edited. External source hashes and
the original input/protocol hashes are locked in `EXECUTION_MANIFEST.json`.

A complete legal FIT patient (11,328 genuine channel-windows) was checked with
the same initialization/RNG. **Logits, loss, every parameter gradient, all BN
buffers and final CPU/CUDA RNG states were bit-identical**; maximum gradient
error0. The alternating warm timings were:

| Execution | Seconds |
|---|---:|
| Deferred checks |1.271985|
| Original checks |1.715477|
| Original checks |1.499978|
| Deferred checks |1.370299|

Warm means: original1.607728s, deferred1.321142s, approximately**1.217× throughput**
(17.8% less elapsed time). These are patient-level engineering timings under
concurrent load, **not** a measured whole-experiment speedup or localization
metric. The cold first pair was2.877529s versus1.210387s and is not used to
advertise steady-state speedup.

The patched execution then passed the original four-backbone forward/backward,
direct-versus-gradient-cache, masked/padded/PR/channel-permutation, FP32 chunk
invariance and full real-FIT admission checks again. No tolerances were loosened.

## Controlled checkpoint cutover and two lanes

`parallel_resume.ps1 -Mode Cutover` first verifies the exact original task parent
and TimeConv child process identities. It waits for a newly atomically published
`LAST_PRIVATE.pt`, which is written after a complete epoch. Only then does it
stop those two named task processes, copy and whole-file-SHA256-verify the saved
checkpoint into a private recovery backup, verify that the original input and
protocol hashes are unchanged, and enter the coordinator. No recursive process
tree kill or deletion is used, and unrelated jobs are not touched.

- **LaneA:** original frozen TimeConv trainer resumes its exact saved model,
  optimizer and RNG; completed folds are reused. Then its frozen Outer evaluator.
- **LaneB:** EEGNet, SEEG-Net, SEEGformer, in that order; for each, the original
  frozen trainer and one frozen Outer pass.
- Each lane has independent process/RNG/checkpoint paths and uniquely tagged
  logs. Both still use complete patients and all genuine windows.
- This changes **execution order/concurrency only**, explicitly in response to
  the acceleration request. It is not a new architecture or outcome-selected
  training change. GPU kernel scheduling need not make already nondeterministic
  library training a new bitwise-reproducibility guarantee across processes.
- Both lanes' individual results remain valid even if the other fails. The
  coordinator recomputes anonymous full tables once both finish, avoiding a
  possible concurrent partial-table-write race, then invokes the original
  independent verifier. It never uses earlier Outer outcomes to alter a later
  model.

On this Windows OpenSSH server, a detached `Start-Process` child can be reaped
when its short-lived owning SSH job exits. The initial detached operational
preflight exited before executing any cutover and did not affect training.
The production coordinator is therefore held by a live SSH/PowerShell execution;
its lane processes use hidden windows. The monitor must inspect actual process
and checkpoint state, not assume that a returned PID proves liveness.

## Monitoring and recovery

Inspect `CUTOVER_STATUS.json`, `CUTOVER_AUDIT.json`, `COORDINATOR_STATUS.json`,
`LaneA_STATUS.json`, `LaneB_STATUS.json`, their tagged logs/errs, and the real
Python process tree. These are in the trusted execution runtime. The legacy
shared `public/RUN_STATUS.json` can show whichever independent lane wrote last;
it is not a combined progress counter.

Never start the old serial `run_all.py` while either lane is active. Resume only
the failed lane after engineering diagnosis, with a **new** `RunTag`, preserving
all previous logs and completed checkpoints. Do not overwrite the execution
manifest or original frozen model/data sources. If the coordinator exits but
both lanes have completed, aggregate and verify only; do not rerun training or
scientific tests. Finite-check/source/data-gate failures require diagnosis, not
blind retry.

Only code, hashes and compact anonymous admission/cutover audits may be public.
All recovery checkpoint copies, patient/channel predictions, embeddings,
identities, weights and runtime logs remain on the trusted server.

## Completion and reporting-only verifier correction

Both lanes completed all training and all four frozen Outer passes by approximately
01:04 Asia/Shanghai on 2026-10-16. The coordinator then stopped at the original
independent verifier's `log_loss` assertion, not a model/inference failure.
The original verifier passed stored float32 probabilities directly to sklearn;
its clipping precision differed from the frozen metric engine, which explicitly
casts probabilities to float64 and clips at 1e-15.

The new, separately hashed `verify_reporting_v2.py` leaves every frozen scientific
file and the four operational files in `EXECUTION_MANIFEST.json` untouched.
It checks the original verifier SHA256, changes only one probability-conversion
expression in memory, and runs all its checks at the original 1e-12 tolerance.
This is not a new training/execution engine or a relaxed numerical comparison.
It performs no fitting, inference, threshold selection or prediction adjustment.

Actually verified: 640 patient metric rows, all 40 checkpoint/validation selections,
paired initialization identity, all eight 80-patient/7,635-channel configurations,
and 10,000 paired patient bootstrap draws. Original float32 sklearn log-loss
maximum discrepancy was 0.16906537126783738; float64 discrepancy is exactly zero.
Bootstrap maximum error was 1.3877787807814457e-16. All 57 protected artifact
hashes (input freeze, predictions, patient metrics, evaluation freezes/seals and
selected checkpoints) were unchanged before/after verification. See the published
`REPORTING_VERIFIER_AUDIT.json` and `INDEPENDENT_VERIFICATION.json` in the experiment
results directory. Original coordinator failure logs/status are preserved privately;
they are not overwritten to hide the precision-check failure.
