# Bounded nonlinear transfer of Raw-conditional physiological selection

This separate internal-only study addresses the untested transfer of the
registered conditional ridge signal to nonlinear neural readouts. It does
not alter previous models or convert the experiment into a new Outer test.
The full research objective, stable minimum and actual end-to-end Raw/M8
increment, remains in scope and unproven. These intermediate readouts are not
a substitute for full M8 evidence.

Each of15 patient-held-out assessment paths chooses its own k from its
existing canonical feature-choice path, without using assessment labels for
that decision. Three matched heads receive192 frozen Raw coordinates and64
masked physiological slots. Their90 logical fits cover Raw-only, selected and
original7, with two fixed head initialization seeds. Identical masks share
fits. The15 encoders remain frozen; no new Raw inference/extraction/training
or original Development/Outer access occurs.

The exact model, loss, partition/preprocessing, calibration rules, seed scope,
cost and prospective allocation/stop gate are in `PROTOCOL_LOCK.json`.
`run_checked.ps1` holds an exclusive coordinator lock and foreground owner,
preserves uniquely named logs and runs training then sealed-score replay/
report. Keep the SSH owner alive. On failure inspect process/log/checkpoint
state; do not start duplicates or alter the protocol. Private artifacts stay
on the server. Public records contain no patient or channel identities.

`attribute_audit.py` independently examines existing public selections at
exact-feature, measurement-attribute and coarse-family granularity. That is
a post hoc descriptive sensitivity to definition grouping. It neither merges
model inputs nor proves interchangeability of measurements or changes a
selected set.

No positive result is assumed and no further full model is automatically
launched. Source and protocol are pushed before any new readout fitting.

## Actual outcome and terminal reproduction

All90 small readouts completed928 epochs. Every sealed logit, calibration
threshold and epoch was replayed;1530 patient metric rows were independently
checked. The prospective allocation gate failed. No full Raw/M8 model or
new Outer is launched. See [REPORT.md](REPORT.md) and the complete anonymous
tables under `results/readout/`.

The original `run_checked.ps1` records the historical execution. Its initial
report stage stopped on exact replay because a legacy import reset PyTorch
threads. `diagnose_replay.py` established the setting mismatch without reading
assessment metrics. `replay_one_thread.py` restores the declared production
one-thread setting after that import, leaving every frozen source and output
unchanged. Use that wrapper for explicit report replay, not a training restart:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
& 'E:\Anaconda\envs\persist_stable_251\python.exe' -u `
  'D:\m8_conditional_readout_validation_seed42_v1\replay_one_thread.py'
```

Keep the foreground SSH owner alive, use new log filenames and inspect live
workers first. `run_readouts.py` refuses to refit an already sealed experiment.
The public-only `verify_release.py` needs NumPy/pandas, verifies transferred
bytes and published arithmetic, and does not require private data or models.
