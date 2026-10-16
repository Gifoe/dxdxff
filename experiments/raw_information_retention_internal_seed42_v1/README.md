# Raw information retention: internal-only study

This follows the original Raw-only EZ-F1>0.65 goal without changing its metric.
The completed parent study is retained byte-for-byte. Its models, protocol,
prediction seal and repeated-test Outer report are not amended or reused as
training/selection inputs. This registration makes **no new Outer pass**.

The next hypotheses follow from the input pipeline and model source:

- Per-window variance normalization removes positive waveform amplitude changes
  above its existing SD floor. Test one same-channel pre-onset waveform scale,
  used only in normalization, never appended as a physiological descriptor.
- The original CNN has33-sample local nonlinear interactions before global
  averaging. Compare a small residual temporal CNN spanning535samples before
  global averaging. Both still consume the same whole500-sample windows.
- A zero-initialized latent-only patient conditioner tests whether a learned
  distribution of Raw channel embeddings helps the same single classifier.
- Four separately trained deletion controls test pre-latent reference, temporal
  phases, between-seizure SD and full-patient Z on an otherwise matched baseline.

All weights start from scratch. No manual feature/D1 tensor, trained encoder,
new waveform cache, channel sampling or window sampling is used. Every fit uses
only its own original outer-FIT patient subset; original Development and Outer
are excluded. The inherited inner role arrays are metadata, not model inputs.

The fixed budget is45 first-path cells (9 arms x5 folds), plus at most20 raced
cells. Each own-FIT chooses its own eligible candidate; no global architecture
winner is selected from overlapping folds. Held scores used for racing are
selection data. Repeated-source and patient-ID cluster intervals are descriptive,
not independent validation or evidence that the final Outer target was reached.
All data support/threshold/training/racing rules are in `PROTOCOL_LOCK.json`.

## Execution revision before formal training

The initial gradient admission failed before any formal epoch. The original
failure, direct self-replay noise and failed cross-backend gradient comparison
remain in `results/failed_execution_v1/`. They are not relabeled as passed.
`EXECUTION_AMENDMENT.json` fixes a separate deterministic-cuDNN execution for
all arms and reverse chunk replay matching direct autograd. The original
atol1e-6/rtol1e-5 gradient gate is unchanged. This is not claimed to be a
same-backend numerically equivalent continuation of the failed version.
All9 complete-patient gates must pass anew before any fitting. Original audit,
source and logs are retained; formal epochs discarded=0. The current runtime
is a separate `deterministic_execution_v2` subdirectory.

## Trusted-server execution

The existing waveform registry must be present; do not copy/reextract Raw data.

```powershell
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1'
E:\Anaconda\python.exe D:\raw_information_retention_internal_seed42_v1\run_tests_checked.py
powershell -NoProfile -File D:\raw_information_retention_internal_seed42_v1\run_checked.ps1 -RunTag 01 -StopAfter All
```

Keep the foreground SSH owner alive. Launchers hold an exclusive coordinator
lock and require independent completion gates, not just a process exit code.
After a diagnosed engineering failure, preserve all source/log/checkpoint files
and resume the same frozen run under a new RunTag. Never duplicate a live worker.
Only source and compact anonymous executed summaries are published; individual
identities, waveforms, normalization coefficients, weights and scores stay on
the trusted server. Unrun cells are never filled or represented as results.

## Completed internal results

The study completed **53 cells / 773 epochs**, including the45 balanced first-
path cells and8 registered additional-path cells in two original FITs. The
independent reconstruction checked908 private patient result rows and replayed
every calibration score with maximum error0. The first-path comparison has86
patient appearances per arm, involving59 unique held patients. New Outer
passes=0. Frozen v2 scientific inputs/source and the protected parent test seal
remain unchanged.

The highest descriptive candidate EZ-F1 was0.404191, versus C0's0.395678;
its paired gain0.008513 had95% interval[-0.016941,0.035089]. Additional raced
paths gave paired EZ-F1 gain0.001221[-0.032062,0.036047]. These are internal
selection-conditioned results, not evidence that the original>0.65 goal was
reached. See [REPORT.md](REPORT.md) for all nine metrics/arms, end-to-end
component contrasts, source/fold summaries, joint normalization-by-encoder
interaction, and the explicitly non-deployable oracle-threshold diagnostic.

### Reporting reproduction

After the original independent internal terminal, trusted-server reporting runs:

```powershell
E:\Anaconda\python.exe D:\raw_information_retention_internal_seed42_v1\finalize_reporting.py
E:\Anaconda\python.exe D:\raw_information_retention_internal_seed42_v1\diagnose_threshold_ceiling.py
```

These only aggregate already saved scores. They neither fit models nor change
thresholds. The second command adds a retrospective, label-aware threshold
ceiling diagnostic, not a legal model result or original preregistered endpoint.
`PUBLIC_EXPORT_MANIFEST.json` allowlists21 compact aggregate artifacts. Preserve
existing local files when transferring; verify identical hashes and skip them,
never overwrite an existing file merely to repeat a transfer.

With those anonymous artifacts under `results/deterministic_execution_v2`:

```powershell
python make_report.py --results results/deterministic_execution_v2 --output REPORT.md
python plot_results.py --results results/deterministic_execution_v2 --output figures
```

The plotting source produces editable SVG/PDF and600dpi PNG/TIFF. The released
figure QA records actual visual inspection; a fresh render returns its QA to
pending until inspected. Input/admission failures remain in the technical
appendix rather than being presented as classification performance.
